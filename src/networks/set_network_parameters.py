import numpy as np
import numpy.ma as ma

from algorithm.parameters import params
from networks.reset import clear_memory
from networks.scheduling import set_scheduling


def set_users(self):
    """Sets parameters for all users"""
    
    self.SC_attached_UEs = []
    # store the cell IDs and powers from best serving cells (MC and SCs)
    # for each UE
    
    if params['REALISTIC'] and params['DIFFICULTY'] == 1:
        from bottleneck import partsort
        cell_powers_tp = self.cgm.transpose()
        del (self.cgm)
    macro_attachment_id = np.argmax(
        self.all_cell_powers[:self.n_macro_cells, :], axis=0)
    macro_received_power = self.all_cell_powers[
        macro_attachment_id, list(range(self.n_users))]
    small_attachment_id = np.argmax(
        self.cell_attachment[self.n_macro_cells:, :], axis=0)
    small_received_power = self.all_cell_powers[
        small_attachment_id + self.n_macro_cells, list(range(self.n_users))]
    
    for i, user in enumerate(self.users):
        # Need to find the top N strongest cells per UE
        if params['REALISTIC'] and params['DIFFICULTY'] == 1:
            cell_powers_tp[i][cell_powers_tp[i] < -
            partsort(-cell_powers_tp[i], self.SINR_interference_limit)[
            :self.SINR_interference_limit][-1]] = np.nan
        user['attachment'] = 'None'
        user['extended'] = False
        user['SINR_frame'] = [0 for _ in range(40)]
        user['previous_downlink_frame'] = []
        user_id = user['id']
        
        # Small cell operations, add best serving SC
        user['small'] = small_attachment_id[user_id]
        user['small_received_power'] = small_received_power[user_id]
        small = self.small_cells[user['small']]
        small['potential_users'].append(user['id'])
        
        # Macro cell operations, add best serving MC
        user['macro'] = macro_attachment_id[user_id]
        user['macro_received_power'] = macro_received_power[user_id]
        macro = self.macro_cells[user['macro']]
        macro['potential_users'].append(user['id'])
        
        self = set_user_attachment(self, user, macro, small)
    
    if params['REALISTIC'] and params['DIFFICULTY'] == 1:
        self.channel_gain_matrix = cell_powers_tp.transpose()
    
    self.SC_attached_UEs = np.array(self.SC_attached_UEs)
    
    return self


def set_user_attachment(self, user, macro, small):
    """Set the attachment for a user to the cell which provides the best
       signal strength, but also which includes a sufficient Cell Selection
       Bias."""
    
    m_down = user['macro_received_power']
    s_down = user['small_received_power']
    
    if m_down > (s_down + small['bias']):
        user['attachment'] = 'macro'
        self.MC_UES.append(user['id'])
        user['attachment_id'] = macro['id']
        self.SC_attached_UEs.append(False)
        if user['id'] not in macro['attached_users']:
            macro['attached_users'].append(user['id'])
        else:
            print("attempting duplicate MC user")
    else:
        user['attachment'] = 'small'
        self.SC_UES.append(user['id'])
        self.SC_attached_UEs.append(True)
        user['attachment_id'] = self.n_macro_cells + small['id']
        if user['id'] not in small['attached_users']:
            small['attached_users'].append(user['id'])
        else:
            print("attempting duplicate SC user")
        if m_down > s_down:
            small['extended_users'].append(user['id'])
            user['extended'] = True
            macro['ABS_MSUEs'] += 1
    
    return self


def get_ABS(self):
    """gets the ABS ratio for a MC. Adjusts the downlink rates for MCs and
    SCs to suit."""
    
    self.cumulatvie_ABS_frames = np.asarray([0 for _ in range(40)])
    
    ABS_MSUEs = np.array(
        [self.macro_cells[j]['ABS_MSUEs'] for j in range(self.n_macro_cells)])
    ABS_MUEs = np.array([len(self.macro_cells[j]['attached_users']) for j in
                         range(self.n_macro_cells)])
    non_ABS_MUEs = ABS_MUEs + ABS_MSUEs
    alpha = 1  # 1/(1-(ABS_MSUEs/non_ABS_MUEs))
    
    numbers = np.asarray([params['min_ABS_ratio'] for _ in range(
        self.n_macro_cells)])  # ABS_MUEs/(ABS_MUEs + ABS_MSUEs)
    
    if self.BENCHMARK_ABS:
        # print "Benchmark ABS"
        numbers = (1 - alpha) + (alpha * ABS_MSUEs / non_ABS_MUEs)
        numbers = np.round(numbers / 0.125) / 8
        numbers[numbers >= 1] = params['min_ABS_ratio']
        numbers[numbers <= 0] = 0.125
        numbers[np.where(ABS_MUEs == 0)[0]] = 0.125
        numbers = 1 - numbers
    
    elif self.ABS_algorithm:
        # print "Evolved ABS"
        numbers = eval(self.ABS_algorithm)
        if type(numbers) == float:
            numbers = np.array([numbers for _ in range(self.n_macro_cells)])
        numbers = np.round(numbers / 0.125) / 8
        numbers[numbers >= 1] = params['min_ABS_ratio']
        numbers[numbers <= 0] = 0.125
        numbers[np.where(ABS_MUEs == 0)[0]] = 0.125
    
    else:
        numbers = np.round(numbers / 0.125) / 8
        numbers[numbers >= 1] = params['min_ABS_ratio']
        numbers[numbers <= 0] = 0.125
        numbers[np.where(ABS_MUEs == 0)[0]] = 0.125
    
    if params['SYNCHRONOUS_ABS']:
        new_ratios = int(round(np.average(numbers) * 8)) / 8
        numbers = [new_ratios for _ in range(self.n_macro_cells)]
    
    for id, macro in enumerate(self.macro_cells):
        macro['ABS_ratio'] = numbers[id]
        macro['ABS_pattern'] = np.array([1 for _ in range(40)])
        for i in range(int(round(round((1 - macro['ABS_ratio']), 3) / 0.025))):
            macro['ABS_pattern'][(i * 8) % 39] = 0
        macro['ABS_pattern'] = np.asarray(macro['ABS_pattern'])
        self.cumulatvie_ABS_frames += (-macro['ABS_pattern'] + 1)
        
    return self


def set_SINR(self):
    """ Set all the SINR values for all UEs in the network for a full frame
    """
    
    # convert to linear
    # Actual
    signal_W = (10 ** ((self.all_cell_powers - 30) / 10))
    signal_W[np.isnan(signal_W)] = 0
    del (self.all_cell_powers)
    
    if params['REALISTIC'] and params['DIFFICULTY'] == 1:
        # Estimtated
        cgm_W = 10 ** ((self.channel_gain_matrix - 30) / 10)
        cgm_W[np.isnan(cgm_W)] = 0
        del (self.channel_gain_matrix)
    
    # encode the ABS patterns in a num_SFs*num_cells matrix, i.e. each row
    # i indicates which cells mute in SF i
    small_ABS = [1 for _ in range(self.n_small_cells)]
    self.ABS_activity = np.array(
        [[[macro['ABS_pattern'][i] for macro in self.macro_cells] + small_ABS]
         for i in range(40)])[:, 0, :]
    
    # Actual
    # signal_W is a num_cells*num_users matrix until this point, we
    # broadcast it with the ABS information to give a
    # num_SFs*num_cells*num_users matrix. From this we can get the SINRs
    # across all SFs for each UE as follows...
    tiled_signal_W = np.tile(signal_W, (40, 1, 1))
    
    if params['REALISTIC'] and params['DIFFICULTY'] == 1:
        # Estimtated
        # Also have to take the channel gain matrix and apply the same
        # operations in order to map it to the ABS pattern. Channel gain matrix
        # is limited to the top N interfering cells
        full_c_g_m = np.tile(cgm_W, (40, 1, 1))
    
    indices_zeros = np.where(self.ABS_activity == 0)
    
    # turn off the muted cells in each SF. Achieved by setting the power
    # received by each UE to 0 from the muted cell
    # Actual
    tiled_signal_W[indices_zeros[0], indices_zeros[1], :] = 0
    if params['REALISTIC'] and params['DIFFICULTY'] == 1:
        # Estimtated
        full_c_g_m[indices_zeros[0], indices_zeros[1], :] = 0
    
    # store the id of the cell each UE attaches to in a num_users vector
    attached_cells = [self.users[i]['attachment_id'] for i in
                      range(self.n_users)]
    # divide the received_powers by (interference + noise) where these are
    # num_SFs*num_users matrices
    received_powers = tiled_signal_W[:, attached_cells,
                      list(range(len(attached_cells)))]
    actual_interference = np.sum(tiled_signal_W, axis=1)
    self.SINR_SF_UE_act = received_powers / (
    (actual_interference - received_powers) + self.noise_W)
    
    l1, l2 = params['SINR_limits'][0], params['SINR_limits'][1]
    
    if params['REALISTIC']:
        # Need to limit ACTUAL SINR values to max 23 db (199.5262315 in
        # linear).
        self.SINR_SF_UE_act = np.clip(self.SINR_SF_UE_act, 0, 10 ** (l2 / 10))
    
    # Calculate CQI data.
    self.CQI = np.zeros(shape=(40, self.n_users))
    
    for idx, small in enumerate(self.small_cells):
        small['macro'] = self.SC_interferers[idx]
        # Need to find the strongest interfering/serving MC for
        # this SC. The ABS pattern of this MC will determine the
        # reported SINR values of the SC attached UEs.
        small['ABS_pattern'] = self.macro_cells[small['macro']]['ABS_pattern']
    
    for user in self.users:
        ind = user['id']
        
        if user['attachment'] == "small":
            small = self.small_cells[user['small']]
            governing_ABS = small['ABS_pattern']
            ABSclass = np.invert(governing_ABS.astype(bool)).astype(int)
            nonABSclass = governing_ABS
            a = self.SINR_SF_UE_act[:, ind] * ABSclass
            b = self.SINR_SF_UE_act[:, ind] * nonABSclass
            
            ABS_SINR = 10 ** (
            np.clip(np.around(10 * np.log10(np.mean(a[np.nonzero(a)]))), l1,
                    l2) / 10)
            user['ABS_SINR'] = ABS_SINR
            non_ABS_SINR = 10 ** (
            np.clip(np.around(10 * np.log10(np.mean(b[np.nonzero(b)]))), l1,
                    l2) / 10)
            user['non_ABS_SINR'] = non_ABS_SINR
            if params['DIFFICULTY'] == 2:
                averaged = self.SINR_SF_UE_act[:, ind]
            elif params['DIFFICULTY'] == 3:
                averaged = (ABSclass * ABS_SINR) + (nonABSclass * non_ABS_SINR)
        else:
            a = self.SINR_SF_UE_act[:, ind]
            if params['DIFFICULTY'] == 2:
                averaged = a
            elif params['DIFFICULTY'] == 3:
                macro = self.macro_cells[user['macro']]
                averaged = np.mean(a[np.nonzero(a)]) * macro['ABS_pattern']
        
        # Need to convert from linear to logarithmic (db)
        averaged_db = 10 * np.log10(averaged)
        
        # Need to quantize and clip our logarithmic SINR values
        quantized_db = np.around(averaged_db)
        self.CQI[:, ind] = np.clip(quantized_db, l1, l2)
    
    if params['REALISTIC']:
        
        # Need to convert back to linear from logarithmic (db)
        self.SINR_SF_UE_est = (10 ** ((self.CQI) / 10))
        del (self.CQI)
        
        if params['DIFFICULTY'] == 1:
            received_powers_cgm = full_c_g_m[:, attached_cells,
                                  list(range(len(attached_cells)))]
            reported_interference = np.sum(full_c_g_m, axis=1)
            
            # finally we get the SINR for each UE in each SF in a
            # num_SFs*num_users matrix
            self.SINR_SF_UE_est = received_powers_cgm / (
            (reported_interference - received_powers_cgm) + self.noise_W)
        
        # Need to limit reported SINR values to max 23 db / (199.5262315 in
        # linear).
        self.SINR_SF_UE_est = np.clip(self.SINR_SF_UE_est, 0, 10 ** (l2 / 10))
    
    else:
        self.SINR_SF_UE_est = self.SINR_SF_UE_act
    
    # extract statistics for each UE like max SINR etc, we'll need this
    # info when scheduling UEs
    self.max_SINR_over_frame = np.max(self.SINR_SF_UE_est, axis=0)
    self.min_SINR_over_frame = np.min(
        ma.masked_where(self.SINR_SF_UE_est == 0, self.SINR_SF_UE_est),
        axis=0).data
    self.avg_SINR_over_frame = np.average(self.SINR_SF_UE_est, axis=0)
    self.potential_slots = self.SINR_SF_UE_est >= self.SINR_limit
    self.good_subframes_over_frame = np.sum(self.potential_slots, axis=0)
    self.helpless_UEs = self.max_SINR_over_frame <= self.SINR_limit

    return self


def update_network(self):
    """
    Sets the user attachment and ABS configurations for a network of given
    powers and biases. Sets SINR values.
    
    :param self:
    :param FIST:
    :return:
    """
    
    self = clear_memory(self)
    self.small_powers = [self.small_cells[j]['power'] for j in
                         range(self.n_small_cells)]
    self.small_biases = [self.small_cells[j]['bias'] for j in
                         range(self.n_small_cells)]
    self.power_bias = [self.small_powers, self.small_biases]
    self.all_powers = [self.macro_cells[0]['power'] for _ in range(
        self.n_macro_cells)] + self.small_powers
    
    # get the locations of each user and extract only the columns of the
    # gain matrix for the UEs on the map, add the powers to get received
    # signal with pathloss. Also get the strongest MC interferers per SC.
    self.UE_locations_x = [self.users[i]['location'][0] for i in
                           range(self.n_users)]
    self.UE_locations_y = [self.users[i]['location'][1] for i in
                           range(self.n_users)]
    self.SC_locations_x = [self.small_cells[i]['location'][0] for i in
                           range(self.n_small_cells)]
    self.SC_locations_y = [self.small_cells[i]['location'][1] for i in
                           range(self.n_small_cells)]
    self.all_cell_powers = self.gains[:, self.UE_locations_y,
                           self.UE_locations_x] + np.array(self.all_powers)[:,
                                                  np.newaxis]
    self.all_cell_bias = np.asarray(
        [0 for i in range(self.n_macro_cells)] + self.small_biases)
    self.cell_attachment = self.all_cell_powers + self.all_cell_bias[:,
                                                  np.newaxis]
    
    if params['REALISTIC'] and params['DIFFICULTY'] == 1:
        # We need to quantize the received channel gains to the nearest 3
        # dBm
        quantized_cgm = self.gains[:, self.UE_locations_y, self.UE_locations_x]
        # Round off gains to nearest 3 dBm
        quantized_cgm = np.around(quantized_cgm / 3) * 3
        self.cgm = quantized_cgm + np.array(self.all_powers)[:, np.newaxis]
        
        # Set the lower limit for received powers to -123.4 dBm
        self.cgm[self.cgm <= -123.4] = np.nan
    
    # Need to find the strongest serving MC for each SC. Dictates ABS
    # ratios for SC attached UEs
    SC_interferers = self.gains[:, self.SC_locations_y,
                     self.SC_locations_x] + np.array(self.all_powers)[:,
                                            np.newaxis]
    self.SC_interferers = np.argmax(SC_interferers[:self.n_macro_cells,
                                    :], axis=0)
    del (self.SC_locations_x)
    del (self.SC_locations_y)
    del (self.UE_locations_x)
    del (self.UE_locations_y)
    del (self.all_powers)
    
    self = set_users(self)
    self = get_ABS(self)
    self = set_SINR(self)
    if not self.PRE_COMPUTE:
        self.scheduling_decisions, self = set_scheduling(self)
    
    return self
