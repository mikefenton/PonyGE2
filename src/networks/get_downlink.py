import numpy as np


def get_proportional_fairness_downlink(self):
    """Gets the downlink rates for UEs using Shannon's formula and
       congestion information. Uses proportional fairness to divide
       available bandwidth up amongst scheduled UEs.
       """
    
    for small in self.small_cells:
        SC_attached_users = small['attached_users']
        schedule = small['schedule'][self.subframe]
        avg_downlink = []
        if schedule:
            sched = []
            for ind in schedule:
                user = self.users[ind]
                SINR = user['SINR_frame'][self.subframe]
                if (self.subframe == 0) and not user[
                    'previous_downlink_frame']:
                    # No proportional fairness because we don't know how
                    # everyone will perform yet. Just split the bandwidth
                    # evenly across the spectrum.
                    user['proportion'][self.subframe] = 1
                else:
                    # We can implement proportional fairness. We want to
                    # find out who needs the most bandwidth, and approtion
                    # it out appropriately.
                    r_1 = float(self.bandwidth) * log((1 + SINR), 2)
                    # Instantaneous downlink (no bandwidth split)
                    if user['previous_downlink_frame']:
                        length = 40 + len(
                            user['downlink_frame'][:self.subframe - 1])
                        r_2 = (sum(user['previous_downlink_frame']) + sum(
                            user['downlink_frame'][
                            :self.subframe - 1])) / length
                        if r_2 == 0:
                            r_2 = 1
                    else:
                        if self.subframe != 0:
                            r_2 = sum(
                                user['downlink_frame'][:self.subframe]) / len(
                                user['downlink_frame'][:self.subframe])
                            if r_2 == 0:
                                r_2 = 1
                    user['proportion'][self.subframe] = r_1 / r_2
                sched.append(user['proportion'][self.subframe])
            for ind in schedule:
                user = self.users[ind]
                SINR = user['SINR_frame'][self.subframe]
                proportion = user['proportion'][self.subframe]
                downlink = proportion * (
                float(self.bandwidth) / sum(sched)) * log((1 + SINR), 2)
                user['downlink_frame'][self.subframe] = downlink
    
    for i, macro in enumerate(self.macro_cells):
        MC_attached_users = macro['attached_users']
        schedule = macro['schedule'][self.subframe]
        avg_downlink = []
        if schedule:
            sched = []
            for ind in schedule:
                user = self.users[ind]
                SINR = user['SINR_frame'][self.subframe]
                if (self.subframe == 0) and not user[
                    'previous_downlink_frame']:
                    # No proportional fairness because we don't know how
                    # everyone will perform yet. Just split the bandwidth
                    # evenly across the spectrum.
                    user['proportion'][self.subframe] = 1
                else:
                    # We can implement proportional fairness. We want to
                    # find out who needs the most bandwidth, and approtion
                    # it out appropriately.
                    r_1 = float(self.bandwidth) * log((1 + SINR), 2)
                    # Instantaneous downlink (no bandwidth split)
                    if user['previous_downlink_frame']:
                        length = 40 + len(
                            user['downlink_frame'][:self.subframe - 1])
                        r_2 = (sum(user['previous_downlink_frame']) + sum(
                            user['downlink_frame'][
                            :self.subframe - 1])) / length
                    else:
                        if self.subframe != 0:
                            r_2 = sum(
                                user['downlink_frame'][:self.subframe]) / len(
                                user['downlink_frame'][:self.subframe])
                            if r_2 == 0:
                                r_2 = 1
                    user['proportion'][self.subframe] = r_1 / r_2
                sched.append(user['proportion'][self.subframe])
            for ind in schedule:
                user = self.users[ind]
                SINR = user['SINR_frame'][self.subframe]
                if SINR:
                    proportion = user['proportion'][self.subframe]
                    downlink = proportion * (
                    self.bandwidth / sum(sched)) * log((1 + SINR), 2)
                user['downlink_frame'][self.subframe] = downlink
    
    self.received_downlinks = np.zeros(shape=(40, self.n_users))
    for user_id, user in enumerate(self.users):
        self.received_downlinks[:, user_id] = np.array(
            self.users[user_id]['downlink_frame'])

    return self


def get_basic_downlink(self):
    """ Gets the downlink rates for UEs using Shannon's formula and
        congestion information. Divides the bandwidth equally between UEs.
       """
    
    # Instantaneous_downlinks is a num_SFs*num_users matrix storing
    # downlinks received by each UE if no congestion and if each UE was
    # always scheduled.
    instantaneous_downlinks = self.bandwidth * np.log2(1 + self.SINR_SF_UE_act)
    
    # Next we take account of the scheduling by dividing the bandwidth by
    # the number sharing each SF.
    self.received_downlinks = instantaneous_downlinks * self.schedule_info1
    
    # our method yields infinities and nans which are summarily mapped to 0
    self.received_downlinks[np.isnan(self.received_downlinks)] = 0
    self.received_downlinks[self.received_downlinks >= 1E308] = 0
    
    # for user in range(self.n_users):
    #     self.users[user]['downlink_frame'] = self.received_downlinks[:, user]

    return self


def get_new_downlink(self):
    """ Gets the downlink rates for UEs using Shannon's formula and
        congestion information. Uses a new thing to divide
        available bandwidth up amongst scheduled UEs. The SINR of every UE
        attached to a cell for a particular subframe is analysed. Each UE
        is then given an amount of the available bandwidth which is
        inversly proportional to the SINR of that UE compared to other UEs
        in that subframe. I.e., if there are three UEs with SINR of 5 and
        one with an SINR of 85, then the UE with the better SINR is given
        proportionally less bandwidth than the others. In this case, the
        UE with an SINR of 85 would get 5%  of the available bandwidth,
        whereas the UEs with an SINR of 5 would each get 31.666%  of the
        available bandwidth. The idea is to try to give everyone equal
        downlink rates for a particular subframe.

        This might not necessarily work; initial tests seem to show it
        performing worse than proportional fairness. Maybe we should try to
        assign people bandwidth based on their potential downlink
        throughputs, rather than the proportions based on SINR.

       """
    
    for small in self.small_cells:
        SC_attached_users = small['attached_users']
        schedule = small['schedule'][self.subframe]
        avg_downlink = []
        if schedule:
            
            total_SINR = 0
            total_proportion = 0
            for ind in schedule:
                user = self.users[ind]
                SINR = user['SINR_frame'][self.subframe]
                total_SINR += SINR
            for ind in schedule:
                user = self.users[ind]
                SINR = user['SINR_frame'][self.subframe]
                if SINR == total_SINR:
                    proportion = 1
                else:
                    proportion = 1 - (SINR / total_SINR)
                total_proportion += proportion
                user['proportion'][self.subframe] = proportion
            for ind in schedule:
                user = self.users[ind]
                SINR = user['SINR_frame'][self.subframe]
                proportion = user['proportion'][self.subframe]
                downlink = (
                           proportion / total_proportion) * self.bandwidth * log(
                    (1 + SINR), 2)
                user['downlink_frame'][self.subframe] = downlink
    
    for macro in self.macro_cells:
        MC_attached_users = macro['attached_users']
        schedule = macro['schedule'][self.subframe]
        avg_downlink = []
        if schedule:
            total_SINR = 0
            total_proportion = 0
            for ind in schedule:
                user = self.users[ind]
                SINR = user['SINR_frame'][self.subframe]
                total_SINR += SINR
            for ind in schedule:
                user = self.users[ind]
                SINR = user['SINR_frame'][self.subframe]
                if SINR == total_SINR:
                    proportion = 1
                else:
                    proportion = 1 - (SINR / total_SINR)
                total_proportion += proportion
                user['proportion'][self.subframe] = proportion
            for ind in schedule:
                user = self.users[ind]
                SINR = user['SINR_frame'][self.subframe]
                proportion = user['proportion'][self.subframe]
                downlink = (
                           proportion / total_proportion) * self.bandwidth * log(
                    (1 + SINR), 2)
                user['downlink_frame'][self.subframe] = downlink

    return self
