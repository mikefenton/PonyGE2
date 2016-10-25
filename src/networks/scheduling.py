import numpy as np
from operator import itemgetter
from copy import copy, deepcopy
import scipy.io as io

from algorithm.parameters import params
from utilities.fitness.math_functions import pdiv


def set_scheduling(self, METHOD="baseline"):
    """ Compute per-subframe scheduling for each UE in the network """

    scheduling_decisions = np.ones((40, params['N_USERS']), dtype=bool)
    
    if METHOD != "baseline":
        # this all gives scheduling decisions for MCs too but we just don't
        # use these decisions

        if METHOD == "evolved":
            scheduling_decisions = compute_scheduling(self)
        elif METHOD == "benchmark":
            scheduling_decisions = benchmark(self)
                    
        if type(scheduling_decisions) is bool or type(
                scheduling_decisions) is np.bool_:
            if scheduling_decisions:
                scheduling_decisions = np.ones(shape=(40, params['N_USERS']),
                                               dtype=bool)
            else:
                scheduling_decisions = np.zeros(shape=(40, params['N_USERS']),
                                                dtype=bool)
    
    scheduling_decisions[self.SINR_SF_UE_est <= self.SINR_limit] = False
    
    if params['REALISTIC']:
        # Need to model dropped data transmissions due to actual
        # SINR <= self.SINR_limit
        
        check = np.logical_and((self.SINR_SF_UE_act <= self.SINR_limit),
                               scheduling_decisions)
        available = np.logical_and(self.potential_slots,
                                   np.logical_not(scheduling_decisions))
        
        copied_dropped = deepcopy(check)
        for sf, lookup in enumerate(copied_dropped):
            if any(lookup):
                available_copy = deepcopy(available)
                boolean = np.zeros((40, self.n_users))
                # Only operate on those UEs who have dropped calls
                boolean[min((sf + 4), 39):] = lookup
                # Find free permissible slots in which these UEs can be
                # rescheduled (that we know of)
                slots = np.logical_and(available_copy, boolean)
                if np.any(slots):
                    # Find array indices of available slots
                    indices = np.array(np.where(slots)).transpose().tolist()
                    # Sort the indices by UE
                    indices.sort(key=itemgetter(1))
                    # Get UE ids
                    troubled_UEs = np.where(lookup)[0]
                    # Find which UEs CAN be rescheduled
                    lucky_UEs = list(
                        set([i[1] for i in indices if i[1] in troubled_UEs]))
                    for user in lucky_UEs:
                        new_slot = [list(i) for i in indices if i[1] == user][
                            0]
                        available[new_slot[0]][new_slot[1]] = False
                        if self.SINR_SF_UE_act[new_slot[0]][
                            new_slot[1]] >= self.SINR_limit:
                            # Reschedule the UE in their new slot
                            scheduling_decisions[new_slot[0]][
                                new_slot[1]] = True
                        else:
                            # The slot we had thought was ok is actually
                            # not. The call will be dropped again.
                            copied_dropped[new_slot[0]][new_slot[1]] = True
        
        scheduling_decisions[self.SINR_SF_UE_act <= self.SINR_limit] = False
        self.potential_slots[self.SINR_SF_UE_act <= self.SINR_limit] = False
    
    if params['SAVE']:
        name = None
        if self.BASELINE_SCHEDULING:
            name = "Baseline"
        elif self.BENCHMARK_SCHEDULING:
            name = "Benchmark"
        elif self.OPT_SCHEDULING:
            name = "GA"
        elif self.NEW_SCHEDULING:
            name = "New_" + "_".join(str(self.weight).split("."))
        elif self.EVOLVED_SCHEDULING:
            name = "Evolved"
        if name:
            for small in [cell for cell in self.small_cells if
                          len(cell['attached_users']) == 10]:
                unsorted_ids = np.array(small['attached_users'])
                ids = unsorted_ids[
                    np.argsort(self.avg_SINR_over_frame[unsorted_ids])]
                schedule = scheduling_decisions[:, ids]
                dic = {'schedule': schedule[:8, ]}
                io.savemat(params['FILE_PATH'] + "Heatmaps/input/" +
                           name + "_" + str(self.iteration) + "_" +
                           str(small['id']), dic)
    
    for user in self.users:
        ue_id = user['id']
        user['SINR_frame'] = self.SINR_SF_UE_act[:, ue_id]
        user['max_SINR'] = self.max_SINR_over_frame[ue_id]
        user['min_SINR'] = self.min_SINR_over_frame[ue_id]
    
    return scheduling_decisions, self


def benchmark(self):
    """
    Benchmark scheduling, Lopez and Claussen (2013).
    
    :param self: Instance of the network class.
    :return: scheduling decisions.
    """

    scheduling_decisions = np.ones((40, params['N_USERS']), dtype=bool)
    
    for small in [cell for cell in self.small_cells if
                  len(cell['attached_users']) > 1]:
        
        attached = small['attached_users']
        non_ABS = np.array(small['ABS_pattern'])
        ABS = np.array([not i for i in non_ABS], dtype=int)
        num_ABS = np.count_nonzero(ABS[:8])
        num_non_ABS = 8 - num_ABS
        
        ABS_queue = []
        non_ABS_queue = []
        
        def compute_balance(ABS_queue, non_ABS_queue):
            # Step 4: Calculate estimated throughputs for worst UEs
            # in each queue
            
            ABS_sched = [a for a in ABS_queue if
                         a[1] > self.SINR_limit]
            non_ABS_sched = [b for b in non_ABS_queue if
                             b[2] > self.SINR_limit]
            
            if len(ABS_queue) == 0 or len(ABS_sched) == 0:
                return True
            
            elif len(non_ABS_queue) == 0 or len(non_ABS_sched) == 0:
                return False
            
            else:
                
                worst_ABS = ABS_sched[0][1]
                worst_non_ABS = non_ABS_sched[0][2]
                TP_ABS = (self.bandwidth / len(
                    ABS_sched)) * np.log2(
                    1 + worst_ABS) * num_ABS / 8
                TP_no_ABS = (self.bandwidth / len(
                    non_ABS_sched)) * np.log2(
                    1 + worst_non_ABS) * num_non_ABS / 8
                
                if (TP_no_ABS / TP_ABS) > 1:
                    # non ABS UE is better off, need to offload
                    # from ABS queue
                    return False
                
                elif (TP_no_ABS / TP_ABS) < 1:
                    # ABS UE is better off, need to offload from
                    # non ABS queue
                    return True
                
                else:
                    return "finished"
        
        # Steps 1 & 2:
        for ue in attached:
            user = self.users[ue]
            ABS_SINR = user['ABS_SINR']
            non_ABS_SINR = user['non_ABS_SINR']
            if user['extended']:
                ABS_queue.append([ue, ABS_SINR, non_ABS_SINR])
            else:
                non_ABS_queue.append([ue, ABS_SINR, non_ABS_SINR])
                
        # Step 3: Sort each queue with respect to increasing SINR
        # (i.e.) worst performers earliest in the queue
        ABS_queue.sort(key=itemgetter(1))
        non_ABS_queue.sort(key=itemgetter(2))
        
        check_1 = False
        check_2 = True
        
        # If either queue is empty, place one UE from the other
        # queue there
        if (len(ABS_queue) == 0):
            mover = non_ABS_queue.pop(0)
            ABS_queue.append(mover)
            move_to_ABS = True
            check_1 = True
        elif (len(non_ABS_queue) == 0):
            available = [e for e in ABS_queue if
                         e[2] > self.SINR_limit]
            if not available:
                # Then we can't put anyone into the non ABS queue
                # Revert to baseline scheduling
                ABS_queue = [[ue] for ue in attached]
                non_ABS_queue = [[ue] for ue in attached]
                check_2 = False
            else:
                mover = available[-1]
                non_ABS_queue.append(mover)
                ABS_queue.remove(mover)
                move_to_ABS = False
                check_1 = True
        
        if check_2:
            ABS_queue.sort(key=itemgetter(1))
            non_ABS_queue.sort(key=itemgetter(2))
            
            # Place any UEs who can only be scheduled in ABS
            # frames in the ABS queue.
            need_help = [f for f in non_ABS_queue if
                         (f[2] <= self.SINR_limit) and (
                             f[1] > self.SINR_limit)]
            if need_help:
                for ue in need_help:
                    ABS_queue.append(ue)
                    non_ABS_queue.remove(ue)
                    move_to_ABS = True
                    check_1 = True
            
            ABS_queue.sort(key=itemgetter(1))
            non_ABS_queue.sort(key=itemgetter(2))
            
            # If nobody in the ABS queue can be scheduled, revert
            # to baseline scheduling
            if all([g[1] <= self.SINR_limit for g in ABS_queue]):
                ABS_queue = [[ue] for ue in attached]
                non_ABS_queue = [[ue] for ue in attached]
                check_2 = False
        
        if check_2:
            # We need to implement the baseline algorithm
            
            # Initial check for direction move...
            move = compute_balance(ABS_queue, non_ABS_queue)
            if not check_1:
                if move:
                    move_to_ABS = True
                else:
                    move_to_ABS = False
            
            # Iterate until one of the stopping conditions is
            # satisfied then break out of loop
            while True:
                
                if move == "finished":
                    break
                
                # Check to make sure neither queue is empty
                if (len(ABS_queue) == 0):
                    mover = non_ABS_queue.pop(0)
                    ABS_queue.append(mover)
                    if not move_to_ABS:
                        # We previously removed someone from the
                        # ABS queue
                        break
                    move_to_ABS = True
                
                elif (len(non_ABS_queue) == 0):
                    available = [h for h in ABS_queue if
                                 h[2] > self.SINR_limit]
                    if not available:
                        # Then we can't put anyone into the non
                        # ABS queue; Revert to baseline scheduling
                        ABS_queue = [[ue] for ue in attached]
                        non_ABS_queue = [[ue] for ue in attached]
                        break
                    else:
                        mover = available[-1]
                        non_ABS_queue.append(mover)
                        ABS_queue.remove(mover)
                        if move_to_ABS:
                            # We previously placed someone in
                            # the ABS queue
                            break
                        move_to_ABS = False
                
                if move:
                    # Need to move from non-ABS to ABS
                    
                    if not move_to_ABS:
                        # We previously removed someone from the
                        # ABS queue
                        break
                    
                    # If there are too many in the non ABS queue,
                    # take the worst (first in the list) and put
                    # them in the ABS queue
                    mover = non_ABS_queue.pop(0)
                    ABS_queue.append(mover)
                    move_to_ABS = True
                else:
                    # Need to move from ABS to non-ABS
                    
                    if move_to_ABS:
                        # We previously placed someone in the ABS
                        # queue
                        break
                    
                    # If there are too many in the ABS queue, take
                    # the best (last in the list) and put them in
                    # the non-ABS queue, but only if they have an
                    # SINR greater than the lower limit.
                    available = [j for j in ABS_queue if
                                 j[2] > self.SINR_limit]
                    if not available:
                        # Then we can't put more into the non ABS
                        # queue
                        break
                    else:
                        mover = available[-1]
                        non_ABS_queue.append(mover)
                        ABS_queue.remove(mover)
                    move_to_ABS = False
                
                ABS_queue.sort(key=itemgetter(1))
                non_ABS_queue.sort(key=itemgetter(2))
                
                move = compute_balance(ABS_queue, non_ABS_queue)
        
        schedule = np.zeros(shape=(40, self.n_users))
        for ue in attached:
            if ue in [i[0] for i in ABS_queue]:
                schedule[:, ue] = ABS
            elif ue in [i[0] for i in non_ABS_queue]:
                schedule[:, ue] = non_ABS
            elif ue in [i[0] for i in ABS_queue] and ue in [i[0]
                                                            for i
                                                            in
                                                            non_ABS_queue]:
                schedule[:, ue] = [1 for i in range(40)]
        
        inds_schedule = schedule[:, attached]
        inds_schedule[self.SINR_SF_UE_est[:,
                      attached] <= self.SINR_limit] = 0
        
        # Check to make sure every UE is scheduled for at least one
        # SF
        lookup = np.sum(inds_schedule, axis=0) == 0
        inds_schedule[:, lookup] = 1
        
        inds_schedule[self.SINR_SF_UE_est[:,
                      attached] <= self.SINR_limit] = 0
        
        # Check to make sure someone is scheduled in at least every
        # SF
        lookup = np.sum(inds_schedule, axis=1) == 0
        inds_schedule[lookup, :] = 1
        
        scheduling_decisions[:, attached] = inds_schedule
    
    scheduling_decisions[
        self.SINR_SF_UE_est <= self.SINR_limit] = False
    
    return scheduling_decisions


def crosshairs(self):
    """
    New attempt at visuals-based timeframe scheduling by moving crosshairs.
    
    :param self: Instance of the network class.
    :return: scheduling decisions.
    """

    scheduling_decisions = np.ones((40, params['N_USERS']), dtype=bool)

    for small in [cell for cell in self.small_cells if
                  len(cell['attached_users']) > 1]:

        attached = small['attached_users']
        num_attached = len(attached)
        non_ABS = np.array(small['ABS_pattern'])
        ABS = np.array([not i for i in non_ABS], dtype=int)
        num_ABS = np.count_nonzero(ABS[:8])
        UE_rep_mat = np.zeros(shape=(8, self.n_users))
        schedules = np.ones(shape=(8, self.n_users))
        
        weight = self.weight
                
        def calculate_w_SLR(sched):
            
            congestion = np.sum(sched, axis=1)
            congestion = np.repeat(congestion, num_attached)
            congestion = np.reshape(congestion, (8, num_attached))
            rates = self.bandwidth / congestion * UE_inst_mat[:,
                                                  sorted_UEs] * sched
            rates = np.nan_to_num(rates)
            avg_rates = np.average(rates, axis=0)
            log_average_downlinks = np.log(avg_rates)
            num_att = len(avg_rates)
            return np.sum(
                np.sort(log_average_downlinks) * np.array(list(
                    reversed(list(range(1, num_att + 1))))) ** weight)
        
        def compute_direction(ROWS, COLS):
            
            orig = calculate_w_SLR(get_sched(ROWS, COLS))
            COLS -= 1
            new = calculate_w_SLR(get_sched(ROWS, COLS))
            if new > orig:
                return True
            else:
                return False
        
        def get_sched(ROW, COL):
            """Returns a scheduling matrix for a SC given ROWS and COLS."""
            
            sched = np.ones(shape=(8, num_attached))
            sched[:ROW][:, COL:] = 0
            sched[:ROW][:, :COL] = 1
            sched[ROW:][:, :COL] = 0
            sched[ROW:][:, COL:] = 1
            sched[UE_rep_mat[:, sorted_UEs] <= self.SINR_limit] = 0
            return sched
        
        # Step 1: Sort SC attached UEs with respect to increasing
        # SINR (i.e.) worst performers earliest in the queue
        sorted_UEs_arr = []
        for ue in attached:
            user = self.users[ue]
            ABS_SINR = user['ABS_SINR']
            non_ABS_SINR = user['non_ABS_SINR']
            UE_rep_mat[:num_ABS, ue] = ABS_SINR
            UE_rep_mat[num_ABS:, ue] = non_ABS_SINR
            sorted_UEs_arr.append([ue, ABS_SINR, non_ABS_SINR,
                                   np.average(
                                       [ABS_SINR, non_ABS_SINR])])
        sorted_UEs_arr.sort(key=itemgetter(3))
        sorted_UEs = list(
            np.ravel(np.array(sorted_UEs_arr)[:, [0]]))
        UE_inst_mat = np.log2(1 + UE_rep_mat)
        schedules[UE_rep_mat <= self.SINR_limit] = 0
        
        # Step 2: Calculate the minimum number of protected UEs
        min_protected = num_attached - np.count_nonzero(
            schedules[:, sorted_UEs][-1])
        
        # Step 3: Calculate Initial Guess
        max_UEs = UE_rep_mat[:, sorted_UEs][0]
        
        def optimise_COLS(ROW):
            """Optimise number of UEs (COLS) for a given subframe
            (ROWS)."""
            
            # Start with the best UEs in the best subframes
            COL = -max(
                len([i for i in max_UEs if i == max(list(max_UEs))]),
                1)
            cut = compute_direction(ROW, COL)
            previous_cut = cut
            
            while True:
                                
                if cut:
                    if not previous_cut:
                        break
                    previous_cut = True
                else:
                    if previous_cut:
                        break
                    previous_cut = False
                
                if previous_cut:
                    if COL == -num_attached:
                        break
                    COL -= 1
                else:
                    if COL == -1:
                        break
                    COL += 1
                
                cut = compute_direction(ROW, COL)
            return ROW, COL
        
        ROW, COL = optimise_COLS(num_ABS)
        
        def optimise_ROWS(ROW, COL):
            prev = calculate_w_SLR(get_sched(ROW, COL))
            while True:
                if ROW == 7:
                    break
                ROW += 1
                ROW, COL = optimise_COLS(ROW)
                new = calculate_w_SLR(get_sched(ROW, COL))
                if new < prev:
                    ROW -= 1
                    break
                else:
                    prev = new
            return ROW, COL
        
        ROW, COL = optimise_ROWS(ROW, COL)
        
        test_schedules = get_sched(ROW, COL)
        
        inds_schedule = np.vstack((test_schedules, test_schedules,
                                   test_schedules, test_schedules,
                                   test_schedules))
        inds_schedule[self.SINR_SF_UE_est[:,
                      sorted_UEs] <= self.SINR_limit] = 0
        
        # Check to make sure every UE is scheduled for at least one
        # SF
        lookup = np.sum(inds_schedule, axis=0) == 0
        inds_schedule[:, lookup] = 1
        
        inds_schedule[self.SINR_SF_UE_est[:,
                      sorted_UEs] <= self.SINR_limit] = 0
        
        # Check to make sure someone is scheduled in at least every
        # SF
        lookup = np.sum(inds_schedule, axis=1) == 0
        inds_schedule[lookup, :] = 1
        
        scheduling_decisions[:, sorted_UEs] = inds_schedule
    
    scheduling_decisions[
        self.SINR_SF_UE_est <= self.SINR_limit] = False
    
    return scheduling_decisions


def corner_cut(self):
    """
    New attempt at visuals-based timeframe scheduling by setting
    the top right hand diagonal corner of the cell's sorted
    scheduing array to zeros.
    
    :param self: Instance of the network class.
    :return: scheduling decisions.
    """

    scheduling_decisions = np.ones((40, params['N_USERS']), dtype=bool)
    
    for small in [cell for cell in self.small_cells if
                  len(cell['attached_users']) > 1]:
        
        attached = small['attached_users']
        num_attached = len(attached)
        non_ABS = np.array(small['ABS_pattern'])
        ABS = np.array([not i for i in non_ABS], dtype=int)
        num_ABS = np.count_nonzero(ABS[:8])
        UE_rep_mat = np.zeros(shape=(8, self.n_users))
        schedules = np.ones(shape=(8, self.n_users))
        
        weight = self.weight
                
        def calculate_w_SLR(sched):
            
            sched[UE_rep_mat[:, sorted_UEs] <= self.SINR_limit] = 0
            congestion = np.sum(sched, axis=1)
            congestion = np.repeat(congestion, num_attached)
            congestion = np.reshape(congestion, (8, num_attached))
            rates = self.bandwidth / congestion * UE_inst_mat[:,
                                                  sorted_UEs] * sched
            rates = np.nan_to_num(rates)
            avg_rates = np.average(rates, axis=0)
            log_average_downlinks = np.log(avg_rates)
            
            num_att = len(avg_rates)
            return np.sum(
                np.sort(log_average_downlinks) * np.array(list(
                    reversed(list(range(1, num_att + 1))))) ** weight)
        
        # Step 1: Sort SC attached UEs with respect to increasing
        # SINR (i.e.) worst performers earliest in the queue
        sorted_UEs_arr = []
        for ue in attached:
            user = self.users[ue]
            ABS_SINR = user['ABS_SINR']
            non_ABS_SINR = user['non_ABS_SINR']
            UE_rep_mat[:num_ABS, ue] = ABS_SINR
            UE_rep_mat[num_ABS:, ue] = non_ABS_SINR
            sorted_UEs_arr.append([ue, ABS_SINR, non_ABS_SINR,
                                   np.average(
                                       [ABS_SINR, non_ABS_SINR])])
        sorted_UEs_arr.sort(key=itemgetter(3))
        sorted_UEs = list(
            np.ravel(np.array(sorted_UEs_arr)[:, [0]]))
        UE_inst_mat = np.log2(1 + UE_rep_mat)
        schedules[UE_rep_mat <= self.SINR_limit] = 0
        
        # Step 2: Calculate fitness of starting point
        prev_SLR = calculate_w_SLR(schedules[:, sorted_UEs])
        
        # Step 3: Iterative loop
        row = 0
        UE = 0
        
        test_schedules = copy(schedules[:, sorted_UEs])
        
        while True:
            
            if UE == -num_attached:
                print("Crap", -UE, num_attached)
                new_SLR = calculate_w_SLR(test_schedules)
                for sched in test_schedules:
                    print(list(sched))
                quit()
                break
            else:
                UE -= 1
            test_schedules[row][UE:] = 0
            new_SLR = calculate_w_SLR(test_schedules)
            if new_SLR < prev_SLR:
                UE += 1
                test_schedules[row][:UE] = 1
                if row == 6:
                    break
                else:
                    row += 1
                UE = -1
                test_schedules[row][UE] = 0
                new_SLR = calculate_w_SLR(test_schedules)
                if new_SLR < prev_SLR:
                    test_schedules[row][UE] = 1
                    break
                else:
                    prev_SLR = new_SLR
            else:
                prev_SLR = new_SLR
        
        inds_schedule = np.vstack((test_schedules, test_schedules,
                                   test_schedules, test_schedules,
                                   test_schedules))
        inds_schedule[self.SINR_SF_UE_est[:,
                      sorted_UEs] <= self.SINR_limit] = 0
        
        # Check to make sure every UE is scheduled for at least one
        # SF
        lookup = np.sum(inds_schedule, axis=0) == 0
        inds_schedule[:, lookup] = 1
        
        inds_schedule[self.SINR_SF_UE_est[:,
                      sorted_UEs] <= self.SINR_limit] = 0
        
        # Check to make sure someone is scheduled in at least every
        # SF
        lookup = np.sum(inds_schedule, axis=1) == 0
        inds_schedule[lookup, :] = 1
        
        scheduling_decisions[:, sorted_UEs] = inds_schedule
    
    scheduling_decisions[
        self.SINR_SF_UE_est <= self.SINR_limit] = False
    
    return scheduling_decisions


def compute_scheduling(self):
    """ Compute the SC scheduling for a given scenario."""
        
    scheduling_decisions = np.ones(shape=(40, self.n_users), dtype=bool)
    try:
        compile(self.scheduling_algorithm, '<string>', 'eval')
    except MemoryError:
        return scheduling_decisions
    
    try:
        check = eval(self.scheduling_algorithm)
    except:
        check = None
    
    if check == None:
        
        for small in [cell for cell in self.small_cells if
                      len(cell['attached_users']) > 1]:
            
            # sweep the tree in an order respecting UE average SINR across their full frames
            unsorted_ids = np.array(small['attached_users'])
            attached_ids = unsorted_ids[
                np.argsort(self.avg_SINR_over_frame[unsorted_ids])]
            ids = attached_ids
            
            # get all the required statistics
            mat_SINR = self.SINR_SF_UE_est[:8, ids]
            
            ones = np.ones(shape=mat_SINR.shape)
            num_attached = mat_SINR.shape[1]
            mat_num_shared = ones * num_attached
            mat_good_subframes = ones * (mat_SINR > self.SINR_limit).sum(0)[
                                        None, :]
            
            mat_least_congested_downlinks = np.log2(1 + mat_SINR)
            
            mat_avg_down_F = ones * np.average(mat_least_congested_downlinks,
                                               axis=0)[None, :]
            mat_min_down_F = ones * np.min(mat_least_congested_downlinks,
                                           axis=0)[None, :]
            mat_max_down_F = ones * np.max(mat_least_congested_downlinks,
                                           axis=0)[None, :]
            mat_LPT_down_F = ones * np.percentile(
                mat_least_congested_downlinks, 25, axis=0)[None, :]
            mat_UPT_down_F = ones * np.percentile(
                mat_least_congested_downlinks, 75, axis=0)[None, :]
            
            mat_avg_down_SF = ones * np.average(mat_least_congested_downlinks,
                                                axis=1)[:, None]
            mat_min_down_SF = ones * np.min(mat_least_congested_downlinks,
                                            axis=1)[:, None]
            mat_max_down_SF = ones * np.max(mat_least_congested_downlinks,
                                            axis=1)[:, None]
            mat_LPT_down_SF = ones * np.percentile(
                mat_least_congested_downlinks, 25, axis=1)[:, None]
            mat_UPT_down_SF = ones * np.percentile(
                mat_least_congested_downlinks, 75, axis=1)[:, None]
            
            mat_avg_down_cell = ones * np.average(
                mat_least_congested_downlinks)
            mat_min_down_cell = ones * np.min(mat_least_congested_downlinks)
            mat_max_down_cell = ones * np.max(mat_least_congested_downlinks)
            mat_LPT_down_cell = ones * np.percentile(
                mat_least_congested_downlinks, 25)
            mat_UPT_down_cell = ones * np.percentile(
                mat_least_congested_downlinks, 75)
            
            ues = np.array(list(range(num_attached)))
            ue_ids = np.tile(ues, (8, 1))
            
            sfs = np.array(list(range(8)))
            sf_ids = np.tile(sfs, (num_attached, 1)).T
            
            dropped_calls = deepcopy(mat_SINR)
            dropped_calls[dropped_calls <= self.SINR_limit] = -1.0
            dropped_calls[dropped_calls > self.SINR_limit] = 1.0
            
            T1 = mat_num_shared
            T2 = mat_good_subframes
            
            T3 = mat_least_congested_downlinks
            
            T4 = mat_avg_down_F
            T5 = mat_min_down_F
            T6 = mat_max_down_F
            T7 = mat_LPT_down_F
            T8 = mat_UPT_down_F
            
            T9 = mat_avg_down_SF
            T10 = mat_min_down_SF
            T11 = mat_max_down_SF
            T12 = mat_LPT_down_SF
            T13 = mat_UPT_down_SF
            
            T14 = mat_avg_down_cell
            T15 = mat_min_down_cell
            T16 = mat_max_down_cell
            T17 = mat_LPT_down_cell
            T18 = mat_UPT_down_cell
            
            T19 = ue_ids
            T20 = sf_ids
            T21 = dropped_calls
            
            ABS = np.array([not i for i in small['ABS_pattern'][:8]],
                           dtype=int)
            ABS = np.tile(ABS, (num_attached, 1)).T
            ABS[ABS == 0] = -1
            
            # print np.sqrt(T9)
            
            schedule = eval(self.scheduling_algorithm)
            
            if self.SCHEDULING_TYPE.split("_")[-1] == "threshold":
                schedule[schedule >= 0] = 1
                schedule[schedule < 0] = 0
            
            elif self.SCHEDULING_TYPE.split("_")[-1] == "topology":
                top = eval(str(self.topology))
                rows = ((-schedule).argsort(axis=0)[:top, :]).reshape(
                    top * num_attached)
                cols = list(range(num_attached)) * top
                schedule[:, :] = 0
                schedule[rows, cols] = 1
            
            schedule[mat_SINR <= self.SINR_limit] = 0
            
            lookup = np.sum(schedule, axis=0) == 0
            schedule[:, lookup] = 1
            
            schedule[mat_SINR <= self.SINR_limit] = 0
            
            lookup = np.sum(schedule, axis=1) == 0
            schedule[lookup, :] = 1
            
            schedule = np.vstack(
                (schedule, schedule, schedule, schedule, schedule))
            scheduling_decisions[:, ids] = schedule.astype(bool)
            small['schedule'] = schedule
    
    return scheduling_decisions
