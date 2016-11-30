from copy import deepcopy
from operator import itemgetter

import numpy as np

from algorithm.parameters import params
from networks.math_functions import return_percent
from networks.pre_compute import pre_compute_network as PCN
from utilities.fitness.math_functions import pdiv


class Standalone_Fitness:
    """
    A class for calculating the fitness of a schedule without running the
    network.
    
    """

    def __init__(self, distribution="training"):
        import networks.Optimise_Network as OPT
        self.OPT = OPT.Optimise_Network(
                   PB_ALGORITHM="pdiv(ms_log_R, N_s)",
                   ABS_ALGORITHM="pdiv(ABS_MUEs,non_ABS_MUEs+ABS_MSUEs)",
                   DISTRIBUTION=distribution)
        self.REALISTIC = params['REALISTIC']
        self.DIFFICULTY = params['DIFFICULTY']
        self.bandwidth = self.OPT.bandwidth
        self.SINR_limit = self.OPT.SINR_limit

    def return_pre_compute_fitness(self, scheduler, scheduler_type, name=None):
        """ Run a full frame of the network using the pre-computed stats """

        self.improvement_R_list = []
        self.ave_improvement_R = None
        self.scheduling_algorithm = scheduler
        self.scheduler_type = scheduler_type

        try:
            compile(self.scheduling_algorithm, '<string>', 'eval')
        
        except MemoryError:
            return None

        try:
            check = eval(self.scheduling_algorithm)
        except:
            check = False

        if check or (type(check) is np.float64) or (type(check) is float):
            return None

        for index, scenario in enumerate(PCN.pre_computed_network):

            self.PRE_COMPUTE = True
            self.NAME = name
            self.SCHEDULING = True

            self.first_log_R = scenario["first_log_R"]
            self.all_cell_dict = scenario["all_cell_dict"]
            self.small_users = scenario['small_users']
            self.n_small_users = len(self.small_users)
            self.small_cells = scenario["small_cells"]
            self.SINR_SF_UE_est = scenario["SINR_SF_UE_est"]
            self.SINR_SF_UE_act = scenario["SINR_SF_UE_act"]
            self.potential_slots = scenario["potential_slots"]
            self.avg_SINR_over_frame = scenario["avg_SINR_over_frame"]

            scheduling_decisions = np.ones((40, self.OPT.n_users), dtype=bool)

            for small in [cell for cell in self.small_cells if
                          len(cell['attached_users']) > 1]:
                unsorted_ids = np.array(small['attached_users'])
                attached_ids = unsorted_ids[
                    np.argsort(self.avg_SINR_over_frame[unsorted_ids])]
                ids = attached_ids
                num_attached = len(ids)
                terminals = self.all_cell_dict[str(small['id'])]

                T1 = terminals["T1"]
                T2 = terminals["T2"]
                T3 = terminals["T3"]
                T4 = terminals["T4"]
                T5 = terminals["T5"]
                T6 = terminals["T6"]
                T7 = terminals["T7"]
                T8 = terminals["T8"]
                T9 = terminals["T9"]
                T10 = terminals["T10"]
                T11 = terminals["T11"]
                T12 = terminals["T12"]
                T13 = terminals["T13"]
                T14 = terminals["T14"]
                T15 = terminals["T15"]
                T16 = terminals["T16"]
                T17 = terminals["T17"]
                T18 = terminals["T18"]
                T19 = terminals["T19"]
                T20 = terminals["T20"]
                T21 = terminals["T21"]
                ABS = terminals["ABS"]

                schedule = eval(self.scheduling_algorithm)

                if self.scheduler_type.split("_")[-1] == "threshold":
                    schedule[schedule >= 0] = 1
                    schedule[schedule < 0] = 0

                elif self.scheduler_type.split("_")[-1] == "topology":
                    top = eval(str(self.OPT.topology))
                    rows = ((-schedule).argsort(axis=0)[:top, :]).reshape(
                        top * num_attached)
                    cols = list(range(num_attached))*top
                    schedule[:, :] = 0
                    schedule[rows, cols] = 1
                    
                mat_SINR = self.SINR_SF_UE_est[:8,ids]
                schedule[mat_SINR < self.SINR_limit] = 0
                lookup = np.sum(schedule, axis=0) == 0
                schedule[:, lookup] = 1
                schedule[mat_SINR < self.SINR_limit] = 0
                lookup = np.sum(schedule, axis=1) == 0
                schedule[lookup, :] = 1
                schedule = np.vstack((schedule, schedule, schedule, schedule, schedule))
                scheduling_decisions[:, ids] = schedule.astype(bool)

            scheduling_decisions[self.SINR_SF_UE_est < self.SINR_limit] = False

            if self.REALISTIC:
                # Need to model dropped data transmissions due to actual
                # SINR < self.SINR_limit

                check = np.logical_and((self.SINR_SF_UE_act < self.SINR_limit), scheduling_decisions)
                available = np.logical_and(self.potential_slots, np.logical_not(scheduling_decisions))
                unavailable = self.SINR_SF_UE_act < self.SINR_limit

                copied_dropped = deepcopy(check)
                for sf, lookup in enumerate(copied_dropped):
                    if any(lookup):
                        available_copy = deepcopy(available)
                        boolean = np.zeros((40, self.OPT.n_users))
                        # Only operate on those UEs who have dropped calls
                        boolean[min((sf+4), 39):] = lookup
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
                            lucky_UEs = list(set([i[1] for i in indices if i[1] in troubled_UEs]))
                            for user in lucky_UEs:
                                new_slot = [list(i) for i in indices if i[1]==user][0]
                                available[new_slot[0]][new_slot[1]] = False
                                if self.SINR_SF_UE_act[new_slot[0]][new_slot[1]] >= self.SINR_limit:
                                    # print "Subframe", sf, "rescheduled for UE", user, "in subframe", new_slot[0]
                                    # Reschedule the UE in their new slot
                                    scheduling_decisions[new_slot[0]][new_slot[1]] = True
                                else:
                                    # The slot we had thought was ok is actually
                                    # not. The call will be dropped again.
                                    copied_dropped[new_slot[0]][new_slot[1]] = True
                                    # print "Subframe", sf, "falsely rescheduled for UE", user, "in subframe", new_slot[0]

                scheduling_decisions[self.SINR_SF_UE_act < self.SINR_limit] = False
            self.scheduling_decisions = scheduling_decisions
            self.run_frame()
            self.get_downlink()
            fitness = self.get_fitness()
            self.improvement_R_list.append(fitness)
        self.ave_improvement_R = np.average(self.improvement_R_list)

        return self.ave_improvement_R

    def run_frame(self):

        self.congestion = np.zeros(shape=(40, self.OPT.n_users))

        for small in [cell for cell in self.small_cells if cell['attached_users']]:
            SC_attached_users = small['attached_users']
            sf_congestion = self.scheduling_decisions[:, SC_attached_users].sum(axis=1)
            self.congestion[:, SC_attached_users] = self.scheduling_decisions[:, SC_attached_users]
            self.congestion[:, SC_attached_users] *= pdiv(1,sf_congestion[:, np.newaxis])

    def get_downlink(self):
        """ Gets the downlink rates for UEs using Shannon's formula and
            congestion information. Divides the bandwidth equally between UEs.
           """

        # Instantaneous_downlinks is a num_SFs*num_users matrix storing
        # downlinks received by each UE if no congestion and if each UE was
        # always scheduled.
        instantaneous_downlinks = self.bandwidth * np.log2(1+self.SINR_SF_UE_act[:,self.small_users])

        # Next we take account of the scheduling by dividing the bandwidth by
        # the number sharing each SF.
        self.received_downlinks = instantaneous_downlinks * self.congestion[:,self.small_users]

        # our method yields infinities and nans which are summarily mapped to 0
        self.received_downlinks[np.isnan(self.received_downlinks)] = 0
        self.received_downlinks[self.received_downlinks >= 1E308] = 0

    def get_fitness(self):

        average_downlinks = np.average(self.received_downlinks, axis=0)
        log_average_downlinks = np.log(average_downlinks[average_downlinks > 0])
        log_average_downlinks[log_average_downlinks == -np.inf] = 0
        sum_log_R = np.sum(log_average_downlinks)
        fitness = return_percent(self.first_log_R, sum_log_R)
        return fitness
