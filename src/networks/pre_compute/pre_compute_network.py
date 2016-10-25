from copy import deepcopy
from random import seed
import numpy as np

from networks.set_power_bias import set_benchmark_pb
from networks.set_network_parameters import update_network
from networks.run_frames import run_full_frame

pre_computed_network = []
standalone_scheduler = []


def return_pre_compute_fitness(self, scheduler, scheduler_type,
                               PRE_COMPUTED_SCENARIOS):
    """ Run a full frame of the network using the pre-computed stats """
    
    self.improvement_R_list = []
    self.improvement_SINR_list = []
    self.improvement_SINR5_list = []
    self.improvement_SINR50_list = []
    self.improvement_DL5_list = []
    self.improvement_DL50_list = []
    self.ave_improvement_R = None
    self.ave_improvement_SINR = None
    self.ave_improvement_SINR5 = None
    
    for scenario in PRE_COMPUTED_SCENARIOS:
        self.scheduling_algorithm = scheduler
        self.SCHEDULING_TYPE = scheduler_type
        self.PRE_COMPUTE = True
        self.SCHEDULING = True
        
        self.first_log_R = scenario["first_log_R"]
        self.first_log_SINR = scenario["first_log_SINR"]
        self.first_SINR5 = scenario["first_SINR5"]
        self.first_SINR50 = scenario["first_SINR50"]
        self.first_DL5 = scenario["first_DL5"]
        self.first_DL50 = scenario["first_DL50"]
        
        self.users = scenario["users"]
        self.small_cells = scenario["small_cells"]
        self.macro_cells = scenario["macro_cells"]
        self.SINR_SF_UE_est = scenario["SINR_SF_UE_est"]
        self.SINR_SF_UE_act = scenario["SINR_SF_UE_act"]
        self.max_SINR_over_frame = scenario['max_SINR_over_frame']
        self.min_SINR_over_frame = scenario['min_SINR_over_frame']
        self.potential_slots = deepcopy(scenario["potential_slots"])
        
        answers = self.run_full_frame(two=self.PRINT, three=self.SAVE)
    return answers


def save_pre_compute_scenarios(self):
    """ Pre-compute all SINR data for all training scenarios so that
    evolutionary runs can be done much faster. """
    
    self.PRE_COMPUTE = False
    self.seed = 13
    # original seed is 13
    seed(self.seed)
    np.random.seed(self.seed)
    
    for frame in range(self.iterations):
        self.iteration = self.scenario + frame
        self.users = self.user_scenarios[frame]
        
        # Use benchmark power/bias and ABS
        self.BENCHMARK_ABS = True
        self = set_benchmark_pb(self)
        self = update_network(self)
        self = run_full_frame(self, first=True)
        
        self.PRE_COMPUTE = True
        self = update_network(self)
        
        all_cell_dict = compute_cell_requirements(self)
        
        # Get Sum Log R for each small cell
        small_ues = [UE['id'] for UE in self.users if
                     UE['attachment'] == 'small']
        average_downlinks = np.average(self.received_downlinks, axis=0)[
            small_ues]
        log_average_downlinks = np.log(
            average_downlinks[average_downlinks > 0])
        log_average_downlinks[log_average_downlinks == -np.inf] = 0
        small_log_R = np.sum(log_average_downlinks)
        
        vals = {"first_log_R": small_log_R,
                "small_cells": [{"attached_users": cell['attached_users'],
                                 "id": cell['id']} for cell in
                                self.small_cells],
                "small_users": small_ues,
                "all_cell_dict": all_cell_dict,
                "SINR_SF_UE_est": self.SINR_SF_UE_est,
                "SINR_SF_UE_act": self.SINR_SF_UE_act,
                "avg_SINR_over_frame": self.avg_SINR_over_frame,
                "potential_slots": self.potential_slots}

        pre_computed_network.append(deepcopy(vals))
        self.ALL_TOGETHER = False


def compute_cell_requirements(self):
    """ Prepare all necessary info for pre-computing the network."""
    
    all_cell_dict = {}
    
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
        mat_good_subframes = ones * (mat_SINR > self.SINR_limit).sum(0)[None,
                                    :]
        
        mat_least_congested_downlinks = np.log2(1 + mat_SINR)
        
        mat_avg_down_F = ones * np.average(mat_least_congested_downlinks,
                                           axis=0)[None, :]
        mat_min_down_F = ones * np.min(mat_least_congested_downlinks, axis=0)[
                                None, :]
        mat_max_down_F = ones * np.max(mat_least_congested_downlinks, axis=0)[
                                None, :]
        mat_LPT_down_F = ones * np.percentile(mat_least_congested_downlinks,
                                              25, axis=0)[None, :]
        mat_UPT_down_F = ones * np.percentile(mat_least_congested_downlinks,
                                              75, axis=0)[None, :]
        
        mat_avg_down_SF = ones * np.average(mat_least_congested_downlinks,
                                            axis=1)[:, None]
        mat_min_down_SF = ones * np.min(mat_least_congested_downlinks, axis=1)[
                                 :, None]
        mat_max_down_SF = ones * np.max(mat_least_congested_downlinks, axis=1)[
                                 :, None]
        mat_LPT_down_SF = ones * np.percentile(mat_least_congested_downlinks,
                                               25, axis=1)[:, None]
        mat_UPT_down_SF = ones * np.percentile(mat_least_congested_downlinks,
                                               75, axis=1)[:, None]
        
        mat_avg_down_cell = ones * np.average(mat_least_congested_downlinks)
        mat_min_down_cell = ones * np.min(mat_least_congested_downlinks)
        mat_max_down_cell = ones * np.max(mat_least_congested_downlinks)
        mat_LPT_down_cell = ones * np.percentile(mat_least_congested_downlinks,
                                                 25)
        mat_UPT_down_cell = ones * np.percentile(mat_least_congested_downlinks,
                                                 75)
        
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
        
        cell_dict = {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5,
                     "T6": T6,
                     "T7": T7, "T8": T8, "T9": T9, "T10": T10, "T11": T11,
                     "T12": T12, "T13": T13, "T14": T14, "T15": T15,
                     "T16": T16,
                     "T17": T17, "T18": T18, "T19": T19, "T20": T20,
                     "T21": T21,
                     "ABS": ABS}
        
        all_cell_dict[str(small['id'])] = cell_dict
    
    return all_cell_dict
