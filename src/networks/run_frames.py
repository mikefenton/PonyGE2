from networks.get_downlink import get_basic_downlink
from networks.scheduling import set_scheduling
from utilities.fitness.math_functions import pdiv
from networks.set_power_bias import set_benchmark_pb
from networks.set_network_parameters import update_network
from networks.network_statistics import get_user_statistics, generate_stats, stats
from networks.plotting.CDF import CDFs
from algorithm.parameters import params

import numpy as np


def run_baseline_frame(self):
    """
    Run baseline scheduling.
    
    :param self:
    :return: Self.
    """
    
    print("Baseline scheduling")
    self.ALL_TOGETHER = True
    self.BASELINE_SCHEDULING = True
    
    # Set benchmark power & bias levels
    self = set_benchmark_pb(self)
    
    # Set all network parameters including ABSrs, SINRs, UE attachments, etc.
    self = update_network(self)
    
    if self.PRE_COMPUTE:
        self.scheduling_decisions, self = set_scheduling(self)
    
    self = run_full_frame(self, first=True)
    self.BASELINE_SCHEDULING = False
    
    if params['COLLECT_STATS']:
        stats['OLR'] = stats['sum_log_R']
        CDFs['baseline_CDF'] = CDFs['CDF_downlink']
        CDFs['ave_CDF_baseline'] += CDFs['CDF_downlink']
        
    return self
    

def run_benchmark_frame(self):
    """
    Run benchmark scheduling.
    
    :param self:
    :return:
    """

    print("Benchmark scheduling")
    self.BENCHMARK_SCHEDULING = True
    
    if self.PRE_COMPUTE:
        self.scheduling_decisions, self = set_scheduling(self,
                                                         METHOD="benchmark")

    self = run_full_frame(self)
    self.BENCHMARK_SCHEDULING = False
    
    if params['COLLECT_STATS']:
        stats['b_mark_5'] = round(stats['DL5'] / 1024 / 1024, 2)
        stats['b_mark_50'] = round(stats['DL50'] / 1024 / 1024, 2)
        stats['BLR'] = stats['sum_log_R']
        CDFs['benchmark_CDF'] = CDFs['CDF_downlink']
        CDFs['ave_CDF_benchmark'] += CDFs['CDF_downlink']
    
    return self
    

def run_evolved_frame(self):
    """
    Run evolved scheduling.
    
    :param self:
    :return: Self.
    """
    
    print("Evolved scheduling")
    self.EVOLVED_SCHEDULING = True
    
    if self.PRE_COMPUTE:
        self.scheduling_decisions, self = set_scheduling(self,
                                                         METHOD="evolved")
    self = run_full_frame(self)
    self.EVOLVED_SCHEDULING = False
    
    if params['COLLECT_STATS']:
        stats['evolved_5'] = round(stats['DL5'] / 1024 / 1024, 2)
        stats['evolved_50'] = round(stats['DL50'] / 1024 / 1024, 2)
        stats['ELR'] = stats['sum_log_R']
        CDFs['evolved_CDF'] = CDFs['CDF_downlink']
        CDFs['ave_CDF_evolved'] += CDFs['CDF_downlink']

    return self


def run_full_frame(self, first=False):
    """ Run the network for a full frame of 40ms and calculate the
        performance of the network for that frame.
    """
    
    # Need to figure out a schedule for SC attached UEs. We need to say
    # which UEs will recieve data in which subframes. MC UEs are all
    # scheduled constantly, but this is not the case for SC UEs as ABS
    # subframes will be in effect and some UEs will get better performance
    # only during an ABS subframe. UEs cannot be scheduled if their SINR
    # is less than 1, as this will result in a transmission outage. The
    # only way to save these UEs is to reconfigure the network.
    
    self.frame += 1
    
    # self.schedule_info1 is num_SFs*num_users and stores in cell (x,y)
    # the number of other UEs sharing SF x with UE y
    self.schedule_info1 = np.zeros(shape=(40, self.n_users))
    
    for macro in self.macro_cells:
        macro['schedule'] = []
        if macro['attached_users']:
            MC_attached_users = macro['attached_users']
            sf_congestion = self.potential_slots[:, MC_attached_users].sum(
                axis=1)
            self.schedule_info1[:, MC_attached_users] = self.potential_slots[:,
                                                        MC_attached_users]
            potential = self.potential_slots[:, MC_attached_users]
            potential[potential == 0] = np.nan
            macro['schedule'] = potential * np.array(MC_attached_users)
            self.schedule_info1[:, MC_attached_users] *= pdiv(1,
                                                              sf_congestion[:,
                                                              np.newaxis])
    
    for small in self.small_cells:
        small['schedule'] = []
        if small['attached_users']:
            SC_attached_users = small['attached_users']
           
            if self.BASELINE_SCHEDULING:
                sf_congestion = self.potential_slots[:, SC_attached_users].sum(
                    axis=1)
                self.schedule_info1[:,
                SC_attached_users] = self.potential_slots[:, SC_attached_users]
                potential = self.potential_slots[:, SC_attached_users]
            else:
                sf_congestion = self.scheduling_decisions[:,
                                SC_attached_users].sum(axis=1)
                self.schedule_info1[:,
                SC_attached_users] = self.scheduling_decisions[:,
                                     SC_attached_users]
                potential = self.scheduling_decisions[:, SC_attached_users]
            small['schedule'] = potential * np.array(SC_attached_users)
            self.schedule_info1[:, SC_attached_users] *= pdiv(1,
                                                              sf_congestion[:,
                                                              np.newaxis])
    self = get_basic_downlink(self)
    self = get_user_statistics(self, FIRST=first)
    generate_stats(self, FIRST=first)
    
    return self
