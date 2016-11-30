from copy import copy
from math import floor, ceil
from operator import itemgetter

import numpy as np
from numpy import mean

from algorithm.parameters import params
from networks.math_functions import return_percent
from networks.plotting.CDF import CDFs
from utilities.fitness.math_functions import ave

stats = {"SINR5": None,
         "DL5": None,
         "sum_log_R": None,
         "CDF_downlink": None,
         "first_CDF_downlink": None,
         "first_CDF_SINR": None,
         "improvement_R": None,
         "ave_improvement_R": None,
         "sum_log_SINR": None,
         "ave_improvement_SINR": None,
         "ave_improvement_SINR5": None,
         "ave_improvement_SINR50": None,
         "ave_improvement_DL5": None,
         "ave_improvement_DL50": None,
         "max_downlink": None,
         "min_downlink": None,
         "power_bias": None,
         "unscheduled_UEs": None,
         "helpless_UEs": None,
         "b_mark_5": None,
         "b_mark_50": None,
         "evolved_5": None,
         "evolved_50": None,
         "BLR": None,
         "ELR": None,
         "OLR": None}


stats_lists = {"improvement_R_list": [],
               "improvement_SINR_list": [],
               "improvement_SINR5_list": [],
               "improvement_SINR50_list": [],
               "improvement_DL5_list": [],
               "improvement_DL50_list": [],
               "downlink_5_list": [],
               "downlink_50_list": [],
               "helpless_UEs": [],
               "max_downs_list": [],
               'max_perks_list': [],
               "SLR_difference_list": [],
               "SLR_difference_list_2": [],
               "SLR_difference_list_3": []}


def get_comparison_stats():
    stats['diff_5'] = return_percent(stats['b_mark_5'], stats['evolved_5'])
    stats['diff_50'] = return_percent(stats['b_mark_50'], stats['evolved_50'])
    stats['diff_SLR'] = return_percent(stats['BLR'], stats['ELR'])
    stats['orig_SLR'] = return_percent(stats['OLR'], stats['ELR'])


def generate_stats(self, FIRST=False):
    """ Runs all programs to set the users for a given network state. Also
        Gets downlinks for all users."""

    stats['DL5'] = get_downlink_percentile(self.users, 0.05)
    stats['SINR5'] = get_SINR_percentile(self.users, 0.05)
    stats['DL50'] = get_downlink_percentile(self.users, 0.5)
    stats['SINR50'] = get_SINR_percentile(self.users, 0.5)

    stats["unscheduled_UEs"] = len(
        [UE for UE in self.users if UE['downlink'] == 0])
    stats["helpless_UEs"] = len(
        [UE for UE in self.users if max(UE['SINR_frame']) <= self.SINR_limit])
    
    if FIRST:
        stats['first_log_R'] = stats['sum_log_R']
        stats['first_log_SINR'] = stats['sum_log_SINR']
        stats['first_SINR5'] = stats['SINR5']
        stats['first_SINR50'] = stats['SINR50']
        stats['first_DL5'] = stats['DL5']
        stats['first_DL50'] = stats['DL50']

    else:
        stats['improvement_R'] = return_percent(stats['first_log_R'],
                                                stats['sum_log_R'])
        stats['improvement_SINR'] = return_percent(stats['first_log_SINR'],
                                                   stats['sum_log_SINR'])
        stats['improvement_SINR5'] = return_percent(stats['first_SINR5'],
                                                    stats['SINR5'])
        stats['improvement_SINR50'] = return_percent(stats['first_SINR50'],
                                                     stats['SINR50'])
        stats['improvement_DL5'] = return_percent(stats['first_DL5'],
                                                  stats['DL5'])
        stats['improvement_DL50'] = return_percent(stats['first_DL50'],
                                                   stats['DL50'])
        
        stats_lists['improvement_R_list'].append(stats['improvement_R'])
        stats_lists['improvement_SINR_list'].append(stats['improvement_SINR'])
        stats_lists['improvement_SINR5_list'].append(
            stats['improvement_SINR5'])
        stats_lists[
            'improvement_SINR50_list'].append(stats['improvement_SINR50'])
        stats_lists['improvement_DL5_list'].append(stats['improvement_DL5'])
        stats_lists['improvement_DL50_list'].append(stats['improvement_DL50'])
        
        stats['ave_improvement_R'] = ave(stats_lists['improvement_R_list'])
        stats['ave_improvement_SINR'] = ave(stats_lists[
                                                'improvement_SINR_list'])
        stats['ave_improvement_SINR5'] = ave(stats_lists[
                                                 'improvement_SINR5_list'])
        stats['ave_improvement_SINR50'] = ave(stats_lists[
                                                  'improvement_SINR50_list'])
        stats['ave_improvement_DL5'] = ave(stats_lists['improvement_DL5_list'])
        stats['ave_improvement_DL50'] = ave(stats_lists[
                                                'improvement_DL50_list'])
    
    if params['PRINT']:
        print_stats(self, FIRST)
    
    if params['SAVE']:
        save_stats(self, FIRST)
    
    return self


def print_stats(self, FIRST):
    
    if self.frame:
        print(self.frame)
    print("\tMin:", round(stats['min_downlink'][0], 2), end=' ')
    print("\tMax:", round(stats['max_downlink'][0], 2), end=' ')
    print("\t5th %:", round(stats['DL5'] / 1024 / 1024, 2), end=' ')
    print("\t50th %:", round(stats['DL50'] / 1024 / 1024, 2), end=' ')
    if FIRST:
        print("\tSLR:", round(stats['sum_log_R'], 2))
    else:
        print("\tSLR:", round(stats['sum_log_R'], 2), end=' ')
        if self.difference:
            print("\tImpr:", round(stats['improvement_R'], 4))
        else:
            print("\tImpr:", round(stats['improvement_R'], 4), end=' ')
            print("   \tAve Impr:", round(stats['ave_improvement_R'], 4))
    print("\tHelpless:", stats['helpless_UEs'], end=' ')
    print("\tUnscheduled:", stats['unscheduled_UEs'])


def save_stats(self, FIRST):
    filename = params['FILE_PATH'] + "Network_Stats.txt"
    savefile = open(filename, 'a')
    savefile.write(str(self.frame))
    savefile.write("\tMin: " + str(round(stats['min_downlink'][0], 2)))
    savefile.write("\tMax: " + str(round(stats['max_downlink'][0], 2)))
    savefile.write("\t5th %:" + str(round(stats['DL5'] / 1024 / 1024, 2)))
    savefile.write("\t50th %:" + str(round(stats['DL50'] / 1024 / 1024, 2)))
    savefile.write("\tSLR: " + str(round(stats['sum_log_R'], 2)))
    if not FIRST:
        savefile.write("\tImpr: " + str(round(stats['improvement_R'], 2)))
        if not self.difference:
            savefile.write("\tAve Impr: " + str(round(stats[
                                                          'ave_improvement_R'], 4)))
    savefile.write("\tHelpless UEs: " + str(stats['helpless_UEs']))
    savefile.write("\tUnscheduled UEs: " + str(stats['unscheduled_UEs']))
    savefile.write("\n")
    savefile.close()


def get_downlink_percentile(users, percentile):
    """Get the total downlink rate for all users in a network. Return
       downlink based on the required input percentile."""
    
    # Step 1: Order all users by downlink value, from lowest to highest
    all_users = []
    for user in users:
        if user['downlink']:
            all_users.append([user['id'], user['downlink']])
    all_users.sort(key=itemgetter(1))
    
    # Step 2: Multiply percentile by total number of users
    index = percentile * len(users)
    if index < 1:
        index = 1
    
    # step 3: Check if index is an integer. If not, then return the
    # average of two values, if so then return the index value.
    if int(index) != index:  # means it's not an integer
        one = int(floor(index))
        two = int(ceil(index))
        percentile_down_1 = all_users[one]
        percentile_down_2 = all_users[two]
        downlink_percentiles = mean(
            [percentile_down_1[1], percentile_down_2[1]])
    else:
        downlink_percentiles = all_users[int(index)][1]
    
    return downlink_percentiles


def get_SINR_percentile(users, percentile):
    """Get the total SINR for all users in a network. Return SINR
       based on the required input percentile."""
    
    # Step 1: Order all users by SINR value, from lowest to highest
    all_users = []
    for user in users:
        all_users.append([user['id'], user['average_SINR']])
    all_users.sort(key=itemgetter(1))
    
    # Step 2: Multiply percentile by total number of users
    index = percentile * len(users)
    if index < 1:
        index = 1
    
    # step 3: Check if index is an integer. If not, then return the
    # average of two values, if so then return the index value.
    if int(index) != index:  # means it's not an integer
        one = int(floor(index))
        two = int(ceil(index))
        percentile_SINR_1 = all_users[one]
        percentile_SINR_2 = all_users[two]
        total_SINR = mean([percentile_SINR_1[1], percentile_SINR_2[1]])
    else:
        total_SINR = all_users[int(index)][1]
    
    return total_SINR


def get_user_statistics(self, FIRST=False):
    """ Get the measurement statistics for all UEs in the network:
            Average downlink rate of each UE
            Average SINR of each UE
            Peak downlink rate of the overall network
            Minimum downlink rate of the overall network
            CDF of both downlink and SINR (based on average values)
    """
    
    average_downlinks = np.average(self.received_downlinks, axis=0)
    average_SINRs = np.average(self.SINR_SF_UE_act, axis=0,
                               weights=self.SINR_SF_UE_act.astype(bool))
    self.average_SINRs = average_SINRs
    
    log_average_downlinks = np.log(average_downlinks[average_downlinks > 0])
    log_average_downlinks[log_average_downlinks == -np.inf] = 0
    
    log_average_SINRs = np.log(average_SINRs)
    log_average_SINRs[log_average_SINRs == -np.inf] = 0
    
    if params['SHOW'] or params['SAVE']:
        if params['CDF_log']:
            CDFs['CDF_downlink'] = (log_average_downlinks).tolist()
            CDFs['CDF_downlink'].extend(
                [0 for _ in average_downlinks[average_downlinks == 0]])
        else:
            CDFs['CDF_downlink'] = (average_downlinks / 1024 / 1024).tolist()
    
        if params['CDF_log']:
            CDFs['CDF_SINR'] = (log_average_SINRs).tolist()
        else:
            CDFs['CDF_SINR'] = (average_SINRs).tolist()

    stats['sum_log_SINR'] = np.sum(log_average_SINRs)
    stats['sum_log_R'] = np.sum(log_average_downlinks)

    stats['max_downlink'] = [np.max(average_downlinks) / 1024 / 1024,
                         np.argmax(average_downlinks)]
    stats['min_downlink'] = [
        np.min(average_downlinks[average_downlinks > 0]) / 1024 / 1024,
        np.argmax(average_downlinks[average_downlinks > 0])]
    
    if self.BENCHMARK_SCHEDULING:
        self.benchmark_downlinks = average_downlinks
    elif not FIRST:
        self.evolved_downlinks = average_downlinks
    
    for user_id, user in enumerate(self.users):
        user['downlink'] = average_downlinks[user_id]
        user['average_SINR'] = average_SINRs[user_id]
    
    for cell_id in range(self.n_all_cells):
        if cell_id < self.n_macro_cells:
            cell = self.macro_cells[cell_id]
            attached_users = cell['attached_users']
        else:
            cell = self.small_cells[cell_id - self.n_macro_cells]
            attached_users = cell['attached_users']
        attached_ue_ids = attached_users
        if len(attached_ue_ids) > 0:
            cell['average_downlink'] = np.average(
                average_downlinks[attached_ue_ids])
            available_down = average_downlinks[attached_ue_ids]
            cell_log_ave_down = np.log(available_down[available_down > 0])
            cell_log_ave_down[cell_log_ave_down == -np.inf] = 0
            cell['sum_log_R'] = np.sum(cell_log_ave_down)
        else:
            cell['average_downlink'] = 0
            cell['sum_log_R'] = 0
        if FIRST:
            cell['first_log_R'] = copy(cell['sum_log_R'])
        if self.BENCHMARK_SCHEDULING:
            cell['bench_log_R'] = copy(cell['sum_log_R'])
        if self.OPT_SCHEDULING:
            cell['OPT_log_R'] = copy(cell['sum_log_R'])
        if self.NEW_SCHEDULING:
            cell['new_log_R'] = copy(cell['sum_log_R'])

    if params['SHOW'] or params['SAVE']:
        CDFs['CDF_downlink'].sort()
        CDFs['CDF_downlink'] = np.asarray(CDFs['CDF_downlink'])
        CDFs['CDF_downlink'][CDFs['CDF_downlink'] == 0] = None
        CDFs['CDF_downlink'] = CDFs['CDF_downlink'][self.SC_attached_UEs]
        CDFs['CDF_downlink'] = CDFs['CDF_downlink'].tolist()
        CDFs['CDF_SINR'].sort()
    
    return self
