import numpy as np
from os import getcwd
from copy import deepcopy
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

from utilities.fitness.math_functions import ave
from networks.plotting import heatmaps
from networks.math_functions import return_percent
from algorithm.parameters import params
from networks.network_statistics import stats, stats_lists
from networks.plotting.generalisation import plot_generalisation


def get_average_performance(self):
    """ Look at all SCs in the network and see how performance varies
        across cells of differing sizes."""
    
    SC_UEs = [user['id'] for user in self.users if
              user['attachment'] == "small"]
    
    downlink_difference = (
                          self.evolved_downlinks - self.benchmark_downlinks) / 1024 / 1024
    perks = [return_percent(orig, new) for orig, new in
             zip(self.evolved_downlinks[SC_UEs],
                 self.benchmark_downlinks[SC_UEs])]
    downs = downlink_difference[SC_UEs]
    
    stats_lists['max_downs_list'].append(max(downs))
    stats_lists['max_perks_list'].append(max(perks))
    stats_lists['downlink_5_list'].append(stats['diff_5'])
    stats_lists['downlink_50_list'].append(stats['diff_50'])
    stats_lists['SLR_difference_list'].append(stats['diff_SLR'])
    stats_lists['SLR_difference_list_2'].append(stats['orig_SLR'])
    # stats_lists['SLR_difference_list_3'].append(stats['opt_SLR'])
    
    print("Average max downlink improvement over benchmark:  ",
          np.average(stats_lists['max_downs_list']))
    print("Average max percentage improvement over benchmark:",
          np.average(stats_lists['max_perks_list']))
    
    print("Average 5th percentile improvement:",
          np.average(stats_lists['downlink_5_list']))
    print("Average 50th percentile improvement:",
          np.average(stats_lists['downlink_50_list']))
    
    print("Average Sum(Log(R)) improvement over baseline:",
          np.average(stats_lists['SLR_difference_list_2']))
    print("Average Sum(Log(R)) improvement over benchmark:",
          np.average(stats_lists['SLR_difference_list']))

    for cell in self.small_cells:
        orig = cell['first_log_R']
        new = cell['sum_log_R']
        bench = cell['bench_log_R']

        perc = 0
        b_perc = 0

        if orig:
            perc = return_percent(orig, new)
            b_perc = return_percent(orig, bench)
        
        if len(cell['attached_users']) in self.SC_dict:
            self.SC_dict[len(cell['attached_users'])].append(
                [b_perc, perc])
        else:
            self.SC_dict[len(cell['attached_users'])] = [
                [b_perc, perc]]
        
    return self


def output_final_average_performance(self):
    opts = {}
    
    keys = []
    b_mark = []
    b_mark_std = []
    evolved = []
    evolved_std = []
    
    bench_perf_imp = []
    
    for entry in self.SC_dict:
        keys.append(entry)
        
        b = np.average([i[0] for i in self.SC_dict[entry]])
        b_std = np.std([i[0] for i in self.SC_dict[entry]])
        b_mark.append(b)
        b_mark_std.append(b_std)
        
        e = np.average([i[1] for i in self.SC_dict[entry]])
        e_std = np.std([i[1] for i in self.SC_dict[entry]])
        evolved.append(e)
        evolved_std.append(e_std)
        
        bench_imp = -return_percent(e, b)
        
        if bench_imp > 0:
            bench_perf_imp.append(bench_imp)
        
        print(("%d\t%d\tB_mark: %.2f  \tEvolved: %.2f \tB-E Diff: %.2f" % (
            entry, len(self.SC_dict[entry]), b, e, bench_imp)))
    
    opts['Evolved'] = np.asarray(evolved)
    opts['Benchmark'] = np.asarray(b_mark)
    
    bench_perf_imp = np.asarray(bench_perf_imp)
    
    lens = [len(entry) for entry in list(self.SC_dict.values())]
    benz = []
    for entry in self.SC_dict:
        benz.extend([entry for i in range(len(self.SC_dict[entry]))])
    
    ave_lens = np.average(benz)
    std_lens = np.std(benz)
    print("\nAverage SC size:", ave_lens)
    print("Standard deviation of SC size:", std_lens, "\n")
    
    print("Average cell SLR improvement over baseline:\t",
          np.average(b_mark))
    print("Average benchmark improvement over baseline:\t",
          np.average(evolved))
    
    print(
        "\nAverage increase in performance over benchmark (over baseline):",
        np.average(bench_perf_imp))
    
    print("\n")
    
    if params['SHOW'] or params['SAVE']:
        plot_generalisation(keys, lens, opts)


def get_benchmark_difference(self):
    """ Find the difference in performance between the given method and
        the benchmark method. Return the difference as a percentage."""
    
    self.ALL_TOGETHER = True
    self.SYNCHRONOUS_ABS = False
    self.save_scheduling_algorithm = deepcopy(self.scheduling_algorithm)
    self.save_ABS_algorithm = deepcopy(self.ABS_algorithm)
    self.save_scheduling_type = deepcopy(self.SCHEDULING_TYPE)
    
    baseline_fitness = []
    benchmark_fitness = []
    evolved_fitness = []
    
    if self.SAVE:
        self.generate_save_folder()
    
    # Step 1: Get benchmark performance
    for frame in range(self.iterations):
        self.iteration = self.scenario + frame
        # How many full frames of 40 subframes do we want to run?
        # 25 Full frames = 1 second
        
        self.users = self.user_scenarios[frame]
        
        # Set Baseline
        if self.PRINT:
            print("Baseline")
        self.ABS_algorithm = None
        self.SCHEDULING = False
        self.scheduling_algorithm = None
        
        # self.reset_to_zero()
        # self.update_network(FIST=True)
        # answers = self.run_full_frame(first=True, two=self.PRINT)
        self.set_benchmark_pb()
        self.update_network(FIST=True)
        answers_bline = self.run_full_frame(first=True, two=self.PRINT,
                                            three=self.SAVE)
        self.first_log_R = answers_bline['sum_log_R']
        baseline_fitness.append(answers_bline)
        
        baseline_x = self.CDF_downlink
        baseline_y = self.CDF_SINR
        
        if self.MAP:
            heatmaps.save_heatmap(self, 'Non-optimised_' + str(self.iteration))
        
        # Set Benchmark
        if self.PRINT:
            print("Benchmark")
        self.SCHEDULING = True
        self.BENCHMARK_ABS = True
        self.BENCHMARK_SCHEDULING = True
        
        self.set_benchmark_pb()
        self.update_network()
        answers = self.run_full_frame(first=True, two=self.PRINT)
        answers_bmark = self.run_full_frame(two=self.PRINT, three=self.SAVE)
        benchmark_fitness.append(answers_bmark)
        
        benchmark_x = self.CDF_downlink
        benchmark_y = self.CDF_SINR
        
        # Set Evolved
        if self.PRINT:
            print("Evolved")
        self.BENCHMARK_ABS = True
        self.BENCHMARK_SCHEDULING = False
        self.SCHEDULING_TYPE = self.save_scheduling_type
        self.scheduling_algorithm = self.save_scheduling_algorithm
        self.ABS_algorithm = self.save_ABS_algorithm
        self.reset_to_zero()
        self.update_network(FIST=True)
        answers = self.run_full_frame(first=True, two=self.PRINT)
        self.balance_network()
        answers_evo = self.run_full_frame(two=self.PRINT, three=self.SAVE)
        evolved_fitness.append(answers_evo)
        
        evolved_x = self.CDF_downlink
        evolved_y = self.CDF_SINR
        
        if self.PRINT:
            print("-----------")
        
        fig = plt.figure(figsize=[20, 15])
        ax1 = fig.add_subplot(1, 1, 1)
        
        yar = self.actual_frequency
        
        ax1.plot(evolved_x, yar, 'r', label="Evolved ABS & Scheduling")
        ax1.plot(benchmark_x, yar, 'b', label="Benchmark ABS & Scheduling")
        ax1.plot(baseline_x, yar, 'k', label="Baseline ABS & Scheduling")
        
        ax1.set_ylabel('Cumulative distribution')
        ax1.set_xlabel('Log of downlink rates (bits/sec)', color='b')
        ax1.set_ylim([0, 1])
        ax1.legend(loc='best')
        
        if self.SAVE:
            if self.SYNCHRONOUS_ABS:
                plt.savefig(getcwd() + '/Network_Stats/' + str(
                    self.TIME_STAMP) + '/Synchronous_ABS_Complete_Comparison_' + str(
                    self.iteration) + '.pdf', bbox_inches='tight')
            else:
                plt.savefig(
                    getcwd() + self.slash + 'Network_Stats' + self.slash + str(
                        self.TIME_STAMP) + self.slash + 'Asynchronous_ABS_Complete_Comparison_' + str(
                        self.iteration) + '.pdf', bbox_inches='tight')
        if self.SHOW:
            plt.show()
        if self.SHOW or self.SAVE:
            plt.close()
        
        if self.MAP:
            heatmaps.save_heatmap(self, 'Optimised_' + str(self.iteration))
    
    ave_baseline_sum_log_r = ave([i['sum_log_R'] for i in
                                  baseline_fitness])
    ave_benchmark_sum_log_r = ave([i['sum_log_R'] for i in
                                   benchmark_fitness])
    ave_evolved_sum_log_r = ave([i['sum_log_R'] for i in evolved_fitness])
    
    benchmark_difference = self.return_percent(ave_baseline_sum_log_r,
                                               ave_benchmark_sum_log_r)
    evolved_difference = self.return_percent(ave_baseline_sum_log_r,
                                             ave_evolved_sum_log_r)
    
    return evolved_difference - benchmark_difference


def get_normalised_benchmark_difference(self):
    """ Find the difference in performance between the given method and
        the benchmark method. Return the difference as a percentage.
        Levels the playing field so all scheduling methods are compared
        from the same starting point."""
    
    self.ALL_TOGETHER = True
    self.SYNCHRONOUS_ABS = False
    self.save_scheduling_algorithm = deepcopy(self.scheduling_algorithm)
    self.save_ABS_algorithm = deepcopy(self.ABS_algorithm)
    self.save_scheduling_type = deepcopy(self.SCHEDULING_TYPE)
    
    baseline_fitness = []
    benchmark_fitness = []
    evolved_fitness = []
    
    if self.SAVE:
        self.generate_save_folder()
    
    # Step 1: Get benchmark performance
    for frame in range(self.iterations):
        self.iteration = self.scenario + frame
        # How many full frames of 40 subframes do we want to run?
        # 25 Full frames = 1 second
        
        self.users = self.user_scenarios[frame]
        
        # Set Baseline
        if self.PRINT:
            print("Baseline")
        self.SCHEDULING = False
        self.scheduling_algorithm = None
        
        if self.FAIR:
            self.BENCHMARK_ABS = False
            self.reset_to_zero()
            self.update_network(FIST=True)
            answers_bline = self.run_full_frame(first=True, two=self.PRINT)
            self.balance_network()
            answers_bline = self.run_full_frame(two=self.PRINT,
                                                three=self.SAVE)
        else:
            self.BENCHMARK_ABS = True
            self.set_benchmark_pb()
            self.update_network(FIST=True)
            answers_bline = self.run_full_frame(first=True, two=self.PRINT)
            # self.update_network()
        # answers_bline = self.run_full_frame(two=self.PRINT, three=self.SAVE)
        self.first_log_R = answers_bline['sum_log_R']
        baseline_fitness.append(answers_bline)
        
        baseline_x = self.CDF_downlink
        baseline_y = self.CDF_SINR
        
        if self.MAP:
            heatmaps.save_heatmap(self, 'Non-optimised_' + str(
                self.iteration))
        
        # Set Benchmark
        if self.PRINT:
            print("Benchmark")
        self.SCHEDULING = True
        self.BENCHMARK_SCHEDULING = True
        
        if self.FAIR:
            self.BENCHMARK_ABS = False
            self.reset_to_zero()
            self.update_network(FIST=True)
            answers = self.run_full_frame(first=True, two=self.PRINT)
            self.balance_network()
            answers = self.run_full_frame(first=True, two=self.PRINT)
        else:
            self.BENCHMARK_ABS = True
            self.set_benchmark_pb()
            self.update_network(FIST=True)
            answers = self.run_full_frame(first=True, two=self.PRINT)
            self.update_network()
        answers_bmark = self.run_full_frame(two=self.PRINT, three=self.SAVE)
        benchmark_fitness.append(answers_bmark)
        
        benchmark_x = self.CDF_downlink
        benchmark_y = self.CDF_SINR
        
        # Set Evolved
        if self.PRINT:
            print("Evolved")
        
        self.BENCHMARK_SCHEDULING = False
        self.SCHEDULING_TYPE = self.save_scheduling_type
        self.scheduling_algorithm = self.save_scheduling_algorithm
        
        if self.FAIR:
            self.BENCHMARK_ABS = False
            self.reset_to_zero()
            self.update_network(FIST=True)
            answers = self.run_full_frame(first=True, two=self.PRINT)
            self.balance_network()
            answers = self.run_full_frame(first=True, two=self.PRINT)
        else:
            self.BENCHMARK_ABS = True
            self.set_benchmark_pb()
            self.update_network(FIST=True)
            answers = self.run_full_frame(first=True, two=self.PRINT)
            self.update_network()
        answers_evo = self.run_full_frame(two=self.PRINT, three=self.SAVE)
        evolved_fitness.append(answers_evo)
        
        evolved_x = self.CDF_downlink
        evolved_y = self.CDF_SINR
        
        if self.PRINT:
            print("-----------")
        
        fig = plt.figure(figsize=[20, 15])
        ax1 = fig.add_subplot(1, 1, 1)
        
        yar = self.actual_frequency
        
        ax1.plot(evolved_x, yar, 'r', label="Evolved Scheduling")
        ax1.plot(benchmark_x, yar, 'b', label="Benchmark Scheduling")
        ax1.plot(baseline_x, yar, 'k', label="Baseline Scheduling")
        
        ax1.set_ylabel('Cumulative distribution')
        ax1.set_xlabel('Log of downlink rates (bits/sec)', color='b')
        ax1.set_ylim([0, 1])
        ax1.legend(loc='best')
        
        if self.SAVE:
            if self.SYNCHRONOUS_ABS:
                plt.savefig(getcwd() + '/Network_Stats/' + str(
                    self.TIME_STAMP) + '/Synchronous_ABS_Complete_Comparison_' + str(
                    self.iteration) + '.pdf', bbox_inches='tight')
            else:
                plt.savefig(
                    getcwd() + self.slash + 'Network_Stats' + self.slash + str(
                        self.TIME_STAMP) + self.slash + 'Asynchronous_ABS_Complete_Comparison_' + str(
                        self.iteration) + '.pdf', bbox_inches='tight')
        if self.SHOW:
            plt.show()
        if self.SHOW or self.SAVE:
            plt.close()
        
        if self.MAP:
            heatmaps.save_heatmap(self, 'Optimised_' + str(self.iteration))
    
    ave_baseline_sum_log_r = ave([i['sum_log_R'] for i in
                                  baseline_fitness])
    ave_benchmark_sum_log_r = ave([i['sum_log_R'] for i in
                                   benchmark_fitness])
    ave_evolved_sum_log_r = ave([i['sum_log_R'] for i in evolved_fitness])
    
    benchmark_difference = self.return_percent(ave_baseline_sum_log_r,
                                               ave_benchmark_sum_log_r)
    evolved_difference = self.return_percent(ave_baseline_sum_log_r,
                                             ave_evolved_sum_log_r)
    
    return evolved_difference - benchmark_difference
