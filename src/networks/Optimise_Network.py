# Network optimisation program/function which will take a number of inputs
# for a heteregenous cellular network consisting of combinations of Macro
# Cells (MCs) and Small Cells (SCs) and output network configuration
# settings such that the User Equipment (UE) throughput is maximised.

# Copyright (c) 2014
# Michael Fenton

from random import seed, getstate, setstate
from sys import platform
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
from datetime import datetime
import numpy as np

from networks.plotting.CDF import save_CDF
from networks.plotting import heatmaps
from algorithm.parameters import params
from networks.set_power_bias import set_benchmark_pb, balance_bias, \
    balance_network
from networks.comparisons import get_average_performance
from networks.run_frames import run_baseline_frame, run_benchmark_frame, \
    run_evolved_frame
from networks.network_statistics import stats, get_comparison_stats
from networks.plotting.CDF import CDFs

np.seterr(divide='ignore', invalid='ignore')


class Optimise_Network():
    """A class for optimising a network given some network conditions"""

    def __init__(self,
                SC_power=None,
                SC_CSB=None,
                bias_limit=15,
                SCHEDULING_ALGORITHM=None,
                SCHEDULING_TYPE="original_sched",
                PB_ALGORITHM="pdiv(ms_log_R, N_s)",
                ABS_ALGORITHM="pdiv(ABS_MUEs, non_ABS_MUEs + ABS_MSUEs)",
                TOPOLOGY=2,
                ALL_TOGETHER=False,
                DISTRIBUTION="training",
                BENCHMARK=False,
                SCENARIO_INPUTS=None,
                DIFFERENCE=False,
                scenario=0,
                iterations=0):
        """This is the master func which runs all the individual functions
           within the network optimisation class."""

        import networks.pre_compute.hold_network_info as HNI
        network = HNI.get_network(DISTRIBUTION)

        # Network Values
        self.power_limits = network.power_limits
        self.CSB_limits = [0, bias_limit]
        self.n_macro_cells = network.n_macro_cells
        self.n_small_cells = network.n_small_cells
        self.n_all_cells = network.n_all_cells
        self.macro_cells = network.macro_cells
        self.small_cells = network.small_cells
        self.BS_locations = network.BS_locations
        self.user_locations = []
        self.environmental_encoding = network.environmental_encoding
        self.gains = network.gains
        for i, small in enumerate(self.small_cells):
            if SC_power:
                small['power'] = float(SC_power[i])
            else:
                small['power'] = 23
            if SC_CSB:
                small['bias'] = float(SC_CSB[i])
            else:
                small['bias'] = 0
            small['potential_users'] = []
            small['attached_users'] = []
            small['macro_interactions'] = []
            small['sum_log_R'] = 0
            small['OPT_log_R'] = 0
            small['first_log_R'] = 0
            small['bench_log_R'] = 0
            small['simple_log_R'] = 0
            small['new_log_R'] = 0
            small['average_downlink'] = 0
            small['extended_users'] = []
            small['SINR_frame'] = [[] for _ in range(40)]
            small['sum_SINR'] = [0 for _ in range(40)]
        for i, macro in enumerate(self.macro_cells):
            macro['potential_users'] = []
            macro['attached_users']=[]
            macro['small_interactions'] = []
            macro['sum_log_R'] = 0
            macro['SINR_frame'] = [[] for _ in range(40)]
            macro['sum_SINR'] = [0 for _ in range(40)]
        self.users = network.users
        self.hotspots = network.hotspots
        self.n_users = network.n_users
        self.size = network.size
        # self.perc_signal = network.perc_signal
        self.iterations = network.iterations
        self.scenario = network.scenario
        self.user_scenarios = network.user_scenarios
        self.STRESS_TEST = network.STRESS_TEST
        self.stress_percentage = network.stress_percentage
        self.bandwidth = params['BANDWIDTH']
        del(network)

        # Options
        if type(SCENARIO_INPUTS) is str:
            self.scenario_inputs = eval(SCENARIO_INPUTS)
        else:
            self.scenario_inputs = SCENARIO_INPUTS
        if self.scenario_inputs:
            self.hotspots = []
        self.PRE_COMPUTE = False
        self.difference = DIFFERENCE
        self.topology = TOPOLOGY
        self.SINR_limit = 10**(-5/10.0)  # -5 dB
        self.SINR_limits = [-5, 23]
        self.ALL_TOGETHER = ALL_TOGETHER
        self.PRINT = params['PRINT']
        self.step = 0.01
        self.SCHEDULING = False
        self.MAP = params['MAP']
        self.scheduling_algorithm = SCHEDULING_ALGORITHM
        if self.scheduling_algorithm:
            self.SCHEDULING = True
        self.pb_algorithm = PB_ALGORITHM
        self.ABS_algorithm = ABS_ALGORITHM
        self.ABS_activity = np.asarray([1 for _ in range(self.n_all_cells)])
        self.min_ABS_ratio = 0.875
        # 0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1
        # A higher number means less frames will be blanked. min ratio of 1
        # means no frames will be blanked as a minimum. A min ratio of 0.125
        # means only one frame remains unblanked.
        self.SINR_interference_limit = 11  # self.n_all_cells
        # number of interfering cells is this -1
        self.DIFFICULTY = params['DIFFICULTY']
        self.show_all_UEs = False

        # Fitness Values
        self.sum_log_R_list = []
        self.SINR_list = []
        self.first_xar = []
        self.first_yar = []
        self.max_downs_list = []
        self.max_perks_list = []
        self.downlink_5_list = []
        self.downlink_50_list = []
        self.SLR_difference_list = []
        self.SLR_difference_list_2 = []
        self.SLR_difference_list_3 = []
        self.SC_dict = {}

        self.helpless_UEs = []
        self.CDF_log = False
        self.OPT_SCHEDULING = False
        self.SYNCHRONOUS_ABS = True
        self.BENCHMARK = BENCHMARK
        self.BENCHMARK_SCHEDULING = False
        self.BENCHMARK_ABS = False
        self.NEW_SCHEDULING = False
        if self.BENCHMARK:
            self.BENCHMARK_SCHEDULING = True
            self.BENCHMARK_ABS = True
        self.REALISTIC = params['REALISTIC']
        self.FAIR = params['FAIR']
        self.SCHEDULING_TYPE = SCHEDULING_TYPE
        if not self.REALISTIC:
            self.SINR_interference_limit = self.n_all_cells
        if scenario:
            self.SCENARIO = scenario
        if iterations:
            self.iterations = iterations

        # Variables
        self.small_powers = []
        self.small_biases = []
        self.power_bias = []
        self.all_powers = []
        self.seed = 13
        self.cgm = []
        self.MC_UES = []
        self.SC_UES = []
        self.frame = 0
        # Set minimum number of scheduled sub-frames for all UEs at 5 sub-
        # frames out of a full frame of 40 sub-frames. This is because the
        # maximum permissible ABS ratio is 35/40, meaning there will always
        # be a minimum of 5 frames out of 40 with full MC transmission.
        self.SHOW = params['SHOW']
        
        self.schedule_info = np.zeros(shape=(self.n_all_cells, 40, self.n_users))
        self.schedule_info1 = np.zeros(shape=(40, self.n_users))
        self.noise_W = 10**((-124-30)/10)

        if not self.iterations:
            self.iterations = 1

    def run_all(self):
        """run all functions in the class"""

        state = getstate()

        # original seed is 13
        seed(self.seed)
        np.random.seed(self.seed)
        self.time_list = []
        time1 = datetime.now()
        self.time_list.append(time1)

        self = balance_bias(self)

        if self.difference:
            self.get_normalised_benchmark_difference()
            quit()

        else:

            """ Run the network for a specified number of frames here. Simply
                delete "if None:#".
            """
            for frame in range(self.iterations):
                self.iteration = self.scenario + frame
                self.users = self.user_scenarios[frame]

                # self.reset_to_zero()
                # self.update_network(FIST=True)
                # answers = self.run_full_frame(first=True, two=self.PRINT)

                if self.BENCHMARK:
                    if self.FAIR:
                        self = balance_network(self)
                    else:
                        self = set_benchmark_pb(self)
                        self.update_network()
                    answers = self.run_full_frame(two=self.PRINT, three=self.SAVE)

                elif self.ALL_TOGETHER:
                    # If we're evolving everything together then we don't need to
                    # run things separately to get individual fitnesses. We only
                    # need to run the network multiple times in order to get
                    # individual fitnesses for ABS and Scheduling (i.e. the
                    # fitness for scheduling will be the increase in fitness over
                    # ABS, etc.). If we're doing everything together, then we can
                    # just do it all in one step and save a ton of time since we
                    # get the same answer anyway. Good stuff!
                    self = balance_network(self)
                    answers = self.run_full_frame(two=self.PRINT, three=self.SAVE)

                elif not self.ABS_algorithm and not self.SCHEDULING:
                    # Just the fitness from the pb algorithm
                    self = balance_network(self)
                    answers = self.run_full_frame(two=self.PRINT, three=self.SAVE)

                elif self.ABS_algorithm and not self.SCHEDULING:
                    # Just the fitness from the ABS algorithm
                    self = balance_network(self)
                    answers = self.run_full_frame(first=True, two=self.PRINT, three=self.SAVE)
                    self.update_network()
                    answers = self.run_full_frame(two=self.PRINT, three=self.SAVE)

                else:
                    # Just the fitness from the scheduling algorithm
                    self.ALL_TOGETHER = True
                    self.SCHEDULING = False

                    # self.balance_network()

                    self.BENCHMARK_ABS = True
                    self = set_benchmark_pb(self)
                    self.update_network(FIST=True)

                    answers = self.run_full_frame(first=True, two=self.PRINT, three=self.SAVE)
                    self.SCHEDULING = True
                    self.update_network()
                    answers = self.run_full_frame(two=self.PRINT, three=self.SAVE)
                    self.ALL_TOGETHER = False

                if self.SHOW or self.SAVE:
                    self.save_CDF_plot("Scheduling_"+str(frame), SHOW=self.SHOW, SAVE=self.SAVE)

                if self.MAP:
                    heatmaps.save_heatmap(self, "Optimised")

                if answers == 0:
                    # no point checking other scenarios this guy does nothing
                    break

                if self.ALL_TOGETHER and answers < 2:
                    # no point checking other scenarios this guy sucks
                    break

                elif answers < -5:
                    # no point checking other scenarios this guy sucks
                    break

                # self.get_average_performance()

            total_time = self.time_list[-1] - self.time_list[0]
            if self.PRINT:
                print("\nTotal time taken:", total_time)
            setstate(state)

            return answers

    def run_all_2(self):
        """run all functions in the class"""

        seed(self.seed)
        np.random.seed(self.seed)

        self.PRE_COMPUTE = True

        for frame in range(self.iterations):
            self.iteration = self.scenario + frame
            print("-----\nIteration", self.iteration)
            self.users = self.user_scenarios[frame]

            self.BENCHMARK_ABS = True

            self = run_baseline_frame(self)

            self = run_benchmark_frame(self)

            self = run_evolved_frame(self)
            
            self.ALL_TOGETHER = False

            get_comparison_stats()

            if params['MAP']:
                heatmaps.save_heatmap(self, "Optimised")

            if stats['ave_improvement_R'] == 0 or stats['ave_improvement_R'] < -5:
                # no point checking other scenarios this guy does nothing
                break

            if self.ALL_TOGETHER and stats['ave_improvement_R'] < 2:
                # no point checking other scenarios this guy sucks
                break

            self = get_average_performance(self)

        if params['SHOW'] or params['SAVE']:
            
            plot_name = str(self.n_small_cells) + " SCs CDF"

            CDFs['ave_CDF_baseline'] = np.sort(
                np.array(CDFs['ave_CDF_baseline']))
            CDFs['ave_CDF_benchmark'] = np.sort(
                np.array(CDFs['ave_CDF_benchmark']))
            CDFs['ave_CDF_evolved'] = np.sort(
                np.array(CDFs['ave_CDF_evolved']))
            
            plots = {"Baseline": CDFs['ave_CDF_baseline'],
                     "Evolved": CDFs['ave_CDF_evolved'],
                     "Benchmark": CDFs['ave_CDF_benchmark']}

            save_CDF("CDF", plots)
            save_CDF(plot_name, plots, part="bottom")
            save_CDF(plot_name, plots, part="top")

        return stats['ave_improvement_R']

    def run_with_plot(self):
        """
        Run the network continuously with live CDF plot.
        
        :return: Nothing.
        """
        from matplotlib import animation
        
        def go_frame_go():
            """ Run the network with a live plot of the CDF of Downlink rates
            """
            self.frame += 1
            self.run_full_frame(two=self.PRINT, three=self.SAVE)
            self.ax1.clear()
        
            xar = self.CDF_downlink
            yar = self.actual_frequency
            zar = self.CDF_SINR
        
            self.ax1.plot(xar, yar, 'b', self.first_xar, yar, 'r')
            self.ax1.set_ylabel('Cumulative distribution')
            self.ax1.set_ylim([0, 1])
            self.ax1.set_xlabel('Log of downlink rates (bits/sec)', color='b')
        
            self.ax2.plot(zar, yar, 'g', self.first_zar, yar, '#ffa500')
            self.ax2.set_xlabel('Log of SINR', color='g')
    
        self.fig = plt.figure(figsize=[20,15])
        self.ax1 = self.fig.add_subplot(1,1,1)
        self.ax2 = self.ax1.twiny()
        ani = animation.FuncAnimation(self.fig, go_frame_go)
        plt.show()
