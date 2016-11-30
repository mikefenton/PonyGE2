# Network optimisation program/function which will take a number of inputs
# for a heteregenous cellular network consisting of combinations of Macro
# Cells (MCs) and Small Cells (SCs) and output network configuration
# settings such that the User Equipment (UE) throughput is maximised.

# Copyright (c) 2014
# Michael Fenton, David Lynch


from random import randint, random, getstate, setstate, seed, choice

import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')
from subprocess import call
from os import getcwd, name
from copy import deepcopy
from sys import exit
import numpy as np
import scipy.io

from algorithm.parameters import params


class Define_Network():
    """ A class for optimising a network given some network conditions."""

    def __init__(self, simulation, iterations=params['ITERATIONS'],
                 scenario=params['SCENARIO']):
        """inputs"""

        self.state = getstate()
        self.simulation = simulation
        self.path = getcwd()
        self.macro_cells = []
        self.small_cells = []
        self.users = []
        self.hotspots = []
        self.powers = []
        self.power_limits = [23, 35]
        self.CSB_limits = [0, params['BIAS_LIMIT']]
        self.gains = []
        self.BS_type = []
        self.BS_locations = []
        self.n_macro_cells = 0
        self.n_small_cells = 0
        self.n_all_cells = 0
        self.n_users = params['N_USERS']
        self.user_locations = []
        self.SINR = 0
        self.total_heatmap = None
        self.perc_signal = None
        self.user_scenarios = []
        self.iterations = iterations
        self.scenario = scenario
        self.stress_percentage = 0
        self.STRESS_TEST = params['STRESS_TEST']
        if self.STRESS_TEST:
            self.stress_percentage = 0  # 25#50
            self.seed = 17  # 4 # 6 7 9
        else:
            self.seed = 13
        np.random.seed(self.seed)
        seed(self.seed)

    def run_all(self, n_small_cells):
        """Run all functions in the class"""

        if self.simulation:
            self.run_simulation(n_small_cells)
        self.get_environmental_encoding()
        self.get_gain_matrix(n_small_cells)
        self.get_bs_type(n_small_cells)
        self.get_bs_locations(n_small_cells)
        self.set_network()
        self.set_small_cells()
        self.set_macro_cells()
        self.setup_UE_distributions()
        setstate(self.state)
        return self

    def run_simulation(self, n_small_cells):
        """runs the Bell network simulation environment, saves the gain matrix,
           the list of powers, and the list of BS types to separate files."""
        print("Running simulation environment in Matlab...")
        if name == 'posix':
            # We're using a unix-based system such as linux or OS X
            cmd = '/Applications/MATLAB_R2014b.app/bin/matlab -nosplash -nodisplay -nodesktop -r "a=\'' + str(self.path) + '/bell_simulation/run_simulation(' + str(n_small_cells) + ')\';run(a)"'
        elif name == 'nt':
            #We're on Windows
            cmd = 'C:\Program Files\MATLAB\R2014b\\bin\matlab -nosplash -nodisplay -nodesktop -r "a=\'' + str(self.path) + '\\bell_simulation\\run_simulation(' + str(n_small_cells) + ')\';run(a)"'
        else:
            print("Error: Operating system not recognised.")
            exit()
        call(cmd, shell=True)
        print("Simulation environment completed")

    def get_gain_matrix(self, n_small_cells):
        """sets the gain matrix for a network of BSs by opening the relevant
           file and reading it in from there. Hideously slow."""
        print("Loading gain matrix for", n_small_cells, "small cells.")
        mat = scipy.io.loadmat('network_data/gain_'+str(n_small_cells)+'.mat')
        # The matrix from Mathlab is 900*900*100 we need 100*900*900, so we
        # 'roll' the z-axis.
        self.gains = np.rollaxis(mat['G_total_dB'], 2, 0)
        self.gains = np.transpose(self.gains, (0,2,1))
        self.size = self.gains.shape[2] - 1
        print("Gain matrix loaded.")

    def get_environmental_encoding(self):
        """ sets the environmental encoding for a network of BSs by opening the
            relevant file and reading it in from there. Hideously slow."""

        mat = scipy.io.loadmat("network_data/environmental_encoding.mat")
        environment = mat['environment_map']
        self.environmental_encoding = np.transpose(environment)

        if None:

            data = self.environmental_encoding
            fig = plt.figure(figsize=[20,20])
            ax = fig.add_subplot(1,1,1)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_ylabel('S - N', fontsize=40)
            ax.set_xlabel('W - E', fontsize=40)

            heat = plt.imshow(data, origin='lower')
            heat.set_cmap('YlGnBu_r')

           # plt.show()

            plt.savefig("Environmental_Encoding.pdf", bbox_inches='tight')

    def get_power_matrix(self, n_small_cells):
        """Gets the powers of all BSs in a network"""

        power_file = 'network_data/BS_power_dBm_' + str(n_small_cells)
        self.BS_power = []
        results = open(power_file, 'r')
        lines = iter(results)
        for line in lines:
            self.powers.append(float(line))
        results.close()

    def get_bs_type(self, n_small_cells):
        """Gets the types of various BSs. 1 for MC, 2 for SC."""

        power_file = 'network_data/BS_type_' + str(n_small_cells)
        results = open(power_file, 'r')
        lines = iter(results)
        for line in lines:
            self.BS_type.append(int(line))
        results.close()

    def get_bs_locations(self, n_small_cells):
        """Gets the locations of various BSs."""

        locations_file = 'network_data/BS_positions_m_' + str(n_small_cells)
        results = open(locations_file, 'r')
        lines = iter(results)
        for line in lines:
            loc = line.split(',')
            location = [int((round(float(loc[0]))+self.size)/2), int((round(float(loc[1]))+self.size)/2)]
            self.BS_locations.append(location)
        results.close()

    def set_network(self):
        """This is the master func which runs all the individual defs
           within the network optimisation class."""

        self.n_macro_cells = self.BS_type.count(1)
        print(self.n_macro_cells, "macro cells")
        self.n_small_cells = self.BS_type.count(2)
        print(self.n_small_cells, "small cells")
        self.n_all_cells = len(self.BS_type)

    def set_small_cells(self):
        """Sets parameters for all small cells"""

        for i in range(self.n_small_cells):
            idx = i
            power = self.power_limits[1]
            bias = 7
            gain = self.gains[i+self.n_macro_cells]
            location = self.BS_locations[i+self.n_macro_cells]
            small = {'power':power, 'gain':gain, 'id':idx, 'location':location, 'macro_interactions':[], 'bias':bias}
            self.small_cells.append(small)

    def set_macro_cells(self):
        """Sets parameters for all macro cells"""

        for i in range(self.n_macro_cells):
            idx = i
            power = 43.3445
            gain = self.gains[i]
            location = self.BS_locations[i]
            macro = {'power':power, 'id':idx, 'gain':gain, 'location':location, 'attached_SCs':[]}
            self.macro_cells.append(macro)

    def setup_UE_distributions(self):
        """ Set up distributions of UEs for all experiments."""

        self.user_scenarios = []
        for i in range(self.iterations):
            self.user_scenarios.append([])
        self.set_hotspots()
        self.seed += self.scenario

        for i in range(self.iterations):
            if i:
                self.seed += 1
            seed(self.seed)
            np.random.seed(self.seed)
            print("Generating UE location data for scenario",
                  i + self.scenario)
            self.distribute_users(scenario=i)

    def set_hotspots(self):
        """ Set hotspot locations for the network.
        """

        self.hotspots = []
        hot_cells = []
        third = int(self.n_small_cells/3)
        num_hotspots = randint(self.n_small_cells-third,
                               self.n_small_cells+third)

        for i in range(num_hotspots):
            prob_hotspot_near_SC = random()
            num_UEs_in_hotspot = randint(5, 20)

            # Place hotspot on the map
            if prob_hotspot_near_SC < 0.10:
                # Pick a random hotspot location
                hot_spot_location = [randint(0, self.size),
                                     randint(0, self.size)]

            else:
                # Pick random SC to place the hotspot near
                if len(hot_cells) < self.n_small_cells:
                    if hot_cells:
                        available = list(set(range(self.n_small_cells)) - set(hot_cells))
                    else:
                        available = list(range(self.n_small_cells))
                    
                    # Pick a SC
                    selected_SC = choice(available)
                    hot_cells.append(selected_SC)
                    hot_spot_location = self.small_cells[selected_SC]['location']
                    
                    while any([loc < 0 or loc > self.size for loc in
                               hot_spot_location]):
                        # Offset hotspots from SC by 5-10 m
                        hot_spot_location[0] += (choice([-1, 1]) * randint(5, 10))
                        hot_spot_location[1] += (choice([-1, 1]) * randint(5, 10))
                else:
                    # Pick a random hotspot location
                    hot_spot_location = [randint(0, self.size),
                                         randint(0, self.size)]

            hotspot = {"location": hot_spot_location,
                       "target": num_UEs_in_hotspot,
                       "users": 0}
            self.hotspots.append(hotspot)

    def distribute_users(self, scenario=1):
        """Sets parameters for all users"""

        hotspots = deepcopy(self.hotspots)

        for user in range(self.n_users):
            prob_placed_in_hotspot = random()
            prob = 0.50
            if (len(hotspots) > 0) and (prob_placed_in_hotspot < prob):
                selected_spot = randint(0, len(hotspots)-1)
                hotspot = hotspots[selected_spot]
                stop = False
                while stop == False:
                    # x = int(round(gauss(hotspot['location'][0], 5)))
                    # y = int(round(gauss(hotspot['location'][1], 5)))
                    theta = np.random.rand()*2*np.pi
                    r = np.random.uniform(0, 80)
                    # Hotspots have a max radius of 24 meters
                    x = hotspot['location'][0] + int(np.sqrt(r)*np.cos(theta))
                    y = hotspot['location'][1] + int(np.sqrt(r)*np.sin(theta))
                    if (0 <= x <= self.size) and (0 <= y<= self.size):
                        env_loc = self.environmental_encoding[y, x]
                        if (env_loc != 0.5):
                            stop = True
                loc = [x, y]
                hotspot["users"] += 1
                if hotspot['users'] >= hotspot['target']:
                    hotspots.remove(hotspot)
            else:
                loc = self.get_random_location()
            idx = user
            user = {"id":idx, "location":loc}
            self.user_scenarios[scenario].append(user)

    def get_random_location(self):
        stop = False
        while stop == False:
            x, y = [randint(0, self.size), randint(0,self.size)]
            env_loc = self.environmental_encoding[y, x]
            if env_loc != 0.5:
                stop = True
        return [x, y]

if __name__ == '__main__':
    self = Define_Network(None, 1260)
    self.run_all()