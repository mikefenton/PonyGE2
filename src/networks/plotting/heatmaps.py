from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Times New Roman')

from algorithm.parameters import params


def save_heatmap(self, iteration):
    """ Saves a heatmap of the overall state of the network based on cell
        gains, powers, and biases. Includes delineated boundaries and UEs.
    """

    self.masks = []
    
    gain_copy = deepcopy(self.gains)
    power_copy = deepcopy(self.gains)
    for i in range(self.n_all_cells):
        
        if i < 21:
            gain_copy[i] = np.asarray(gain_copy[i]) + self.macro_cells[i]['power']
            power_copy[i] = np.asarray(power_copy[i]) + self.macro_cells[i]['power']
        
        else:
            power_copy[i] = np.asarray(power_copy[i]) + \
                            self.small_cells[i - 21]['power']
            gain_copy[i] = np.asarray(gain_copy[i]) + self.small_cells[i - 21][
                'power'] + self.small_cells[i - 21]['bias']
    
    gain_power_bias = np.asarray(gain_copy)
    gain_power = np.asarray(power_copy)
    
    original_gains = deepcopy(gain_power_bias)
    original_power = deepcopy(gain_power)
    
    gain_power_bias.sort(axis=0)
    gain_power.sort(axis=0)
    
    if params['SHOW'] or params['SAVE']:
        fig = plt.figure(figsize=[20, 20])
        ax = fig.add_subplot(1, 1, 1)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_ylabel('S - N', fontsize=40)
        ax.set_xlabel('W - E', fontsize=40)
    
    for i, cell in enumerate(original_gains):
        mask1 = (cell == gain_power_bias[-1])
        mask2 = (original_power[i] == gain_power[-1])
        self.masks.append([mask1, mask2])
    
    total_heatmap = (10 ** ((original_gains[0] - 30) / 10))
    for i, gain in enumerate(original_gains):
        if i >= 1:
            total_heatmap = total_heatmap + (10 ** ((gain - 30) / 10))
    total_heatmap = 10 * np.log10(1000 * total_heatmap)
    
    if params['SHOW'] or params['SAVE']:
        heat = plt.imshow(total_heatmap, origin='lower')
    
    hist_list = []
    for i, mask in enumerate(self.masks):
        hist2Da, xedgesa, yedgesa = compute_histogram([0, self.size + 1],
                                                           [0, self.size + 1],
                                                           mask[0])
        if i >= 21:
            hist2Db, xedgesb, yedgesb = compute_histogram(
                [0, self.size + 1], [0, self.size + 1], mask[1])
        else:
            hist2Db = None
        hist_list.append([hist2Da, hist2Db])
        if params['SHOW'] or params['SAVE']:
            [L3] = [40]
            if i < 21:
                plt.contour(hist2Da,
                            extent=[xedgesa[0], xedgesa[-1], yedgesa[0],
                                    yedgesa[-1]], levels=[L3],
                            linestyles=['-'], colors=['black'], linewidths=2,
                            alpha=0.75)
            else:
                plt.contour(hist2Da,
                            extent=[xedgesa[0], xedgesa[-1], yedgesa[0],
                                    yedgesa[-1]], levels=[L3],
                            linestyles=['-'], colors=['red'], linewidths=2,
                            alpha=0.5)
                if self.small_cells[i - 21]['bias'] != 0:
                    plt.contour(hist2Db,
                                extent=[xedgesa[0], xedgesa[-1], yedgesa[0],
                                        yedgesa[-1]], levels=[L3],
                                linestyles=['-'], colors=['blue'],
                                linewidths=2, alpha=0.5)
    if params['SHOW'] or params['SAVE']:
        cb = plt.colorbar(heat, fraction=0.0455, pad=0.04)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(30)
        cb.set_label('Gain (dB)', fontsize=40)
        for user in self.users:
            plt.scatter(user['location'][0], user['location'][1])
    if params['SAVE']:
        plt.savefig(params['FILE_PATH'] + 'Network_Map_' +
                    str(iteration) + '.pdf', bbox_inches='tight')
    if params['SHOW']:
        plt.show()
    if params['SHOW'] or params['SAVE']:
        plt.close()
    return total_heatmap, hist_list, xedgesa, yedgesa


def save_helpless_UEs_heatmap(self, iteration):
    """ Saves a heatmap of the overall state of the network based on cell
        gains, powers, and biases. Includes delineated boundaries and the
        locations of UEs who have a maximum SINR of less than 1 (i.e. they
        cannot be scheduled at all).
    """
    total_heatmap = None
    self.masks = []
    
    gain_copy = deepcopy(self.gains)
    power_copy = deepcopy(self.gains)
    for i in range(self.n_all_cells):
        
        if i < 21:
            gain_copy[i] = np.asarray(gain_copy[i]) + self.macro_cells[i][
                'power']
            power_copy[i] = np.asarray(power_copy[i]) + self.macro_cells[i][
                'power']
        
        else:
            power_copy[i] = np.asarray(power_copy[i]) + \
                            self.small_cells[i - 21]['power']
            gain_copy[i] = np.asarray(gain_copy[i]) + self.small_cells[i - 21][
                'power'] + self.small_cells[i - 21]['bias']
    
    gain_power_bias = np.asarray(gain_copy)
    gain_power = np.asarray(power_copy)
    
    original_gains = deepcopy(gain_power_bias)
    original_power = deepcopy(gain_power)
    
    gain_power_bias.sort(axis=0)
    gain_power.sort(axis=0)
    
    if params['SHOW'] or params['SAVE']:
        fig = plt.figure(figsize=[20, 20])
        ax = fig.add_subplot(1, 1, 1)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_ylabel('S - N', fontsize=40)
        ax.set_xlabel('W - E', fontsize=40)
    
    for i, cell in enumerate(original_gains):
        mask1 = (cell == gain_power_bias[-1])
        mask2 = (original_power[i] == gain_power[-1])
        self.masks.append([mask1, mask2])
    
    total_heatmap = (10 ** ((original_gains[0] - 30) / 10))
    for i, gain in enumerate(original_gains):
        if i >= 1:
            total_heatmap = total_heatmap + (10 ** ((gain - 30) / 10))
    total_heatmap = 10 * np.log10(1000 * total_heatmap)
    
    if params['SHOW'] or params['SAVE']:
        heat = plt.imshow(total_heatmap, origin='lower')
    
    hist_list = []
    for i, mask in enumerate(self.masks):
        hist2Da, xedgesa, yedgesa = compute_histogram([0, self.size + 1],
                                                           [0, self.size + 1],
                                                           mask[0])
        if i >= 21:
            hist2Db, xedgesb, yedgesb = compute_histogram(
                [0, self.size + 1], [0, self.size + 1], mask[1])
        else:
            hist2Db = None
        hist_list.append([hist2Da, hist2Db])
        if params['SHOW'] or params['SAVE']:
            [L3] = [40]
            if i < 21:
                plt.contour(hist2Da,
                            extent=[xedgesa[0], xedgesa[-1], yedgesa[0],
                                    yedgesa[-1]], levels=[L3],
                            linestyles=['-'], colors=['black'], linewidths=2,
                            alpha=0.75)
            else:
                plt.contour(hist2Da,
                            extent=[xedgesa[0], xedgesa[-1], yedgesa[0],
                                    yedgesa[-1]], levels=[L3],
                            linestyles=['-'], colors=['red'], linewidths=2,
                            alpha=0.5)
                if self.small_cells[i - 21]['bias'] != 0:
                    plt.contour(hist2Db,
                                extent=[xedgesa[0], xedgesa[-1], yedgesa[0],
                                        yedgesa[-1]], levels=[L3],
                                linestyles=['-'], colors=['blue'],
                                linewidths=2, alpha=0.5)
    if params['SHOW'] or params['SAVE']:
        cb = plt.colorbar(heat, fraction=0.0455, pad=0.04)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(30)
        cb.set_label('Gain (dB)', fontsize=40)
        for ind in self.helpless_UEs:
            user = self.users[ind]
            plt.scatter(user['location'][0], user['location'][1], color="w")
    if params['SAVE']:
        plt.savefig(params['FILE_PATH'] + 'Network_Map_' + str(iteration) + '.pdf',
                    bbox_inches='tight')
    if params['SHOW']:
        plt.show()
    if params['SHOW'] or params['SAVE']:
        plt.close()


def compute_histogram(XRANGE, YRANGE, mask):
    """ This function returns a closed contour that encloses cell coverage
        areas. It computes a 'concave hull' like trace around the set of
        points that constitute a cell coverage area. This is achieved by
        computing a historgram in 2-D over the points.
    """
    
    Bins = 100
    hist2D, xedges, yedges = np.histogram2d(np.nonzero(mask)[0],
                                            np.nonzero(mask)[1],
                                            bins=[Bins, Bins],
                                            range=[XRANGE, YRANGE],
                                            normed=False)
    return hist2D, xedges, yedges


def plot_gain_matrix(self, cell, SAVE=True, SHOW=False, plot_users=True,
                     plot_hotspot=False):
    """ generates a heat map plot of a given cell"""
    
    fig = plt.figure(figsize=[20, 20])
    ax = fig.add_subplot(1, 1, 1)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_ylabel('S - N', fontsize=40)
    ax.set_xlabel('W - E', fontsize=40)
    data = cell['gain']
    # plt.pcolor(data,cmap=plt.cm.Reds,edgecolors='k')
    heat = plt.imshow(data, origin='lower')
    cb = plt.colorbar(heat, fraction=0.0455, pad=0.04)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(30)
    cb.set_label('Gain (dB)', fontsize=40)
    plt.scatter(cell['location'][0], cell['location'][1], color='g')
    
    if plot_hotspot:
        # Find the hottest point in the gain map for the cell
        a = np.unravel_index(cell['gain'].argmax(), cell['gain'].shape)
        plt.scatter(a[1], a[0])
    if plot_users:
        # Plot all the users attached to the cell
        # for i in cell['attached_users']:
        #     user = self.users[i]
        for user in self.users:
            plt.scatter(user['location'][0], user['location'][1])
    
    if SAVE:
        plt.savefig(
            params['FILE_PATH'] + str(cell['id']) +
            '.pdf', bbox_inches='tight')
    if SHOW:
        plt.show()


def generate_maps(self):
    minimum = []
    maximum = []
    optimised = []
    self.SCHEDULING = False
    self.ALL_TOGETHER = True
        
    for frame in range(self.iterations):
        self.iteration = self.scenario + frame
        self.users = self.user_scenarios[frame]
        
        self.reset_to_zero()
        self.update_network(FIST=True)
        answers = self.run_full_frame(first=True, two=self.PRINT)
        
        minimum.append(answers)
        minimum_x = self.CDF_downlink
        minimum_y = self.CDF_SINR
        
        if self.MAP:
            save_heatmap(self, 'Minimum_PB_' + str(
                self.iteration))
        
        self.balance_network()
        answers = self.run_full_frame(two=self.PRINT, three=self.SAVE)
        
        optimised.append(answers)
        optimised_x = self.CDF_downlink
        optimised_y = self.CDF_SINR
        
        if self.MAP:
            save_heatmap(self, 'Optimised_PB_' + str(
                self.iteration))
        
        self.set_benchmark_pb()
        self.update_network(FIST=True)
        answers = self.run_full_frame(first=True, two=self.PRINT)
        
        maximum.append(answers)
        maximum_x = self.CDF_downlink
        maximum_y = self.CDF_SINR
        
        if self.MAP:
            save_heatmap(self, 'Maximum_PB_' + str(self.iteration))
        
        fig = plt.figure()  # figsize=[20,15])
        ax1 = fig.add_subplot(1, 1, 1)
        
        yar = self.actual_frequency
        
        ax1.plot(maximum_x, yar, 'b', label="Power = 35 dBm, Bias = 7 dBm")
        ax1.plot(minimum_x, yar, 'k', label="Power = 23 dBm, Bias = 0 dBm")
        ax1.plot(optimised_x, yar, 'r', label="Optimised Power & Bias")
        
        ax1.set_ylabel('Cumulative distribution')
        ax1.set_xlabel('Log of downlink rates (bits/sec)', color='b')
        ax1.set_ylim([0, 1])
        ax1.legend(loc='best')
        
        if params['SAVE']:
            plt.savefig(
                params['FILE_PATH'] + 'Complete_Comparison_' + str(
                    self.iteration) + '.pdf', bbox_inches='tight')
        if params['SHOW']:
            plt.show()
        if params['SHOW'] or params['SAVE']:
            plt.close()
