import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
from matplotlib.ticker import FormatStrFormatter

from algorithm.parameters import params
from networks.plotting.CDF import CDFs


def percentage_downlink():
    fig = plt.figure()  # figsize=[15, 8])  # figsize=[18, 12])
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = ax1.twinx()
    
    yar = np.array(range(len(CDFs['baseline_downlinks']))) / float(
        len(CDFs['baseline_downlinks'])) * 100
    
    # x = len(yar)/10
    
    CDFs['baseline_downlinks'].sort()
    CDFs['benchmark_downlinks'].sort()
    CDFs['evolved_downlinks'].sort()
    
    baseline_percs = np.asarray(CDFs['baseline_downlinks'])
    benchmark_percs = np.asarray(CDFs['benchmark_downlinks'])
    evolved_percs = np.asarray(CDFs['evolved_downlinks'])
    
    benchmark_percs = benchmark_percs / baseline_percs * 100 - 100
    evolved_percs = evolved_percs / baseline_percs * 100 - 100
    
    ax1.axhline(color="b", label="Baseline", linewidth=1.5)  # , marker='D',
    # markevery=x, markersize=10)
    ax1.plot(yar, benchmark_percs, label="Benchmark", color="g",
             linewidth=1.5)  # , marker='^', markevery=x, markersize=10)
    ax1.plot(yar, evolved_percs, label="Evolved", color="r",
             linewidth=1.5)  # , marker='o', markevery=x, markersize=10)
    
    ax1.grid(True)
    
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    minor_ticks = np.arange(0, 100, 5)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.grid(which='minor', alpha=0.8)
    
    # major_ticks = np.arange(-40, 300, 10)
    # ax1.set_yticks(major_ticks)
    # ax1.grid(which='major', alpha=0.5)
    
    # ax1.set_title("Performance Relative to Baseline")#, fontsize=30)
    ax1.set_ylabel('% Improvement in Downlink Rates')  # , fontsize=25)
    ax1.set_xlabel('Percentile')  # , fontsize=25)
    
    ax1.set_ylim([-100, 400])
    
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    
    legend = ax1.legend(loc='best', shadow=True)  # , prop={'size': 20})
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    
    if params['SAVE']:
        plt.savefig(
            params['FILE_PATH'] + 'downlinks.pdf', bbox_inches='tight')
    if params['SHOW']:
        plt.show()
    
    plt.close()
