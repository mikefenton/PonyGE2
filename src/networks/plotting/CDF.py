from matplotlib.ticker import FormatStrFormatter, LogLocator
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

from algorithm.parameters import params


CDFs = {'CDF_downlink': [],
        'CDF_SINR': [],
        'baseline_CDF': [],
        'benchmark_CDF': [],
        'evolved_CDF': [],
        'ave_CDF_baseline': [],
        'ave_CDF_benchmark': [],
        'ave_CDF_evolved': []}


def save_CDF(name, opts, part="whole"):
    """ Saves a CDF plot of the Sum Log R and SINR of all SC users in the
        network.
    """
    
    fig = plt.figure()  # figsize=[18, 12])
    ax1 = fig.add_subplot(1, 1, 1)
    
    yar = np.array(list(range(len(opts['Baseline'])))) / float(
        len(opts['Baseline']))

    if part == "top":
        half_UEs = int(len(yar) / 2)
        ax1.set_ylim([0.5, 1])
        for opt in sorted(opts.keys()):
            ax1.semilogx(opts[opt][half_UEs:], yar[half_UEs:], label=str(opt))
        
    elif part == "bottom":
        half_UEs = int(len(yar) / 2)
        ax1.set_ylim([0, 0.5])
        for opt in sorted(opts.keys()):
            ax1.plot(opts[opt][:half_UEs], yar[:half_UEs], label=str(opt))

        upper_limit = max([max(opts[opt][:half_UEs]) for opt in opts])
        lower_limit = min([min(opts[opt][:half_UEs]) for opt in opts])

        minor_ticks = np.arange(lower_limit, upper_limit, 1)
        ax1.set_xticks(minor_ticks, minor=True)
    
    elif part == "whole":
        ax1.set_ylim([0, 1])
        for opt in sorted(opts.keys()):
            ax1.semilogx(opts[opt], yar, label=str(opt))

    ax1.grid(True)
    
    if part == "top" or part == "whole":
        majorLocator = LogLocator(10)
        majorFormatter = FormatStrFormatter('%.1f')
        ax1.xaxis.set_major_locator(majorLocator)
        ax1.xaxis.set_major_formatter(majorFormatter)
    
    ax1.grid(which='minor', alpha=0.5)
    
    if part == "top":
        major_ticks = np.arange(0.5, 1, 0.1)
    elif part == "bottom":
        major_ticks = np.arange(0, 0.6, 0.1)
    elif part == "whole":
        major_ticks = np.arange(0, 1, 0.1)
    
    ax1.set_yticks(major_ticks, minor=True)

    # ax1.set_title(name, fontsize=30)
    ax1.set_ylabel('Cumulative distribution')  # , fontsize=25)
    ax1.set_xlabel('Downlink rates [Mbps]')  # , fontsize=25)
    
    legend = ax1.legend(loc='lower right', shadow=True)  # , prop={'size': 20})
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    if params['SAVE']:
        plt.savefig(params['FILE_PATH'] + str(name) + '.pdf',
                    bbox_inches='tight')

    if params['SHOW']:
        plt.show()

    if params['SHOW'] or params['SAVE']:
        plt.close()