import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

from algorithm.parameters import params


def plot_generalisation(keys, lens, opts):
    """
    Plot generalistaion performance of various scheduling methods.
    
    :param keys:
    :param opts:
    :return: Nothing.
    """
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    
    l1 = ax1.plot(keys, opts["Evolved"], color="r", label="Evolved "
                                                          "performance "
                                                          "improvement")
    l2 = ax1.plot(keys, opts["Benchmark"], color="g", label="Benchmark "
                                                            "performance "
                                                            "improvement")
    ax1.axhline(color="k")
    ax1.set_xlabel("Number of UEs per cell")
    ax1.set_ylabel("Percentage SLR improvement over baseline", color="r")
    
    l3 = ax2.plot(keys, lens, color="b", label="Frequency of occurrence")
    ax2.set_ylabel("Frequency of occurrence of cell of size x", color="b")
    box = ax1.get_position()
    
    ax1.set_position(
        [box.x0, box.y0 + box.height * 0.025, box.width, box.height * 0.9])
    ax2.set_position(
        [box.x0, box.y0 + box.height * 0.025, box.width, box.height * 0.9])

    labels_list = l1 + l2 + l3
    labels = [l.get_label() for l in labels_list]
    # ax1.legend(labels_list, labels, loc="best")
    ax1.legend(labels_list, labels, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    
    if params['SAVE']:
        plt.savefig(
            params['FILE_PATH'] + 'generalization.pdf', bbox_inches='tight')
    
    if params['SHOW']:
        plt.show()
    
    if params['SHOW'] or params['SAVE']:
        plt.close()
