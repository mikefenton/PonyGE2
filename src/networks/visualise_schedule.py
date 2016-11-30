from os import listdir, getcwd
from subprocess import call

import matplotlib
import scipy.io as io

matplotlib.use('Agg')
import pylab as pl
import numpy as np


def generate_heatmap(path, time):

    # Find all saved schedules
    files = listdir(path + "input/")

    # Get methods used
    methods = []
    for i in files:
        name = "_".join(i.split("_")[:-2])
        if name not in methods:
            methods.append(name)

    for method in methods:
        all_schedules = []
        available = [mat for mat in files if "_".join(mat.split("_")[:-2]) == method]
        for i in available:
            all_schedules.append(io.loadmat(path + "input/" +str(i))['schedule'])

        all_schedules = np.array(all_schedules)
        flat_sch = np.sum(all_schedules, axis=0)
        flat_sch = flat_sch/float((len(available)))

        schedule_vis = {'mat': flat_sch}
        io.savemat(path + 'output/' + method + '.mat', schedule_vis)

        visualise(path, method, flat_sch)
        print("Running MATLAB to generate heatmaps for", method,"...")

        cmd = ["/Applications/MATLAB_R2014b.app/bin/matlab -nosplash -nodisplay -nodesktop -r \"get_plot(\'" + str(method) + "\', \'" + str(time) + "\');exit\""]
        call(cmd, shell=True)
        print("...done")


def visualise(path, method, schedules):
    pl.pcolor(schedules)
    pl.Normalize(clip=True)
    pl.colorbar()
    pl.xlim([0, schedules.shape[1]])
    pl.ylim([0, schedules.shape[0]])
    pl.gca().invert_yaxis()
    pl.xticks(list(range(10)))
    pl.grid(True, color='k', linestyle='-', linewidth=1.5)
    pl.title(method, fontsize=22)
    pl.xlabel('UE', fontsize=18)
    pl.ylabel('Subframe', fontsize=18)
    # pl.show()
    pl.savefig(path + "output/" + str(method) + '_python.pdf', bbox_inches='tight')
    pl.close()


def mane(path, time):

    generate_heatmap(path, time)


if __name__ == "__main__":
    mane(getcwd() + "/Network_Stats/16_8_3_002312/Heatmaps/", "16_8_3_002312")