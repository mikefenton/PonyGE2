import types
from copy import copy
from datetime import datetime
from os import getcwd, path, mkdir
from sys import stdout

from algorithm.parameters import params
from utilities.fitness.math_functions import ave
from utilities.stats import trackers
from utilities.stats.save_plots import save_best_fitness_plot

"""Algorithm statistics"""
stats = {
        "gen": 0,
        "best_ever": None,
        "total_inds": 0,
        "regens": 0,
        "invalids": 0,
        "unique_inds": len(trackers.cache),
        "unused_search": 0,
        "ave_genome_length": 0,
        "max_genome_length": 0,
        "min_genome_length": 0,
        "ave_used_codons": 0,
        "max_used_codons": 0,
        "min_used_codons": 0,
        "ave_tree_depth": 0,
        "max_tree_depth": 0,
        "min_tree_depth": 0,
        "ave_tree_nodes": 0,
        "max_tree_nodes": 0,
        "min_tree_nodes": 0,
        "ave_fitness": 0,
        "best_fitness": 0,
        "time_taken": 0,
        "total_time": 0
}


def get_stats(individuals, end=False):
    """
    Generate the statistics for an evolutionary run. Save statistics to
    utilities.trackers.stats_list. Print statistics. Save fitness plot
    information.
    
    :param individuals: A population of individuals for which to generate
    statistics.
    :param end: Boolean flag for indicating the end of an evolutionary run.
    :return: Nothing.
    """

    if end or params['VERBOSE'] or not params['DEBUG']:

        # Time Stats
        trackers.time_list.append(datetime.now())
        stats['time_taken'] = trackers.time_list[-1] - trackers.time_list[-2]
        stats['total_time'] = trackers.time_list[-1] - trackers.time_list[0]
        
        # Population Stats
        stats['total_inds'] = params['POPULATION_SIZE'] * (stats['gen'] + 1)
        stats['unique_inds'] = len(trackers.cache)
        stats['unused_search'] = 100 - stats['unique_inds'] / \
                                       stats['total_inds']*100
        stats['best_ever'] = max(individuals)

        available = [i for i in individuals if not i.invalid]

        # Genome Stats
        genome_lengths = [len(i.genome) for i in available]
        stats['max_genome_length'] = max(genome_lengths)
        stats['ave_genome_length'] = ave(genome_lengths)
        stats['min_genome_length'] = min(genome_lengths)

        # Used Codon Stats
        codons = [i.used_codons for i in available]
        stats['max_used_codons'] = max(codons)
        stats['ave_used_codons'] = ave(codons)
        stats['min_used_codons'] = min(codons)

        # Tree Depth Stats
        depths = [i.depth for i in available]
        stats['max_tree_depth'] = max(depths)
        stats['ave_tree_depth'] = ave(depths)
        stats['min_tree_depth'] = min(depths)

        # Tree Node Stats
        nodes = [i.nodes for i in available]
        stats['max_tree_nodes'] = max(nodes)
        stats['ave_tree_nodes'] = ave(nodes)
        stats['min_tree_nodes'] = min(nodes)

        # Fitness Stats
        fitnesses = [i.fitness for i in available]
        stats['ave_fitness'] = ave(fitnesses)
        stats['best_fitness'] = stats['best_ever'].fitness

    # Save fitness plot information
    if params['SAVE_PLOTS'] and not params['DEBUG']:
        if not end:
            trackers.best_fitness_list.append(stats['best_ever'].fitness)
       
        if params['VERBOSE'] or end:
            save_best_fitness_plot()

    # Print statistics
    if params['VERBOSE']:
        if not end:
            print_generation_stats()
    
    elif not params['SILENT']:
        perc = stats['gen'] / (params['GENERATIONS']+1) * 100
        stdout.write("Evolution: %d%% complete\r" % perc)
        stdout.flush()

    # Generate test fitness on regression problems
    if hasattr(params['FITNESS_FUNCTION'], "training_test") and end:
        stats['best_ever'].training_fitness = copy(stats['best_ever'].fitness)
        stats['best_ever'].test_fitness = params['FITNESS_FUNCTION'](
            stats['best_ever'].phenotype, dist='test')
        stats['best_ever'].fitness = stats['best_ever'].training_fitness

    # Save statistics
    if not params['DEBUG']:
        save_stats_to_file(end)
        if params['SAVE_ALL']:
            save_best_ind_to_file(end, stats['gen'])
        elif params['VERBOSE'] or end:
            save_best_ind_to_file(end, "best")

    if end and not params['SILENT']:
        print_final_stats()


def print_generation_stats():
    """Print the statistics for the generation and individuals"""

    print("______\n")
    for stat in sorted(stats.keys()):
        print(" ", stat, ": \t", stats[stat])
    print("\n")


def print_final_stats():
    """
    Prints a final review of the overall evolutionary process
    """

    if hasattr(params['FITNESS_FUNCTION'], "training_test"):
        print("\n\nBest:\n  Training fitness:\t",
              stats['best_ever'].training_fitness)
        print("  Test fitness:\t\t", stats['best_ever'].test_fitness)
    else:
        print("\n\nBest:\n  Fitness:\t", stats['best_ever'].fitness)
    print("  Phenotype:", stats['best_ever'].phenotype)
    print("  Genome:", stats['best_ever'].genome)
    print_generation_stats()
    print("\nTime taken:\t", stats['total_time'])


def save_stats_to_file(end=False):
    """Write the results to a results file for later analysis"""
    if params['VERBOSE']:
        filename = params['FILE_PATH'] + str(params['TIME_STAMP']) + \
                   "/stats.tsv"
        savefile = open(filename, 'a')
        for stat in sorted(stats.keys()):
            savefile.write(str(stat) + "\t" + str(stats[stat]) + "\t")
        savefile.write("\n")
        savefile.close()

    elif end:
        filename = params['FILE_PATH'] + str(params['TIME_STAMP']) + \
                   "/stats.tsv"
        savefile = open(filename, 'a')
        for item in trackers.stats_list:
            for stat in sorted(item.keys()):
                savefile.write(str(item[stat]) + "\t")
            savefile.write("\n")
        savefile.close()

    else:
        trackers.stats_list.append(copy(stats))


def save_stats_headers():
    """
    Saves the headers for all stats in the stats dictionary.
    
    :return: Nothing.
    """

    filename = params['FILE_PATH'] + str(params['TIME_STAMP']) + "/stats.tsv"
    savefile = open(filename, 'w')
    for stat in sorted(stats.keys()):
        savefile.write(str(stat) + "\t")
    savefile.write("\n")
    savefile.close()


def save_final_time_stats():
    """
    Appends the total time taken for a run to the stats file.
    """

    filename = params['FILE_PATH'] + str(params['TIME_STAMP']) + "/stats.tsv"
    savefile = open(filename, 'a')
    savefile.write("Total time taken: \t" + str(stats['total_time']))
    savefile.close()


def save_params_to_file():
    """
    Save evolutionary parameters in a parameters.txt file. Automatically
    parse function and class names.

    :return: Nothing
    """

    # Generate file path and name.
    filename = params['FILE_PATH'] + str(params['TIME_STAMP']) + \
               "/parameters.txt"
    savefile = open(filename, 'w')

    # Justify whitespaces for pretty printing/saving.
    col_width = max(len(param) for param in params.keys())

    for param in sorted(params.keys()):
        savefile.write(str(param) + ": ")
        spaces = [" " for _ in range(col_width - len(param))]

        if isinstance(params[param], types.FunctionType):
            # Object is a function, save function name.
            savefile.write("".join(spaces) + str(params[
                                                     param].__name__) + "\n")
        elif hasattr(params[param], '__call__'):
            # Object is a class instance, save name of class instance.
            savefile.write("".join(spaces) + str(params[
                                                     param].__class__.__name__) + "\n")
        else:
            # Write object as normal.
            savefile.write("".join(spaces) + str(params[param]) + "\n")

    savefile.close()


def save_best_ind_to_file(end=False, name="best"):

    filename = params['FILE_PATH'] + str(params['TIME_STAMP']) + "/" + \
               str(name) + ".txt"
    savefile = open(filename, 'w')
    savefile.write("Generation:\n" + str(stats['gen']) + "\n\n")
    savefile.write("Phenotype:\n" + str(stats['best_ever'].phenotype) + "\n\n")
    savefile.write("Genotype:\n" + str(stats['best_ever'].genome) + "\n")
    savefile.write("Tree:\n" + str(stats['best_ever'].tree) + "\n")
    if hasattr(params['FITNESS_FUNCTION'], "training_test"):
        if end:
            savefile.write("\nTraining fitness:\n" +
                           str(stats['best_ever'].training_fitness))
            savefile.write("\nTest fitness:\n" +
                           str(stats['best_ever'].test_fitness))
        else:
            savefile.write("\nFitness:\n" + str(stats['best_ever'].fitness))
    else:
        savefile.write("\nFitness:\n" + str(stats['best_ever'].fitness))
    savefile.close()


def generate_folders_and_files():
    """
    Generates necessary folders and files for saving statistics and parameters.
    """

    if params['EXPERIMENT_NAME']:
        path_1 = getcwd() + "/../results/"
        if not path.isdir(path_1):
            mkdir(path_1)
        params['FILE_PATH'] = path_1 + params['EXPERIMENT_NAME'] + "/"
    else:
        params['FILE_PATH'] = getcwd() + "/../results/"

    # Generate save folders
    if not path.isdir(params['FILE_PATH']):
        mkdir(params['FILE_PATH'])
    if not path.isdir(params['FILE_PATH'] + str(params['TIME_STAMP'])):
        mkdir(params['FILE_PATH'] + str(params['TIME_STAMP']))

    save_params_to_file()
    save_stats_headers()