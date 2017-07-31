import numpy as np

from algorithm.parameters import params
from representation import individual


def config_genome():
    """
    Generate a random genome
    :return:
    """

    ABS = list(np.random.randint(0, 8, size=21))
    power = list(np.random.rand(params['N_SMALL_TRAINING']))
    bias = list(np.random.rand(params['N_SMALL_TRAINING']))

    return ABS + power + bias


def initialisation(size):
    """
    Initialise a population of size and return.
    
    :param size: The size of the desired population.
    :return: A population of individuals.
    """

    return [individual.Individual(config_genome(), None) for _ in range(size)]


def mutation(genome):
    """
    Experimental hillclimbing mutation operator for network config
    optimisation.
    
    :param genome: A genome.
    :return: A mutated genome.
    """
    
    power_bias_codons = (np.random.rand(
        2 * params['N_SMALL_TRAINING']) >= params['p_mut_power_bias'])
    
    genome[21:][power_bias_codons] += np.random.normal(0, params['sigma_mut'],
                                                     np.sum(power_bias_codons))
    
    ABS_codons = (np.random.rand(21) >= params['p_mut_ABSr'])
    n_altered = np.sum(ABS_codons)
    
    if n_altered > 0:
        ABS_deltas = np.abs(np.round(np.random.normal(0, 1.0, n_altered))) + 1
        ABS_deltas[np.random.randint(0, n_altered, int(n_altered / 2))] *= -1.0
        genome[:21][ABS_codons] += ABS_deltas
    
    # genome[:21][ABS_codons] += np.random.randint(-1, 2, np.sum(ABS_codons))
    # genome[genome[ABS_codons] > 7] = 7
    # genome[genome[ABS_codons] < 1] = 1
    
    return genome
