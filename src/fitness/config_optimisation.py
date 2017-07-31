from algorithm.parameters import params
from fitness.default_fitness import default_fitness
from networks.pre_compute import pre_compute_network as PCN

import numpy as np


class config_optimisation():
    """ Class for optimising network scheduling by pre-computing network stats.
    """

    maximise = True

    def __init__(self):
        self.training_test = True

        if "NETWORK_OPTIMISATION" in params and params['HOLD_NETWORK']:
            from networks.setup_run import main
            main()

    def __call__(self, phenotype, dist='training'):

        # Get ABS.
        ABS = phenotype[:21]
        ABS_arr = range(1, 8)[ABS % 7]

        # Get number of Small Cells
        n_SCs = params['N_SMALL_TRAINING']
        
        # Get power and bias.
        power = phenotype[21:21+n_SCs]
        bias = phenotype[21+n_SCs:]

        delta_powers = 1 / (1 + np.exp(-power))
        delta_biases = 1 / (1 + np.exp(-bias))

        power_l_limit = np.ones(n_SCs) * 23
        power_u_limit = np.ones(n_SCs) * 35
        diff_power = power_u_limit - power_l_limit
        power_arr = (np.ones(n_SCs) * 23) + (diff_power * delta_powers)
        
        bias_l_limit = np.zeros(n_SCs)
        bias_u_limit = np.ones(n_SCs) * 15.0
        diff_bias = bias_u_limit - bias_l_limit
        bias_arr = bias_l_limit + (diff_bias * delta_biases)

        print("ABS:  ", ABS_arr)
        print("Power:", power_arr)
        print("Bias: ", bias_arr)
        
        quit()

        import networks.Optimise_Network as OPT
        network = OPT.Optimise_Network(
                    SC_power=power_arr,
                    SC_CSB=bias_arr,
                    MC_ABS=ABS_arr,
                    DISTRIBUTION=dist)
        stats = network.run_all()
        
        fitness = stats['sum_log_R']

        if not fitness:
            fitness = default_fitness(self.maximise)

        return fitness
