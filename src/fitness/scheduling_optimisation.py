from algorithm.parameters import params
from fitness.default_fitness import default_fitness
from networks.pre_compute import pre_compute_network as PCN


class scheduling_optimisation():
    """ Class for optimising network scheduling by pre-computing network stats.
    """

    maximise = True

    def __init__(self):
        self.training_test = True

        if "NETWORK_OPTIMISATION" in params and params['HOLD_NETWORK']:
            from networks.setup_run import main
            main()

    def __call__(self, phenotype, dist='training'):

        scheduling_algorithm = phenotype
        scheduling_type = params['GRAMMAR_FILE'].split("/")[-1].split(".")[0]

        if scheduling_type.split("_")[-1] == "complete":
            scheduling_type = phenotype.split("break")[0]
            scheduling_algorithm = phenotype.split("break")[1]

        if params['PRE_COMPUTE'] and dist == "training":
            fitness = PCN.standalone_scheduler.return_pre_compute_fitness(
                scheduling_algorithm, scheduling_type)
        else:
            import networks.Optimise_Network as OPT
            network = OPT.Optimise_Network(
                        SCHEDULING_ALGORITHM=scheduling_algorithm,
                        SCHEDULING_TYPE=scheduling_type,
                        DISTRIBUTION=dist)
            fitness = network.run_all()

        if not fitness:
            fitness = default_fitness(self.maximise)

        return fitness
