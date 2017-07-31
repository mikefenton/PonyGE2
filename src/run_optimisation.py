import getopt
import random
import sys
from datetime import datetime

from algorithm.parameters import params, load_params
from networks import setup_run
from networks.comparisons import output_final_average_performance

random.seed(10)
scheduling = {}


def get_command_line_args(command_line_args):
    """
    Parse command line arguments and set parameters correctly.
    
    :param command_line_args: Arguments specified from the command line.
    :return: Nothing.
    """

    try:
        opts, args = getopt.getopt(command_line_args[1:], "",
                                   ["parameters="])
    except getopt.GetoptError as err:
        print("Most parameters need a value associated with them \n",
              "Run python ponyge.py --help for more info")
        print(str(err))
        exit(2)

    for opt, arg in opts:
        if opt == "--parameters":
            load_params("../parameters/" + arg)
        else:
            print("Error: Must specify parameters file with --parameters. "
                  "Cannot accept any other arguements at the moment.")


def generate_time_stamp():
    """
    Generate a time stamp.
    
    :return: Time stamp.
    """
    
    now = datetime.now()
    hms = "%02d%02d%02d" % (now.hour, now.minute, now.second)
    TIME_STAMP = (str(now.year)[2:] + "_" + str(now.month) + "_" +
                  str(now.day) + "_" + hms)
    
    return TIME_STAMP


def mane():
    """allows you to run the Optimise_Network program outside of an
    evolutionary setting."""

    params['TEST_DATA'] = False
    params['PRE_COMPUTE'] = False
    params['SCENARIO'] = 10
    params['ITERATIONS'] = 10
    params['N_SMALL_TRAINING'] = 30
    params['REALISTIC'] = True
    params['N_USERS'] = 1260  # 5000
    params['SHOW'] = False
    params['SAVE'] = True
    params['PRINT'] = True
    params['MAP'] = False
    params['FAIR'] = False
    params['COLLECT_STATS'] = True
    params['TIME_STAMP'] = generate_time_stamp()
    params['PLOT_ALL_UES'] = True
    
    # Hotspot modelling
    params['mean_n_UEs_in_HS_min'] = 5
    params['mean_n_UEs_in_HS_max'] = 20

    params['hotspot_size_min'] = 2.5
    params['hotspot_size_max'] = 12.5

    params['sigma_n_UEs_in_HS_min'] = 0.5
    params['sigma_n_UEs_in_HS_max'] = 2.5
    
    params['dynamic_severity'] = 0.25
    
    # Run setup
    setup_run.main()
    
    if params['SAVE']:
        setup_run.generate_save_folder(params['TIME_STAMP'])
    
    """ Compares given solutions against both baseline and benchmark results.
        Returns the average difference in performance between the evolved and
        benchmark methods (in terms of a percentage)."""
    
    COMPARE = False

    """ Run the network but using the benchmark methods for ABS and scheduling.
    """
    BENCHMARK = False

    """ Apply all optimisation steps on the network in one go rather than
        running each step (power & bias, ABS, scheduling) separately. Gives the
        same SumLogR value ultimately, but takes much less time as the network
        only needs to be run once. Turning to False will mean the returned
        fitness is the improvement realised by the last applied operation (e.g.
        Power/Bias, ABS, or Scheduling optimisation)."""
    OPT_ALL_TOGETHER = False

    """ Main scheduling method to use on the network. Available scheduling
        methods are defined in the "scheduling" dictionary below."""
    scheduling_type = "high_density_threshold"

    scheduling['original_low_density_threshold'] = "(T20-(pdiv(T13, T14)-pdiv((((np.sign(np.sin(T15))+np.sign((T16*T10)))*(T6*pdiv(pdiv(T17, T7), pdiv(T5, T18))))+(((T13*(T8-T4))-(np.sqrt(abs(T9))+(T21+T18)))+pdiv(((T21+T20)-T17), np.sqrt(abs(pdiv(T11, ABS)))))), (((ABS+pdiv(ABS, T8))-pdiv(pdiv(pdiv(T9, T8), np.sqrt(abs(T19))), (T13*T9)))+((np.sign((T16*T7))+pdiv(ABS, np.sign(T20)))-np.sin((np.log(1+abs(T9))*np.log(1+abs(T20)))))))))"

    scheduling['simplified_low_density_threshold'] = "T20-(pdiv(T13, T14)-pdiv(((2*(T6*pdiv(pdiv(T17, T7), pdiv(T5, T18))))+(((T13*(T8-T4))-(1.5 +(T21+T18)))+pdiv(((T21+T20)-T17), 2.14125255))), ABS))"

    scheduling['simplified_threshold'] = "ABS*(0.5+((T6*T17*(T4-T5))-T4))"
    
    scheduling['high_density_threshold'] = "((np.sqrt(abs(np.sqrt(abs(np.log(1+abs(np.sin((pdiv(T8, ABS)*(ABS*T17)))))))))-(np.log(1+abs(pdiv((np.sin(np.sqrt(abs(T8)))-pdiv(pdiv(T16, T15), np.log(1+abs(T17)))), (T6-pdiv(np.log(1+abs(T16)), T17)))))*T7))*ABS)"

    if True:

        if True:
            scheduling["original_sched"] = "self.pdiv(min_SINR, SINR - max_SINR) < (self.pdiv(tan(min_cell_SINR - num_shared), max_cell_SINR * good_slots) + min_cell_SINR)"

            scheduling["genetic_alg"] = "Run a genetic algorithm to compute schedules"

            scheduling["instructive_topology"] = "(pdiv(T3, T9)*(pdiv((np.log(1+abs((pdiv(pdiv(pdiv((+0.1+(T13-T13)), (np.sqrt(abs(T10))*(T3-T5))), (np.sqrt(abs(np.sin(pdiv(pdiv(T12, T7), np.sign((np.log(1+abs(pdiv(T4, T10)))*np.log(1+abs(T11))))))))+(pdiv(T15, T18)+pdiv(T16, T7)))), T2)+T7)))-T12), (T11*T12))--0.7))"

            scheduling["instructive_threshold"] = "pdiv(pdiv((np.log(1+abs((((np.log(1+abs(T5))+T10)-T13)-T10)))*pdiv((T15-T1), (T14-T9))), (np.sqrt(abs(np.sin(T15)))+(np.log(1+abs((T12+pdiv(pdiv(T16, T18), T5))))*(np.sin(T4)+T7)))), (pdiv(((T5-T14)-((np.sqrt(abs(np.sin(np.sin(((pdiv(T3, T1)*np.sqrt(abs(T16)))-(np.sqrt(abs(T2))-np.sqrt(abs(np.log(1+abs(T2))))))))))*(T7+T1))-T15)), ((T18*T18)-np.sqrt(abs(T18))))+((pdiv(T6, T7)-(T5-T17))-np.sign((T7*T8)))))"

            scheduling["evaluative_topology"] = "(pdiv(np.log(1+abs(T10)), (T11-T7))-((((T16+(((np.log(1+abs((np.sqrt(abs((np.log(1+abs(T3))+pdiv(T1, T3))))*((T12+pdiv(pdiv(pdiv(T15, T3), (T13-T5)), T4))*(T9+T4)))))*pdiv(np.sign(T6), np.sign(T4)))-(T16+T5))+(T4-pdiv((T5-T15), (T13*T14)))))+(np.log(1+abs((T7-pdiv(T18, T12))))-T2))-(T3*T4))*pdiv(np.sqrt(abs(T8)), pdiv(T9, T8))))"

            scheduling["evaluative_threshold"] = "(((pdiv(np.sin(pdiv(np.sqrt(abs(pdiv(T17, T13))), T5)), T12)*T20)*((((np.log(1+abs((T3*T2)))*np.sign((T5-T3)))+((T6*pdiv(T2, T3))*np.log(1+abs(T8))))*(T9*pdiv((+0.5-T17), T2)))-((T7*(T3+np.sin(pdiv(T4, T13))))*(np.sqrt(abs(T9))-T14))))+((T15-(np.sign(np.sign(np.sin(np.sin(T2))))*T5))+((pdiv(pdiv(pdiv((T7-T15), (T3-T4)), pdiv(pdiv(T6, T12), pdiv(T20, T6))), ((np.log(1+abs(T5))+(T9-T16))-np.sin(np.log(1+abs(T9)))))+np.sin(pdiv((pdiv(T23, T14)-pdiv(T7, T6)), np.log(1+abs(pdiv(T6, T4))))))+T17)))"

            scheduling['None'] = None

        pb_algorithm = "self.pdiv(ms_log_R, N_s)"#"self.pdiv(R_ms_avg * ms_log_R, R_s_avg * N_s)"#

        abs_algorithm = "self.pdiv(ABS_MUEs, non_ABS_MUEs + ABS_MSUEs)"#"self.pdiv(ABS_MUEs * ABS_MUEs, (float('7') + non_ABS_MUEs * non_ABS_MUEs + float('8')))"#

        import networks.Optimise_Network as OPT

        network = OPT.Optimise_Network(
            PB_ALGORITHM=pb_algorithm,
            ABS_ALGORITHM=abs_algorithm,
            SCHEDULING_ALGORITHM=scheduling[scheduling_type],
            SCHEDULING_TYPE=scheduling_type,
            ALL_TOGETHER=OPT_ALL_TOGETHER,
            DIFFERENCE=COMPARE,
            BENCHMARK=BENCHMARK)
        fitness = network.run_all_2()
        output_final_average_performance(network)
        # if params['SAVE']:
        #
        #     visualise_schedule.mane(params['FILE_PATH'] +
        #                             "Heatmaps/", params['TIME_STAMP'])

        if network.difference:
            print("Performance differential:", fitness)
        else:
            print("Fitness:", fitness)
        print("\n\n")
        quit()

if __name__ == "__main__":
    
    # Parse command line arguments.
    get_command_line_args(sys.argv)
    
    # Run main program.
    mane()
