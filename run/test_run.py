from FDD_RF_Module.FDD_RF_Modeling import FDD_RF_Modeling
import random

randomruns = 10
randomseedpool = random.sample(range(10000), k=randomruns)
i = 0
for x in randomseedpool:
    i += 1
    print('Starting run ' + str(i))
    test_run = FDD_RF_Modeling(weather = 'Norfolk', 
                               labeling_methodology = 'Simple',
                               feature_selection_methodology = 'Embedded', 
                               aggregate_n_runs = 6 * 24, 
                               number_of_trees = 200,
                               randomseed=x)
    test_run.whole_process_training_and_testing()
