from FDD_RF_Module.FDD_RF_Modeling import FDD_RF_Modeling

test_run = FDD_RF_Modeling(weather = 'Norfolk', labeling_methodology = 'Simple',
                           feature_selection_methodology = 'Embedded', aggregate_n_runs = 4 * 24, number_of_trees = 2)

test_run.whole_process_training_and_testing()
