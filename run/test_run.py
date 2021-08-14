from FDD_RF_Module.FDD_RF_Modeling import FDD_RF_Modeling

test_run = FDD_RF_Modeling(weather = 'Norfolk', labeling_methodolog = 'Simple',
                           feature_selection_methodology = 'Embedded', aggregate_n_runs = 6 * 24, number_of_trees = 4,
                           randomseed=20210813)

test_run.whole_process_training_and_testing()
