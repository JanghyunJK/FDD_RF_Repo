from FDD_RF_Module.FDD_RF_Modeling import FDD_RF_Modeling
import random
import json

# setting directories
dir_code = "."
dir_data = "./data"

# setting configs json file path
file_configs = dir_code + "/configs.json"

# reading configs json file
print("reading configuration json file from = {}".format(file_configs))
with open(file_configs, "r") as read_file:
    configs = json.load(read_file)

# adding dir_data into configs json file
configs['dir_data'] = dir_data
configs['dir_data_test'] = dir_data + "/test_data"
configs['dir_data_stream'] = dir_data + "/stream_data"

test_run = FDD_RF_Modeling(configs,
                            weather = configs['weather'], 
                            labeling_methodology = configs['labeling_methodology'],
                            feature_selection_methodology = configs['feature_selection_methodology'], 
                            fdd_reporting_frequency_hrs = configs["fdd_reporting_frequency_hrs"],
                            number_of_trees = configs['number_of_trees'],
                            randomseed=1)

test_run.whole_process_streaming_and_costing()
