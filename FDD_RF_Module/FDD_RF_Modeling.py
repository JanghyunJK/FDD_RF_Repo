import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

class FDD_RF_Modeling():
    """
    parameters:
        configs: configuration file including settings for the workflow
        weather: the weather under which the building is simulated. Select from
            'AK_Fairbanks', 'FL_Miami', 'KY_Louisville', 'MN_Duluth',
            'SAU_Riyadh', 'TN_Knoxville', 'VA_Richmond'. The default value is
            'TN_Knoxville' where FRP locates.
        labeling_methodology: define the labeling methodology for each data
            point. Select from 'Simple' and 'Energy_Difference'. 'Simple' means
            all data points simulated under one fault type are labeled as that
            fault type. 'Energy Difference' means data will be labeled as fault
            only when electricity or gas consumption is 5% different from
            baseline, otherwise it will be labeled as 'baseline'. The default
            value is 'Simple'.
        feature_selection_methodology: Select from 'None', 'Embedded' and
            'Filter'. 'None' means no feature selection applied. 'Embedded'
            means using Random Forest's embedded feature selection to pre-select
             features. 'Filter' means using ANOVA correlation to select high
             correlated features. The default value is 'None'.
        fdd_reporting_frequency_hrs: FDD reporting frequency in hours
        number_of_trees: number of trees in the random forest algorithm

    functions:
        CDDR_tot: classifier performance metrics. Details can be found here:
            https://www.osti.gov/biblio/1503166
        create_folder_structure: create folder structure for data and results
        inputs_output_generator: generate inputs based on result csv and
            weather data
        get_models: train or load trained machine learning models
        make_predictions: make predictions based on inputs for testing and model
        whole_process_only_training: whole process only for training
        whole_process_only_testing: whole process only for testing
        whole_process_training_and_testing: whole process for both
            training and testing

    """
    def __init__(self, configs, weather = 'TN_Knoxville', labeling_methodology = 'Simple',
     feature_selection_methodology = 'None', fdd_reporting_frequency_hrs = 4,
     number_of_trees = 20, randomseed=2021):
        self.configs = configs
        self.weather = weather
        self.labeling_methodology = labeling_methodology
        self.feature_selection_methodology = feature_selection_methodology
        self.number_of_trees = number_of_trees
        self.fdd_reporting_frequency_hrs = fdd_reporting_frequency_hrs
        self.randomseed = randomseed
        self.root_path = os.getcwd()

    def get_timeinterval(self, os_timestamp):
    
        #converting timestamp to pandas datetime
        timestamp = pd.to_datetime(os_timestamp)
        # inferring timestep (frequency) from the dataframe
        dt = timestamp.diff().value_counts().idxmax() # in pandas timedelta
        dt = int(dt.value/(10**9)/60) # in minutes
        print("[Training/Testing Data Processing] timestep ({} min) inferred from the dataframe".format(dt))
        return dt
    
    def CDDR_tot(self, Real_label, Pred_label):
        CD, CP = 0, 0
        for i,j in zip(Real_label,Pred_label):
            if (i == j) & (i != ('baseline')):
                CD += 1
            if i != 'baseline':
                CP += 1
        if CP ==0:
            CDDR_tot = 0
        else:
            CDDR_tot = CD / CP
        return CDDR_tot

    def TPR_FPR_tot(self, Real_label, Pred_label):
        TP, FP, TCP = 0, 0, 0
        for i,j in zip(Real_label,Pred_label):
            if (i != ('baseline')) & (j != ('baseline')):
                TP += 1
            if (i == {'baseline'}) & (j != ('baseline')):
                FP += 1
            if i != 'baseline':
                TCP += 1
        TPR = TP / TCP
        FPR = FP / TCP
        return TPR, FPR

    def create_folder_structure(self):
        print('[Preprocessing] creating folder structure...')
        folders = ['models/', 'results/', 'data/']
        for folder in folders:
            if not os.path.exists(os.path.join(self.root_path, folder)):
                os.makedirs(os.path.join(self.root_path, folder))

    def inputs_output_generator(self, train_or_test):
        print(f'[Training/Testing Data Processing] generating inputs for {train_or_test}ing...')

        if train_or_test == 'train':
            # read and aggregate raw data
            data_file_name_list = [os.path.basename(x) for x in glob.glob(f"data\\{(self.weather)}\\*.csv")]
            meta_data_file_name = [x for x in data_file_name_list if '_metadata' in x][0]
            simulation_data_file_list = [x for x in data_file_name_list if '_metadata' not in x]
            meta_data = pd.read_csv(f'data\\{self.weather}\\{meta_data_file_name}')

            print('[Training/Testing Data Processing] metadata filename: ' + meta_data_file_name)

            fault_inputs_output = pd.DataFrame([])
            fault_inputs_output_test = pd.DataFrame([])

            for simulation_data_file_name in simulation_data_file_list:
                print('[Training/Testing Data Processing] reading data (for both training and testing): ' + simulation_data_file_name)
                temp_raw_FDD_data = pd.read_csv(f'data\\{self.weather}\\{simulation_data_file_name}')
                timestep = self.get_timeinterval(temp_raw_FDD_data.iloc[:,0]) # in minutes
                aggregate_n_runs = int(60/timestep*self.configs["fdd_reporting_frequency_hrs"])
                temp_raw_FDD_data = temp_raw_FDD_data.groupby(temp_raw_FDD_data.index // (aggregate_n_runs)).mean().iloc[:,0:-8]
                temp_raw_FDD_data['label'] = meta_data.loc[meta_data.sensor_filename == simulation_data_file_name[0:-4]].fault_type.values[0]
                # Splitting training and testing data
                temp_raw_FDD_data_train, temp_raw_FDD_data_test = train_test_split(temp_raw_FDD_data, test_size=self.configs["split_test_size"], random_state=np.random.RandomState(self.randomseed))
                fault_inputs_output = pd.concat([fault_inputs_output, temp_raw_FDD_data_train], axis = 0)
                print('[Training/Testing Data Processing] split and save testing data for ' + simulation_data_file_name)
                temp_raw_FDD_data_test.to_csv(self.configs['dir_data_test'] + f'\\{simulation_data_file_name}')

            ind = pd.DataFrame(temp_raw_FDD_data_test.index.tolist())
            ind.to_csv(self.configs['dir_data_test'] + f'\\{self.weather}_ind.csv')
            fault_inputs_output = fault_inputs_output.reset_index(drop = True)

            # Calculating outputs based on labeling methodology
            if self.labeling_methodology == 'Simple':
                self.output_train = fault_inputs_output.iloc[:,-1]
            elif self.labeling_methodology == 'Energy_Difference':
                pd.set_option('mode.chained_assignment', None)
                electricity_gas_label_df = fault_inputs_output[['electricity_facility [W]', 'gas_facility [W]', 'label']]
                baseline_electricity = electricity_gas_label_df.loc[electricity_gas_label_df.label == 'baseline']['electricity_facility [W]']
                baseline_electricity_repeated = pd.concat([baseline_electricity]* int(len(electricity_gas_label_df) / len(baseline_electricity)), ignore_index=True)
                baseline_gas = electricity_gas_label_df.loc[electricity_gas_label_df.label == 'baseline']['gas_facility [W]']
                baseline_gas_repeated = pd.concat([baseline_gas]* int(len(electricity_gas_label_df) / len(baseline_gas)), ignore_index=True)
                electricity_gas_label_df['baseline_electricity_facility [W]'] = baseline_electricity_repeated
                electricity_gas_label_df['baseline_gas_facility [W]'] = baseline_gas_repeated
                electricity_gas_label_df['electricity_over_threshold'] = (abs(electricity_gas_label_df['electricity_facility [W]'] - electricity_gas_label_df['baseline_electricity_facility [W]']) / electricity_gas_label_df['baseline_electricity_facility [W]']) > 0.05
                electricity_gas_label_df['gas_over_threshold'] = (abs(electricity_gas_label_df['gas_facility [W]'] - electricity_gas_label_df['baseline_gas_facility [W]']) / electricity_gas_label_df['baseline_gas_facility [W]']) > 0.05
                electricity_gas_label_df['electricity_over_threshold_or_gas_over_threshold'] = [x or y for x,y in zip(electricity_gas_label_df['electricity_over_threshold'], electricity_gas_label_df['gas_over_threshold'])]
                electricity_gas_label_df['adjusted_label'] = electricity_gas_label_df['label'] * electricity_gas_label_df['electricity_over_threshold_or_gas_over_threshold']
                electricity_gas_label_df['adjusted_label'] = electricity_gas_label_df['adjusted_label'].replace('', 'baseline')
                self.output_train = electricity_gas_label_df['adjusted_label'].rename('label')
            else:
                raise Exception("Error! Enter either 'Simple' or 'Energy_Difference' for labeling_methodology")

            # Calculating inputs based on feature selection methods
            if self.feature_selection_methodology == 'None':
                self.inputs_train = fault_inputs_output.iloc[:,0:-1]
                self.important_features = fault_inputs_output.iloc[0:2,0:-1].columns.tolist()
            elif self.feature_selection_methodology == 'Embedded':
                model_feature_selection = RandomForestClassifier(n_estimators = 50, random_state=42)
                model_feature_selection.fit(fault_inputs_output.iloc[:,0:-1],self.output_train)
                feature_importance_temp = pd.DataFrame([])
                feature_importance_temp.loc[:,'sensor_name'] = fault_inputs_output.iloc[:,0:-1].columns
                feature_importance_temp.loc[:,'importance'] = model_feature_selection.feature_importances_
                self.important_features = feature_importance_temp.sort_values(
                    by=['importance'], ascending = False).sensor_name.tolist()[0:40]
                self.inputs_train = fault_inputs_output[self.important_features]
            elif self.feature_selection_methodology == 'Filter':
                # Use ANOVA to find most correlated inputs
                select_k_best_classifier = SelectKBest(f_classif, k=40)
                select_k_best_classifier.fit_transform(fault_inputs_output.iloc[:,0:-1], self.output_train)
                self.important_features = select_k_best_classifier.get_support(indices=True)
                self.important_features = fault_inputs_output.iloc[0:2, self.important_features].columns.tolist()
                self.inputs_train = fault_inputs_output[self.important_features]
            # Another 'Systematic' feature selection method combining filter method and embedded method with time series feature extraction is not finished yet.
            # This method is introduced in https://www.sciencedirect.com/science/article/abs/pii/S0360132320307071
            else:
                raise Exception("Error! Enter either 'None', 'Embedded' or 'Filter' for feature_selection_methodology")
            pd.DataFrame(self.important_features, columns = ['important_features']).to_csv(f'results/important_features_{self.weather}.csv', index = None)

        elif train_or_test == 'test':
            # read and aggregate tesing data
            test_data_file_name_list = [os.path.basename(x) for x in glob.glob(self.configs['dir_data_test'] + '*.csv')]
            self.test_simulation_data_file_list = [x for x in test_data_file_name_list if '_ind' not in x]

            fault_inputs_output_test = pd.DataFrame([])

            for simulation_data_file_name in self.test_simulation_data_file_list:
                print('[Testing Data Processing] reading testing data ' + simulation_data_file_name)
                temp_raw_FDD_data_test = pd.read_csv(self.configs['dir_data_test'] + f'\\{simulation_data_file_name}')
                fault_inputs_output_test = pd.concat([fault_inputs_output_test, temp_raw_FDD_data_test], axis = 0)

            fault_inputs_output_test = fault_inputs_output_test.reset_index(drop = True)
            self.important_features = pd.read_csv(f'results/important_features_{self.weather}.csv')['important_features'].tolist()
            print('[Testing Data Processing] filtering data only with important features')
            self.inputs_test = fault_inputs_output_test[self.important_features]

            if self.labeling_methodology == 'Simple':
                self.output_test = fault_inputs_output_test.iloc[:,-1]

        elif train_or_test == 'apply':
            # read and aggregate stream data
            data_file_name_list = [os.path.basename(x) for x in glob.glob(self.configs['dir_data_stream'] + f"\\*.csv")]
            self.test_simulation_data_file_list = [x for x in data_file_name_list if '_preformatted' in x]

            fault_inputs_output_test = pd.DataFrame([])

            for stream_data_file_name in self.test_simulation_data_file_list:
                print('[Training/Testing Data Processing] reading stream data for predicting: ' + stream_data_file_name)
                temp_stream_FDD_data_test = pd.read_csv(self.configs['dir_data_stream'] + f'\\{stream_data_file_name}')
                timestep = self.get_timeinterval(temp_stream_FDD_data_test.iloc[:,0]) # in minutes
                aggregate_n_runs = int(60/timestep*self.configs["fdd_reporting_frequency_hrs"])
                df_label = temp_stream_FDD_data_test.set_index("OS_time").iloc[:,-1:]
                df_label.index = pd.to_datetime(df_label.index)
                resample_frequency = str(self.configs["fdd_reporting_frequency_hrs"]) + "H"
                df_label = df_label.resample(resample_frequency).bfill() # bfill() might not work for all cases
                temp_stream_FDD_data_test = temp_stream_FDD_data_test.copy().iloc[:,0:-1]
                temp_stream_FDD_data_test = temp_stream_FDD_data_test.groupby(temp_stream_FDD_data_test.index // (aggregate_n_runs)).mean().iloc[:,0:-8]
                temp_stream_FDD_data_test['label'] = df_label.iloc[:,0].str.strip().values
                temp_stream_FDD_data_test.to_csv(self.configs['dir_data_stream'] + f'\\{stream_data_file_name.replace("_preformatted","_formatted")}')
                fault_inputs_output_test = pd.concat([fault_inputs_output_test, temp_stream_FDD_data_test], axis = 0)

            fault_inputs_output_test = fault_inputs_output_test.reset_index(drop = True)
            self.important_features = pd.read_csv(f'results/important_features_{self.weather}.csv')['important_features'].tolist()
            print('[Testing Data Processing] filtering data only with important features')
            self.inputs_test = fault_inputs_output_test[self.important_features]

            if self.labeling_methodology == 'Simple':
                self.output_test = fault_inputs_output_test.iloc[:,-1]

        else:
            raise Exception("Error! Enter either 'train' or 'test' for train_or_test")

    def get_models(self, train_or_load_model):

        if train_or_load_model == 'train':
            print(f'[Training Model] training model...')

            FDD_model = RandomForestClassifier(n_estimators = self.number_of_trees, random_state=42)
            FDD_model.fit(self.inputs_train, self.output_train)
            # Calculate training accuracy
            self.output_train_predicted = FDD_model.predict(self.inputs_train)
            self.training_accuracy_CDDR = self.CDDR_tot(self.output_train, self.output_train_predicted)
            self.model = FDD_model
            pickle.dump(self.model, open(f'models/{self.weather}.sav', 'wb'))
            print(f'[Training Model] training model completed! training Accuracy (CDDRtotal) is : {self.training_accuracy_CDDR}')
            # Module to be finished: cross-validation performance

        elif train_or_load_model == 'load':
            print(f'[Applying Trained Model] loading model...')
            self.model = pickle.load(open(f'models/{self.weather}.sav', 'rb'))
            print('[Applying Trained Model] loading model completed!')

        else:
            raise Exception ("Error! Enter either 'train' or 'load' for train_or_load_model")

    def make_predictions(self):
        print('[Applying Trained Model] make and saving predictions...')
        self.output_test_predicted = self.model.predict(self.inputs_test)
        self.testing_accuracy_CDDR = self.CDDR_tot(self.output_test, self.output_test_predicted)
        self.testing_accuracy_TPR, self.testing_accuracy_FPR = self.TPR_FPR_tot(self.output_test, self.output_test_predicted)
        prediction_order = ''.join(self.test_simulation_data_file_list)
        pd.DataFrame(self.output_test_predicted, columns = ['output_test' + prediction_order]).to_csv(f'results/{self.weather}_{self.configs["train_test_apply"]}.csv', index = None)
        logpath = f'results/log.csv'
        logdf = pd.DataFrame({'randomseed': self.randomseed,
                            'weather': self.weather,
                            'labeling methodology': self.labeling_methodology,
                            'feature selection methodology': self.feature_selection_methodology,
                            'number of trees': self.number_of_trees,
                            'fdd reporting frequency hrs': self.fdd_reporting_frequency_hrs,
                            'training CDDR': self.training_accuracy_CDDR,
                            'testing CDDR': self.testing_accuracy_CDDR,
                            'testing TPR': self.testing_accuracy_TPR,
                            'testing FPR': self.testing_accuracy_FPR}, index=[0])
        if not os.path.isfile(logpath):
            logdf.to_csv(logpath, mode='a', index=False)
        else:
            logdf.to_csv(logpath, mode='a', index=False, header=False)
        print(f'[Applying Trained Model] applying model completed! testing Accuracy (CDDRtotal) is : {self.testing_accuracy_CDDR}')
        print('Whole Process Completed!')

    def cost_estimation(self):
        print('[Estimating Fault Cost] ...')

        # read output file that includes FDD results for every reporting time step
        # convert the FDD results output into simulation time step
        # read baseline simulation file
        # read faulted simulation file(s)
        # calculate electricity usage difference
        # calculate natural gas usage difference
        # calculate thermal comfort difference
        # convert electricity/gas to $
        # convert thermal comfort to $
        # report and visulize results

    def whole_process_only_training(self):
        self.create_folder_structure()
        self.inputs_output_generator(train_or_test = 'train')
        self.get_models(train_or_load_model = 'train')

    def whole_process_training_and_testing(self):
        self.create_folder_structure()
        self.inputs_output_generator(train_or_test = 'train')
        self.get_models(train_or_load_model = 'train')
        self.inputs_output_generator(train_or_test = 'test')
        self.make_predictions()

    def whole_process_only_testing(self):
        self.create_folder_structure()
        self.get_models(train_or_load_model = 'load')
        self.inputs_output_generator(train_or_test = 'test')
        self.training_accuracy_CDDR = "na"
        self.make_predictions()

    def whole_process_applying_and_costing(self):
        self.create_folder_structure()
        self.get_models(train_or_load_model = 'load')
        self.inputs_output_generator(train_or_test = 'apply')
        self.training_accuracy_CDDR = "na"
        self.make_predictions()
        #self.cost_estimation()
