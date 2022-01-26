import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

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

    ################################################################################################
    #----------------------------------------------------------------------------------------------#
    ################################################################################################

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

    ################################################################################################
    #----------------------------------------------------------------------------------------------#
    ################################################################################################

    def get_timeinterval(self, os_timestamp):
    
        #converting timestamp to pandas datetime
        timestamp = pd.to_datetime(os_timestamp)
        # inferring timestep (frequency) from the dataframe
        dt = timestamp.diff().value_counts().idxmax() # in pandas timedelta
        dt = int(dt.value/(10**9)/60) # in minutes
        print("[Training/Testing Data Processing] timestep ({} min) inferred from the dataframe".format(dt))
        return dt

    ################################################################################################
    #----------------------------------------------------------------------------------------------#
    ################################################################################################
    
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

    ################################################################################################
    #----------------------------------------------------------------------------------------------#
    ################################################################################################

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

    ################################################################################################
    #----------------------------------------------------------------------------------------------#
    ################################################################################################

    def create_folder_structure(self):
        print('[Preprocessing] creating folder structure...')
        folders = ['models/', 'results/', 'data/']
        for folder in folders:
            if not os.path.exists(os.path.join(self.root_path, folder)):
                os.makedirs(os.path.join(self.root_path, folder))

    ################################################################################################
    #----------------------------------------------------------------------------------------------#
    ################################################################################################

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
            test_data_file_name_list = [os.path.basename(x) for x in glob.glob(self.configs['dir_data_test'] + '/*.csv')]
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
            list_streaming = []

            for stream_data_file_name in self.test_simulation_data_file_list:
                print('[Training/Testing Data Processing] reading stream data for predicting: ' + stream_data_file_name)
                fault_streaming = stream_data_file_name.split("_preformatted_")[1]
                list_streaming.append(fault_streaming)
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
                # temp_stream_FDD_data_test.to_csv(self.configs['dir_data_stream'] + f'\\{stream_data_file_name.replace("_preformatted","_formatted")}')
                fault_inputs_output_test = pd.concat([fault_inputs_output_test, temp_stream_FDD_data_test], axis = 0)

            fault_inputs_output_test = fault_inputs_output_test.reset_index(drop = True)
            self.important_features = pd.read_csv(f'results/important_features_{self.weather}.csv')['important_features'].tolist()
            print('[Testing Data Processing] filtering data only with important features')
            self.inputs_test = fault_inputs_output_test[self.important_features]
            self.list_streaming = list_streaming

            if self.labeling_methodology == 'Simple':
                self.output_test = fault_inputs_output_test.iloc[:,-1]

        else:
            raise Exception("Error! Enter either 'train' or 'test' for train_or_test")

    ################################################################################################
    #----------------------------------------------------------------------------------------------#
    ################################################################################################

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

    ################################################################################################
    #----------------------------------------------------------------------------------------------#
    ################################################################################################

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

    ################################################################################################
    #----------------------------------------------------------------------------------------------#
    ################################################################################################

    def impact_estimation(self):

        # have to install plotly & kaleido
        # pip install plotly
        # pip install -U kaleido

        # read output file that includes FDD results for every reporting time step
        # convert the FDD results output into simulation timestep
        # read baseline simulation file
        # read faulted simulation file(s)
        # calculate electricity usage difference
        # calculate natural gas usage difference
        # calculate thermal comfort difference
        # convert electricity/gas to $
        # convert thermal comfort to $
        # report and visulize results

        # reading FDD results file
        df_result = pd.read_csv(self.configs["dir_results"] + "/{}_{}.csv".format( self.configs["weather"], self.configs["train_test_apply"] ))

        # reshaping FDD results for each year
        count_day = 24/self.configs['fdd_reporting_frequency_hrs']
        count_year = int(365*count_day)
        num_batchs = int(df_result.shape[0]/count_year)
        df_result = pd.DataFrame(df_result.values.reshape((count_year,num_batchs)))

        # calculating impact for each stream data
        for count, batch in enumerate(df_result.columns):

            # re-reading fault name generated from the synthetic data creation
            fault_streaming = self.list_streaming[count]
            print("[Estimating Fault Impact] processing for synthetic data including fault [{}]".format(fault_streaming))
        
            # creating empty dataframe with timestamp
            freq = str(self.configs['cost_est_timestep_min']) + 'min'
            df_combined = pd.DataFrame([])
            df_combined['reading_time'] = pd.date_range( self.configs['simulation_date_start'], self.configs['simulation_date_end'], freq=freq)
            df_combined = df_combined.set_index(['reading_time'])[:-1]

            # expanding FDD results into the same user-specified timestep
            df_result_exp = np.repeat(list(df_result[batch].values), self.configs['fdd_reporting_frequency_hrs']*int(60/self.configs['cost_est_timestep_min']))
            df_result_exp = pd.DataFrame(df_result_exp)
            df_result_exp.columns = ['FaultType']
            df_result_exp.index = df_combined.index

            # setting timestamp for raw simulation data
            freq_raw = str(self.configs['simulation_timestep_min']) + 'min'
            df_index = pd.DataFrame([])
            df_index['reading_time'] = pd.date_range( self.configs['simulation_date_start'], self.configs['simulation_date_end'], freq=freq_raw)
            df_index = df_index.set_index(['reading_time'])[:-1]
            timestamp_last = df_index.index[-1]

            # reading baseline simulation results
            print("[Estimating Fault Impact] reading baseline simulation results")
            df_baseline = pd.read_csv(self.configs['dir_data']+"/"+self.configs['weather']+"/baseline.csv", usecols=[self.configs['sensor_name_elec'], self.configs['sensor_name_ng']])
            df_baseline.columns = [f"baseline_elec_{self.configs['sensor_unit_elec']}",f"baseline_ng_{self.configs['sensor_unit_ng']}"]
            df_baseline.index = df_index.index
            df_baseline = df_baseline.resample(str(self.configs['cost_est_timestep_min'])+"T").mean()

            # recreating FDD results with unique fault type (consecutive fault types are removed)
            df_unique = df_result_exp[(df_result_exp.ne(df_result_exp.shift())).any(axis=1)]
            df_unique = df_unique.reset_index()

            # reading individual fault simulation results (based on FDD results) and creating whole year combined results
            count = 1
            print("[Estimating Fault Impact] combining simulation results from the FDD results")

            df_combined_temp = pd.DataFrame()
            for index, row in df_unique.iterrows():
                            
                # specifying start and stop timestamp for each detected fault
                rownum_current = df_unique.loc[df_unique.index==index,:].index[0]
                timestamp_start = df_unique.iloc[rownum_current,:].reading_time
                if rownum_current+1 < df_unique.shape[0]:
                    timestamp_end = df_unique.iloc[rownum_current+1,:].reading_time - pd.Timedelta(minutes=self.configs["simulation_timestep_min"])
                else:
                    timestamp_end = timestamp_last
                    
                print(f"[Estimating Fault Impact] prossessing [{row['FaultType']} ({count}/{df_unique.shape[0]})] from the FDD results covering {timestamp_start} to {timestamp_end}")
                    
                count_file = 0
                for file in glob.glob(self.configs['dir_data']+"/"+self.configs['weather']+f"/*{row['FaultType']}*"):
                    print(f"[Estimating Fault Impact] reading [{file}] file")
                    count_file += 1
                    if count_file == 1:
                        df_temp = pd.read_csv(file, usecols=[self.configs['sensor_name_elec'], self.configs['sensor_name_ng']])
                        df_temp.index = df_index.index
                        df_temp = df_temp.resample(str(self.configs['cost_est_timestep_min'])+"T").mean()
                        df_temp = df_temp[timestamp_start:timestamp_end]
                        df_fault = df_temp.copy()
                    else:
                        df_temp = pd.read_csv(file, usecols=[self.configs['sensor_name_elec'], self.configs['sensor_name_ng']])
                        df_temp.index = df_index.index
                        df_temp = df_temp.resample(str(self.configs['cost_est_timestep_min'])+"T").mean()
                        df_temp = df_temp[timestamp_start:timestamp_end]
                        df_fault += df_temp
                        
                # averaging all fault intensity simulations for a single fault and merging into combined dataframe
                print(f"[Estimating Fault Impact] averaging all fault intensity simulations for a single fault and merging into combined dataframe")
                df_fault = df_fault/count_file
                df_combined_temp = pd.concat([df_combined_temp, df_fault])
                count+=1

            # creating combined dataframe from baseline and faulted timeseries data
            df_combined_temp.columns = [f"faulted_elec_{self.configs['sensor_unit_elec']}",f"faulted_ng_{self.configs['sensor_unit_ng']}"]
            df_combined_temp.index = pd.to_datetime(df_combined_temp.index)
            df_combined = pd.merge(df_combined, df_baseline, how='outer', left_index=True, right_index=True)
            df_combined = pd.merge(df_combined, df_combined_temp, how='outer', left_index=True, right_index=True)

            # creating columns of energy usage differences
            df_combined['diff_elec'] = df_combined["faulted_elec_{}".format(self.configs["sensor_unit_elec"])] - df_combined["baseline_elec_{}".format(self.configs["sensor_unit_elec"])]
            df_combined['diff_ng'] = df_combined["faulted_ng_{}".format(self.configs["sensor_unit_ng"])] - df_combined["baseline_ng_{}".format(self.configs["sensor_unit_ng"])]

            # creating columns for time, date, and month
            df_combined['Time'] = pd.to_datetime(df_combined.index).time
            df_combined['Time'] = df_combined.Time.astype(str).str.rsplit(":",1, expand=True).iloc[:,0]
            df_combined['Date'] = pd.to_datetime(df_combined.index).date
            df_combined['Month'] = pd.to_datetime(df_combined.index).month

            # calculate monthly and annual excess energy usages
            if (self.configs['sensor_unit_ng']=='W') & (self.configs['sensor_unit_elec']=='W'):
                df_monthly = df_combined.groupby(['Month'])[["baseline_elec_{}".format(self.configs["sensor_unit_elec"]),"baseline_ng_{}".format(self.configs["sensor_unit_ng"]),'diff_elec','diff_ng']].sum()/1000/(60/self.configs['cost_est_timestep_min']) #convert W to kWh
                base_annual_elec = round(df_monthly["baseline_elec_{}".format(self.configs["sensor_unit_elec"])].sum()) # in kWh
                base_annual_ng = round(df_monthly["baseline_ng_{}".format(self.configs["sensor_unit_ng"])].sum()) # in kWh
                diff_annual_elec = round(df_monthly.sum()['diff_elec']) # in kWh
                diff_annual_ng = round(df_monthly.sum()['diff_ng']) # in kWh
                perc_annual_elec = round(diff_annual_elec/base_annual_elec*100, 3) # in %
                perc_annual_ng = round(diff_annual_ng/base_annual_ng*100, 3) # in %
            else:
                # add other unit conversions
                print("[Estimating Fault Impact] unit conversion from {} for electricity and {} for natural gas to kWh is not currently supported".format(self.configs['sensor_unit_elec'],self.configs['sensor_unit_ng']))
                
            path_impact = self.configs['dir_results'] + "/{}_FDD_impact_table_{}.csv".format(self.configs["weather"], fault_streaming)
            print("[Estimating Fault Impact] saving fault impact estimation summary in {}".format(path_impact))
            df_combined.to_csv(path_impact)
            self.configs["excess_elec_kWh"] = diff_annual_elec
            self.configs["excess_ng_kWh"] = diff_annual_ng
            self.configs["excess_elec_%"] = perc_annual_elec
            self.configs["excess_ng_%"] = perc_annual_ng 

            # plot setting
            title_font_size = 12
            colorbar_font_size = 12
            tick_font_size = 12
            anot_font_size = 14
            fontfamily = 'verdana'
            barwidth = 0.75
            colorscale=[
                [0.0, 'rgb(5,48,97)'],
                [0.1, 'rgb(33,102,172)'],
                [0.2, 'rgb(67,147,195)'],
                [0.3, 'rgb(146,197,222)'],
                [0.4, 'rgb(209,229,240)'],
                [0.5, 'rgb(247,247,247)'],
                [0.6, 'rgb(253,219,199)'],
                [0.7, 'rgb(244,165,130)'],
                [0.8, 'rgb(214,96,77)'],
                [0.9, 'rgb(178,24,43)'],
                [1.0, 'rgb(103,0,31)']
            ]
            # colorscale=[
            #     [0.0, 'rgb(49,54,149)'],
            #     [0.1, 'rgb(69,117,180)'],
            #     [0.2, 'rgb(116,173,209)'],
            #     [0.3, 'rgb(171,217,233)'],
            #     [0.4, 'rgb(224,243,248)'],
            #     [0.5, 'rgb(255,242,204)'],
            #     [0.6, 'rgb(254,224,144)'],
            #     [0.7, 'rgb(253,174,97)'],
            #     [0.8, 'rgb(244,109,67)'],
            #     [0.9, 'rgb(215,48,39)'],
            #     [1.0, 'rgb(165,0,38)']
            # ]
            color_bar = 'rgb(116,173,209)'
            range_max_elec = max( df_combined['diff_elec'].max() , abs(df_combined['diff_elec'].min()) )
            range_max_ng = max( df_combined['diff_ng'].max() , abs(df_combined['diff_ng'].min()) )


            # plotting
            num_rows = 2
            num_cols = 3
            fig = make_subplots(
                rows=num_rows, 
                cols=num_cols, 
                shared_xaxes=True, 
                vertical_spacing=0.025,
                horizontal_spacing=0.1,
                column_widths=[0.2, 0.4, 0.4],
            )  

            # heatmap
            fig.add_trace(go.Heatmap(
                z=df_combined['diff_elec'],
                x=df_combined['Date'],
                y=df_combined['Time'],
                colorscale='tempo',
                coloraxis='coloraxis1',
            ),
            row=1, col=3)
            fig.add_trace(go.Heatmap(
                z=df_combined['diff_ng'],
                x=df_combined['Date'],
                y=df_combined['Time'],
                colorscale='tempo',
                coloraxis='coloraxis2',
            ),
            row=2, col=3)

            # bar chart
            fig.add_trace(go.Bar(
                x=df_monthly.index,
                y=df_monthly['diff_elec'],
                showlegend=False,
                width=barwidth,
                marker=dict(
                    color=color_bar,
                )
            ),
            row=1, col=2)
            fig.add_trace(go.Bar(
                x=df_monthly.index,
                y=df_monthly['diff_ng'],
                showlegend=False,
                width=barwidth,
                marker=dict(
                    color=color_bar,
                )
            ),
            row=2, col=2)

            # annotation
            if perc_annual_elec > 0:
                text_elec = "+"
            else:
                text_elec = ""

            if perc_annual_ng > 0:
                text_ng = "+"
            else:
                text_ng = ""
            fig.add_annotation(
                x=0.08,
                y=0.75,
                xref="paper",
                yref="paper",
                xanchor='center',
                yanchor='middle',
                text="Excess<br>electricity<br><b>{} kWh/year<br>({}{}%)</b>".format(diff_annual_elec, text_elec, perc_annual_elec),
                font=dict(
                    family=fontfamily,
                    size=anot_font_size,
                    ),
                showarrow=False,
                align="right",
                )
            fig.add_annotation(
                x=0.08,
                y=0.25,
                xref="paper",
                yref="paper",
                xanchor='center',
                yanchor='middle',
                text="Excess<br>natural gas<br><b>{} kWh/year<br>({}{}%)</b>".format(diff_annual_ng, text_ng, perc_annual_ng),
                font=dict(
                    family=fontfamily,
                    size=anot_font_size,
                    ),
                showarrow=False,
                align="right",
                )
            fig.add_annotation(
                x=0.2,
                y=0.75,
                xref="paper",
                yref="paper",
                xanchor='center',
                yanchor='middle',
                text="<b>Electricity [kWh]</b>",
                textangle=270,
                font=dict(
                    family=fontfamily,
                    size=title_font_size,
                    ),
                showarrow=False,
                align="center",
                )
            fig.add_annotation(
                x=0.2,
                y=0.25,
                xref="paper",
                yref="paper",
                xanchor='center',
                yanchor='middle',
                text="<b>Natural gas [kWh]</b>",
                textangle=270,
                font=dict(
                    family=fontfamily,
                    size=title_font_size,
                    ),
                showarrow=False,
                align="center",
                )

            # layout
            fig.update_layout(
                width=900,
                height=400,
                margin=dict(
                    l=0,
                    r=0,
                    t=0,
                    b=0,
                ),
                plot_bgcolor='white',
                coloraxis1=dict(
                    cmin=-range_max_elec,
                    cmid=0,
                    cmax=range_max_elec,
                    colorscale=colorscale, 
                    colorbar = dict(
                        title=dict(
                            text = "Excess electricty [{}]".format(self.configs['sensor_unit_elec']),
                            side='right',
                            font=dict(
                                size=colorbar_font_size,
                                family=fontfamily,
                            ),
                        ),
                        len=0.5,
                        x=1,
                        xanchor='left',
                        y=0.75,
                        yanchor='middle',
                        thickness=23,
                    )
                ),
                coloraxis2=dict(
                    cmin=-range_max_ng,
                    cmid=0,
                    cmax=range_max_ng,
                    colorscale=colorscale, 
                    colorbar = dict(
                        title=dict(
                            text = "Excess natural gas [{}]".format(self.configs['sensor_unit_ng']),
                            side='right',
                            font=dict(
                                size=colorbar_font_size,
                                family=fontfamily,
                            ),
                        ),
                        len=0.5,
                        x=1,
                        xanchor='left',
                        y=0.25,
                        yanchor='middle',
                        thickness=23,
                    ),
                ),
            )

            # axes
            for row in range(1, num_rows+1):
                for col in range(1, num_cols+1):
                    if col==1:
                        fig.update_yaxes(
                            showticklabels=False,
                            row=row, col=col
                        )      
                    elif col==2:     
                        fig.update_yaxes(
                            tickfont = dict(
                                family=fontfamily,
                                size=tick_font_size,
                            ),
                            row=row, col=col
                        )
                    elif col==3:
                        fig.update_yaxes(
                            title = dict( 
                                text="<b>Time</b>",
                                font=dict(
                                    family=fontfamily,
                                    size=title_font_size,
                                ),
                                standoff=0,
                            ),
                            tickfont = dict(
                                family=fontfamily,
                                size=tick_font_size,
                            ),
                            row=row, col=col
                        )      

            fig.update_xaxes(
                title = dict( 
                    text="<b>Date</b>",
                    font=dict(
                        family=fontfamily,
                        size=title_font_size,
                    ),
                ),
                tickfont = dict(
                    family=fontfamily,
                    size=tick_font_size,
                ),
                tickformat="%b",
                dtick="M2",
                row=2, col=3
            )
            fig.update_xaxes(
                title = dict( 
                    text="<b>Month</b>",
                    font=dict(
                        family=fontfamily,
                        size=title_font_size,
                    ),
                ),
                tickfont = dict(
                    family=fontfamily,
                    size=tick_font_size,
                ),
                row=2, col=2
            )
            fig.update_xaxes(
                dtick=1,
                row=1, col=2
            )

            # export
            path_impact_visual = self.configs['dir_results'] + "/{}_FDD_impact_figure_{}.svg".format(self.configs["weather"], fault_streaming)
            print("[Estimating Fault Impact] saving fault impact estimation figure in {}".format(path_impact_visual))
            pio.write_image(fig, path_impact_visual)

    ################################################################################################
    #----------------------------------------------------------------------------------------------#
    ################################################################################################

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
        self.impact_estimation()
