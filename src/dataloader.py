import copy
import time
import calendar
import netCDF4
import json
import numpy as np
import statistics as st 
from numpy import savez_compressed
from numpy import load
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import os.path
import itertools
import pandas as pd
plt.rcParams.update({'axes.titlesize': 14})
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('axes', labelsize=14)     # fontsize of the x and y labels 

import torch 
import torch.nn as nn
from models import engine, configs 


class input_dataframe_generater():
    
    def __init__(self, img_path = None, target_path = None, first_date_string = None, last_date_string = None, target_binarizing_thre = None):

        self.img_path          = img_path
        self.first_date_string = first_date_string
        self.last_date_string  = last_date_string
        self.th = target_binarizing_thre



        if img_path is None: 
            self.img_path = configs.DEFAULT_IMAGE_DIR_NAME
        if target_path is None: 
            self.target_path = configs.DEFAULT_TARGET_DIR_NAME


    def dataframe_generation(self):
        start_time = time.time()
        # cleaning the target dataset: 
        target_dataset_cleaned = self.target_dataset_cleaning(self.first_date_string, 
                                                                self.last_date_string)

        # removing the days with incomplete input observations (for each day there should be 149 inpu netcdf files in the folder): 
        output = self.check_target_input_consistency(target_dataset_cleaned) 
        #print("---Generating data: %s seconds ---" % (time.time() - start_time))
        return output

    def target_dataset_cleaning(self, first_date_string, last_date_string):
        """
            1. select time range 
            2. remove null observations
            3. remove days with less than 4 target observations 
        """
        csv_file_name = '{0:s}/Target.csv'.format(self.target_path)
        target = pd.read_csv(csv_file_name, header=0, sep=',')
        #start_time = time.time()

        target['timecycle'] = target['Name'].apply(lambda x: x[9:])
        target['year']      = target['Name'].apply(lambda x: x[:4])
        target['month']     = target['Name'].apply(lambda x: x[4:6])
        target['day']       = target['Name'].apply(lambda x: x[6:8])


        # 1. select time range: 
        dates = target['Date'].values
        good_indices_target = np.where(np.logical_and(
            dates >= int(first_date_string),
            dates <= int(last_date_string)
        ))[0]

        target = target.take(good_indices_target)
        target.reset_index(inplace = True, drop = True)

        # 2. remove null observations
        nan_targets_index = target[target['VIS_Cat'].isnull().values].index.tolist()
        indexes_Not_null = set(range(target.shape[0])) - set(nan_targets_index)
        target = target.take(list(indexes_Not_null)) 
        target.reset_index(inplace = True, drop = True)

        # 3. delete rows with less than 4 lead-time observations per day: 
        target = target[target['Date'].map(target['Date'].value_counts()) == 4]
        target.reset_index(inplace = True, drop = True)

        target = target.drop(columns=['WXCODE'])
        target = target.drop(columns=['VIS_Cat'])

        target = self.binarize_onehot_label_df(target)

        return target


    def check_target_input_consistency(self, target):
        #start_time = time.time()

        hour_prediction_names = '000'
        nam_file_names, mur_file_names, files_datecycle_string = [], [], []
        for (root, dirs, files) in os.walk(configs.DEFAULT_IMAGE_DIR_NAME):
            dirs.sort()
            files.sort()
            if len(files) == 149:
                for file in files: 
                    pathless_file_name = os.path.split(file)[-1]

                    date_string        = pathless_file_name.replace(pathless_file_name[0:5], '').replace(pathless_file_name[-18:], '')
                    datecycle_string   = pathless_file_name.replace(pathless_file_name[0:5], '').replace(pathless_file_name[-13:], '')
                    hourpredic_string  = pathless_file_name.replace(pathless_file_name[0:19], '').replace(pathless_file_name[-9:], '')

                    namOrmur           = pathless_file_name.replace(pathless_file_name[4:], '')

                    if int(date_string) in target['Date'].values:

                        if  hourpredic_string == hour_prediction_names and (namOrmur == 'maps'):
                            nam_file_names.append(os.path.join(root, file))
                            nam_file_names.sort()

                        elif (namOrmur == 'murs'):
                            mur_file_names.append([os.path.join(root, file)]*4)
                            mur_file_names.sort()
                        
                        files_datecycle_string.append(datecycle_string)

        mur_file_names = np.concatenate(mur_file_names)
        
        target = target[target['Name'].isin(files_datecycle_string)]

        target['nam_nc_files_path'] =  nam_file_names
        target['mur_nc_files_path'] =  mur_file_names
        target = target.rename(columns={'Date_Time': 'date_time', "RoundTime":'round_time', "timecycle":'cycletime', "Date": "date", "Name":"date_cycletime", "VIS":"vis", "VIS_Cat": "vis_class"})
        target = target[['date_time', 'round_time', 'date_cycletime', 'date', 'cycletime', 'year', 'month', 'day', 'nam_nc_files_path', 'mur_nc_files_path', 'vis', 'vis_class', 'vis_category']] 
        target.reset_index(inplace = True, drop = True)
        #print("--- %s seconds ---" % (time.time() - start_time))

        return target
        
    def binarize_onehot_label_df(self, target):
        #label_col = self.dataframe['vis']
        target['vis_category']  = target['VIS'].apply(lambda x: 'fog' if x <= self.th else 'non_fog')

        target['vis_class'] = target['vis_category'].astype('category')
        encode_map = {
            'fog' : 1,
            'non_fog': 0

        }

        target['vis_class'].replace(encode_map, inplace = True)

        return target



class return_train_variable_mean_std():

    def __init__(self, dataset, predictor_names = None, lead_time_pred = None):
        self.dataset = dataset
        self.predictor_names = predictor_names 
        self.lead_time_pred = lead_time_pred

    def return_full_timeseries_names(self, netcdf_file_root_name):

        if self.lead_time_pred == 6:
            netcdf_file_006_names  = netcdf_file_root_name[0:57] + '006_input.nc'
            netcdf_file_names_list = [netcdf_file_root_name, netcdf_file_006_names]
            lead_time_steps = ['000', '006']

        elif self.lead_time_pred == 12:
            netcdf_file_006_names = netcdf_file_root_name[0:57] + '006_input.nc'
            netcdf_file_009_names = netcdf_file_root_name[0:57] + '009_input.nc'
            netcdf_file_012_names = netcdf_file_root_name[0:57] + '012_input.nc'
            netcdf_file_names_list = [netcdf_file_root_name, netcdf_file_006_names, netcdf_file_009_names, netcdf_file_012_names]
            lead_time_steps = ['000', '006', '009', '012']
        elif self.lead_time_pred == 24:
            netcdf_file_006_names = netcdf_file_root_name[0:57] + '006_input.nc'
            netcdf_file_012_names = netcdf_file_root_name[0:57] + '012_input.nc'
            netcdf_file_024_names = netcdf_file_root_name[0:57] + '024_input.nc'
            netcdf_file_names_list = [netcdf_file_root_name, netcdf_file_006_names, netcdf_file_012_names, netcdf_file_024_names]
            lead_time_steps = ['000', '006', '012', '024']

        return netcdf_file_names_list, lead_time_steps

    def return_mean_std_dict(self):

        output_dict = {}
        for name in self.predictor_names: 
            
            list_of_mean = []
            for idx in range(len(self.dataset)):

                nc_nam_timeseries_files_path_list = self.dataset.loc[idx]['nam_nc_files_path'] 
                full_timeseries_names, _ = self.return_full_timeseries_names(nc_nam_timeseries_files_path_list)

                for idx, nc_file_name in enumerate(full_timeseries_names):
            
                    dataset_object = netCDF4.Dataset(nc_file_name)

                    this_predictor_matrix = np.array(
                        dataset_object.variables[name][:], dtype = float)


                    this_predictor_leadtime_mean = np.mean(this_predictor_matrix)

                    list_of_mean.append(this_predictor_leadtime_mean)

            this_predictor_mean = st.mean(list_of_mean)
            this_predictor_std  = st.pstdev(list_of_mean)

            output_dict [name] = [this_predictor_mean, this_predictor_std] 

        return output_dict


def split_data_train_valid_test(dataset, year_split_dict = None):

    train_years = year_split_dict['train']
    valid_years = year_split_dict['valid']
    test_years  = year_split_dict['test']


    train_df = dataset[dataset['year'].isin(train_years)]
    train_df.reset_index(inplace = True, drop=True)

    valid_df = dataset[dataset['year'].isin(valid_years)]
    valid_df.reset_index(inplace = True, drop=True)

    test_df  = dataset[dataset['year'].isin(test_years)]
    test_df.reset_index(inplace = True, drop=True)


    return [train_df, valid_df, test_df]



class DataAdopter(): 

    def __init__(self, dataframe, map_structure:str, 
                                predictor_names: list, 
                                lead_time_pred: int, 
                                mean_std_dict: dict,
                                mean_std_mur_dict:dict, 
                                point_geolocation_dic: list): 

        self.dataframe            = dataframe
        self.map_structure        = map_structure
        self.predictor_names      = predictor_names
        self.lead_time_pred       = lead_time_pred
        self.mean_std_dict        = mean_std_dict 
        self.mean_std_mur_dict    = mean_std_mur_dict
        self.point_geolocation_dic= point_geolocation_dic

        #self.dataframe       = self.binarize_onehot_label_df()
        self.label_onehot_df = pd.get_dummies(self.dataframe.vis_category)
        
        
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx): 

        # reading the data date and cycletime: 
        date_time      = self.dataframe.loc[idx]['date_time']
        round_time     = self.dataframe.loc[idx]['round_time']
        date_cycletime = self.dataframe.loc[idx]['date_cycletime']
        visibility     = self.dataframe.loc[idx]['vis']

        # reading the nam map
        nc_nam_timeseries_files_path_list = self.dataframe.loc[idx]['nam_nc_files_path']
        nc_mur_timeseries_files_path_list = self.dataframe.loc[idx]['mur_nc_files_path']
        timeseries_nam_predictor_matrix, timeseries_st_predictor_matrix, timeseries_mur_predictor_matrix = self.read_nc_nam_maps(nc_nam_timeseries_files_path_list, 
                                                                                                                                 nc_mur_timeseries_files_path_list)

        timeseries_predictors_matrix = self.gen_combined_predictor(timeseries_nam_predictor_matrix, 
                                                                   timeseries_st_predictor_matrix, 
                                                                   timeseries_mur_predictor_matrix)
        

        if self.map_structure == '1D': 
            timeseries_nam_predictor_matrix   = timeseries_nam_predictor_matrix.values#.flatten()
            #print(nam_timeseries_predictor_matrix.shape)
            #nam_timeseries_predictor_matrix   = nam_timeseries_predictor_matrix[:, :512]
            timeseries_nam_predictor_matrix   = torch.as_tensor(timeseries_nam_predictor_matrix, dtype=torch.float32)
            
        else: 
            timeseries_nam_predictor_matrix   = timeseries_nam_predictor_matrix[0, :,:, :93]
            timeseries_nam_predictor_matrix   = torch.as_tensor(timeseries_nam_predictor_matrix, dtype = torch.float32)
            timeseries_nam_predictor_matrix   = timeseries_nam_predictor_matrix.permute(2, 0, 1)
        
        #=============================================== reading the target visibility =======================================#
        onehotlabel = self.label_onehot_df.loc[idx]
        onehotlabel = torch.as_tensor(onehotlabel, dtype=torch.long)

        label = self.dataframe.loc[idx]['vis_class'] 
        label = torch.as_tensor(label, dtype=torch.long)#dtype = torch.float32
        #label = torch.unsqueeze(label, dim=0)
        #=============================================== return the sample =======================================#
        sample = {"input": timeseries_nam_predictor_matrix, 
                                        "onehotlabel":onehotlabel, 
                                        "label_class": label, 
                                        "date_time": date_time, 
                                        "round_time": round_time, 
                                        "date_cycletime": date_cycletime,  
                                        "vis": visibility}

        return sample
        #return  nam_timeseries_predictor_matrix, label

    # Need to be updated ==================================================
    def gen_combined_predictor(self, timeseries_nam_predictor_matrix, 
                               timeseries_st_predictor_matrix, 
                               timeseries_mur_predictor_matrix):
        

        upsampled_timeseries_st_predictor_matrix   = scipy.ndimage.zoom(timeseries_st_predictor_matrix, 11.75, order=3)

        surface_index      = np.where(timeseries_mur_predictor_matrix[0, :, :, 0] == -32768)
        listOfCoordinates  = list(zip(surface_index[0], surface_index[1]))

        for m in range(upsampled_timeseries_st_predictor_matrix.shape[0]): 
            for l in listOfCoordinates:
                timeseries_mur_predictor_matrix[m, l(0) , l(1), :] = upsampled_timeseries_st_predictor_matrix[m, l(0) , l(1), :]

            downsampled_timeseries_mur_predictor_matrix = scipy.ndimage.zoom(timeseries_mur_predictor_matrix, 0.0851, order=3)

            NETCDF_TMPDPT      = np.subtract(up_this_nam_tmp2m , up_this_nam_dpt)
            NETCDF_TMPSST      = np.subtract(up_this_nam_tmp2m , this_mur_file)
            NETCDF_DPTSST      = np.subtract(up_this_nam_dpt , this_mur_file)



        timeseries_nam_predictor_matrix = timeseries_nam_predictor_matrix 

        return timeseries_nam_predictor_matrix





    def read_nc_nam_maps(self, netcdf_file_root_names, netcdf_mur_root_names): #

        NETCDF_PREDICTOR_NAMES = self.predictor_names

        if self.lead_time_pred == 6:

            netcdf_file_006_names  = netcdf_file_root_names[0:57] + '006_input.nc'
            netcdf_file_names_list = [netcdf_file_root_names, netcdf_file_006_names]
            lead_time_steps = ['000', '006']

        elif self.lead_time_pred == 12:

            netcdf_file_006_names = netcdf_file_root_names[0:57] + '006_input.nc'
            netcdf_file_009_names = netcdf_file_root_names[0:57] + '009_input.nc'
            netcdf_file_012_names = netcdf_file_root_names[0:57] + '012_input.nc'
            netcdf_file_names_list = [netcdf_file_root_names, netcdf_file_006_names, netcdf_file_009_names, netcdf_file_012_names]
            lead_time_steps = ['000', '006', '009', '012']

        elif self.lead_time_pred == 24:

            netcdf_file_006_names = netcdf_file_root_names[0:57] + '006_input.nc'
            netcdf_file_012_names = netcdf_file_root_names[0:57] + '012_input.nc'
            netcdf_file_024_names = netcdf_file_root_names[0:57] + '024_input.nc'
            netcdf_file_names_list = [netcdf_file_root_names, netcdf_file_006_names, netcdf_file_012_names, netcdf_file_024_names]
            lead_time_steps = ['000', '006', '012', '024']

            # Apr 02: adding mur name dara 
            netcdf_mur_024_names   = netcdf_mur_root_names[0:57] + '009_input.nc'
            netcdf_mur_names_list = [netcdf_mur_024_names, netcdf_mur_024_names, netcdf_mur_024_names, netcdf_mur_024_names]

        timeseries_predictor_matrix = None
        timeseries_surfacetemp_matrix = None
        timeseries_mur_matrix = None

        for idx, nc_file_name in enumerate(netcdf_file_names_list):
            #data_type = netcdf_file_names_list[:4]

            #if data_type == 'maps': 
            dataset_object = netCDF4.Dataset(nc_file_name)

            nam_predictor_matrix = None
            
            for this_predictor_name in NETCDF_PREDICTOR_NAMES:
                
                
                this_predictor_matrix = np.array(
                    dataset_object.variables[this_predictor_name][:], dtype=float
                )

                this_predictor_matrix = np.expand_dims(
                    this_predictor_matrix, axis=-1)
                
                this_predictor_mean  = self.mean_std_dict[this_predictor_name][0]
                this_predictor_std   = self.mean_std_dict[this_predictor_name][1] 

                this_predictor_matrix = (this_predictor_matrix - this_predictor_mean)/this_predictor_std

                if nam_predictor_matrix is None:
                    nam_predictor_matrix = this_predictor_matrix + 0.
                else:
                    nam_predictor_matrix = np.concatenate(
                        (nam_predictor_matrix, this_predictor_matrix), axis=-1
                    )

            surfacetemp_matrix = np.array(
                dataset_object.variables['TMP_surface'][:], dtype=float
            )

            surfacetemp_matrix = np.expand_dims(
                    surfacetemp_matrix, axis=-1)
                
            surfacetemp_mean  = self.mean_std_dict['TMP_surface'][0]
            surfacetemp_std   = self.mean_std_dict['TMP_surface'][1] 

            surfacetemp_matrix = (surfacetemp_matrix - surfacetemp_mean)/surfacetemp_std
                
            #elif data_type == 'murs':

            #    mur_dataset_object = netCDF4.Dataset(nc_file_name)
            #    mur_predictor_matrix = np.array(
            #               mur_dataset_object.variables['analysed_sst'][:], dtype=float
            #           )

            if self.map_structure == '1D':

                time_steps = lead_time_steps[idx]
                this_timeseries_predictor_matrix = self.return_tabular_data(nam_predictor_matrix, time_steps) 

                if timeseries_predictor_matrix is None:
                    timeseries_predictor_matrix = this_timeseries_predictor_matrix

                else:
                    timeseries_predictor_matrix = pd.concat((
                        timeseries_predictor_matrix, this_timeseries_predictor_matrix), axis=1)

            elif self.map_structure == '2D':

                if timeseries_predictor_matrix is None:
                    timeseries_predictor_matrix = nam_predictor_matrix + 0.

                else:
                    timeseries_predictor_matrix = np.concatenate(
                        (timeseries_predictor_matrix, nam_predictor_matrix), axis=-1
                    )



            elif self.map_structure == '3D':

                if timeseries_predictor_matrix is None:
                    timeseries_predictor_matrix = nam_predictor_matrix + 0.

                else:
                    timeseries_predictor_matrix = np.concatenate(
                        (timeseries_predictor_matrix, nam_predictor_matrix), axis=-1
                    )

                if timeseries_surfacetemp_matrix is None:
                    timeseries_surfacetemp_matrix = surfacetemp_matrix + 0.

                else:
                    timeseries_surfacetemp_matrix = np.concatenate(
                        (timeseries_surfacetemp_matrix, surfacetemp_matrix), axis=-1
                    )

            elif self.map_structure == '4D':

                if timeseries_predictor_matrix is None:
                    timeseries_predictor_matrix = nam_predictor_matrix + 0.

                else:
                    timeseries_predictor_matrix = np.concatenate(
                        (timeseries_predictor_matrix, nam_predictor_matrix), axis=0
                    )
        
        # Preparing MUR Dataset: 

        if self.lead_time_pred == 24:
            # Apr 02: adding mur name dara 
            netcdf_mur_024_names   = netcdf_mur_root_names[0:57] + '009_input.nc'
            netcdf_mur_names_list = [netcdf_mur_024_names, netcdf_mur_024_names, netcdf_mur_024_names, netcdf_mur_024_names]


        for idx, mur_file_name in enumerate(netcdf_mur_names_list):

            #if data_type == 'maps': 
            mur_dataset_object = netCDF4.Dataset(mur_file_name)

            mur_predictor_matrix = None
            
            for mur_predictor_name in NETCDF_PREDICTOR_NAMES:
                
                this_mur_predictor_matrix = np.array(
                    mur_dataset_object.variables[mur_predictor_name][:], dtype=float
                )

                this_mur_predictor_matrix = np.expand_dims(
                    this_mur_predictor_matrix, axis=-1)
                
                mur_predictor_mean  = self.mean_std_mur_dict[mur_predictor_name][0]
                mur_predictor_std   = self.mean_std_mur_dict[mur_predictor_name][1] 

                this_mur_predictor_matrix = (this_mur_predictor_matrix - mur_predictor_mean)/mur_predictor_std

                if mur_predictor_matrix is None:
                    mur_predictor_matrix = this_mur_predictor_matrix + 0.
                else:
                    mur_predictor_matrix = np.concatenate(
                        (mur_predictor_matrix, this_mur_predictor_matrix), axis=-1
                    )


            if self.map_structure == '3D':

                if timeseries_mur_matrix is None:
                    timeseries_mur_matrix = mur_predictor_matrix + 0.

                else:
                    timeseries_mur_matrix = np.concatenate(
                        (timeseries_mur_matrix, mur_predictor_matrix), axis=-1
                    )


        return timeseries_predictor_matrix, timeseries_surfacetemp_matrix, timeseries_mur_matrix


    def return_tabular_data(self, nam_predictor_matrix, time_steps):
            
        output_dfs = []

        for key in self.point_geolocation_dic:
            i = self.point_geolocation_dic[key][0]
            j = self.point_geolocation_dic[key][1]
            this_point_df = pd.DataFrame() 
            for idx in range(nam_predictor_matrix.shape[3]):

                this_predictor_matrix = nam_predictor_matrix[0, :, :, idx] 
                feature_value         = this_predictor_matrix[i, j]
                feature_name          = self.predictor_names[idx] + '_' + key + '_' + time_steps

                this_point_df.at[0, feature_name] = feature_value
            
            output_dfs.append(this_point_df)
        
        output = pd.concat(output_dfs, axis=1)

        return output 
        
        

            



#===========
def return_data_loaders (data_config_dict, training_config_dict, Exp_name):

    save_dir = '/data1/fog/SparkMET/EXPs/' + Exp_name
    isExist  = os.path.isdir(save_dir)

    if not isExist:
        os.mkdir(save_dir)
        os.makedirs(save_dir + '/coords')

    train_df_name  = save_dir + '/coords/train.csv' 
    valid_df_name  = save_dir + '/coords/valid.csv'
    test_df_name   = save_dir + '/coords/test.csv' 

    dict_name      = save_dir + '/coords/mean_std.json' 

    # creating the entire data: 
    isDFExists = os.path.isfile(train_df_name)
    if not isDFExists:
        dataset = input_dataframe_generater(img_path               = None, 
                                            target_path            = None, 
                                            first_date_string      = data_config_dict['start_date'], 
                                            last_date_string       = data_config_dict['finish_date'], 
                                            target_binarizing_thre = data_config_dict['vis_threshold']).dataframe_generation()

        # split the data into train, validation and test:
        train_df, valid_df, test_df = split_data_train_valid_test(dataset, year_split_dict = data_config_dict['data_split_dict'])
        train_df.to_csv(train_df_name)
        valid_df.to_csv(valid_df_name)
        test_df.to_csv(test_df_name)
    else: 
        train_df = pd.read_csv(train_df_name)
        valid_df = pd.read_csv(valid_df_name)
        test_df  = pd.read_csv(test_df_name)


    _ = engine.print_report(train_df, valid_df, test_df)

    # calculating the mean and std of training variables: 
    #start_time = time.time()
    isDictExists = os.path.isfile(dict_name)
    if not isDictExists:
        norm_mean_std_dict = return_train_variable_mean_std(train_df, 
                                                    predictor_names = data_config_dict['predictor_names'], 
                                                    lead_time_pred = data_config_dict['lead_time_pred']).return_mean_std_dict()
        #print("--- Normalize data: %s seconds ---" % (time.time() - start_time))
        with open(dict_name, "w") as outfile:
            json.dump(norm_mean_std_dict, outfile)
    else: 
        with open(dict_name, 'r') as file:
            norm_mean_std_dict = json.load(file)

                                                                                                                                                                                                                                       
    train_dataset = DataAdopter(train_df, 
                                map_structure         = data_config_dict['data_straucture'], 
                                predictor_names       = data_config_dict['predictor_names'], 
                                lead_time_pred        = data_config_dict['lead_time_pred'], 
                                mean_std_dict         = norm_mean_std_dict,
                                point_geolocation_dic = data_config_dict['points_coords'])

    valid_dataset = DataAdopter(valid_df, 
                                map_structure         = data_config_dict['data_straucture'], 
                                predictor_names       = data_config_dict['predictor_names'], 
                                lead_time_pred        = data_config_dict['lead_time_pred'], 
                                mean_std_dict         = norm_mean_std_dict,
                                point_geolocation_dic = data_config_dict['points_coords'])

    test_dataset = DataAdopter(test_df, 
                                map_structure         = data_config_dict['data_straucture'], 
                                predictor_names       = data_config_dict['predictor_names'], 
                                lead_time_pred        = data_config_dict['lead_time_pred'],  
                                mean_std_dict         = norm_mean_std_dict,
                                point_geolocation_dic = data_config_dict['points_coords'])


    data_loader_training = torch.utils.data.DataLoader(train_dataset, batch_size= training_config_dict['batch_size'], 
                                                    shuffle=True,  num_workers=8) 
    data_loader_validate = torch.utils.data.DataLoader(valid_dataset, batch_size= training_config_dict['batch_size'], 
                                                    shuffle=False,  num_workers=8)
    data_loader_testing  = torch.utils.data.DataLoader(test_dataset, batch_size= training_config_dict['batch_size'], 
                                                    shuffle=False,  num_workers=8)
    

    return data_loader_training, data_loader_validate, data_loader_testing
        



