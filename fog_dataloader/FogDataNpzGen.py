
import os.path
import torch 
import netCDF4
import json
import numpy as np
import statistics as st 
import scipy.ndimage
import matplotlib.pyplot as plt
import pandas as pd

year_information          = {'2009':['20090101', '20091231'],
                            '2010':['20100101', '20101231'],
                            '2011':['20110101', '20111231'],
                            '2012':['20120101', '20121231'],
                            '2013':['20130101', '20131231'],
                            '2014':['20140101', '20141231'],
                            '2015':['20150101', '20151231'],
                            '2016':['20160101', '20161231'],
                            '2017':['20170101', '20171231'],
                            '2018':['20180101', '20181231'],
                            '2019':['20190101', '20191231'],
                            '2020':['20200101', '20201231']}


data_split_dict_ = {'train': ['2013', '2014', '2015', '2016', '2017'], 
                    'valid': ['2009', '2010', '2011'], 
                    'test': ['2018', '2019', '2020']},
class FogData_Configs():
    def __init__(self, input_path: str, target_path: str, start_date: str, finish_date: str, data_split_dict: dict, 
                 data_straucture: str, lead_time_pred: int, vis_threshold: int, points_coords:dict, predictor_names: list):
        self.input_path            = input_path
        self.target_path           = target_path
        self.start_date            = start_date
        self.finish_date           = finish_date
        self.data_split_dict       = data_split_dict
        self.data_straucture       = data_straucture
        self.lead_time_pred        = lead_time_pred
        self.vis_threshold         = vis_threshold
        self.points_coords         = points_coords
        self.predictor_names       = predictor_names

    def return_config(self):
        data_config_dict = dict(
            input_path      = self.input_path,
            target_path     = self.target_path, 
            start_date      = self.start_date,
            finish_date     = self.finish_date,
            data_split_dict = self.data_split_dict,
            data_straucture = self.data_straucture,
            lead_time_pred  = self.lead_time_pred,
            vis_threshold   = self.vis_threshold,
            points_coords   = self.points_coords,
            predictor_names = self.predictor_names
            )
        
        return data_config_dict


class input_dataframe_generater():
    
    def __init__(self, img_path= None, target_path = None, first_date_string = None, last_date_string = None, 
                 target_binarizing_thre = None,  year_split_dict = None):

        self.img_path          = img_path
        self.first_date_string = first_date_string
        self.last_date_string  = last_date_string
        self.th                = target_binarizing_thre
        self.year_split_dict  = year_split_dict

        if img_path is None: 
            self.img_path = DEFAULT_IMAGE_DIR_NAME
        if target_path is None: 
            self.target_path = DEFAULT_TARGET_DIR_NAME

    def dataframe_generation(self):
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
        for (root, dirs, files) in os.walk(DEFAULT_IMAGE_DIR_NAME):
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
    
    def split_data_train_valid_test(self):
        target_dataset_cleaned = self.target_dataset_cleaning(self.first_date_string, 
                                                                self.last_date_string)
        # removing the days with incomplete input observations (for each day there should be 149 inpu netcdf files in the folder): 
        dataset = self.check_target_input_consistency(target_dataset_cleaned) 

        train_years = self.year_split_dict['train']
        valid_years = self.year_split_dict['valid']
        test_years  = self.year_split_dict['test']

        train_df = dataset[dataset['year'].isin(train_years)]
        train_df.reset_index(inplace = True, drop=True)

        valid_df = dataset[dataset['year'].isin(valid_years)]
        valid_df.reset_index(inplace = True, drop=True)

        test_df  = dataset[dataset['year'].isin(test_years)]
        test_df.reset_index(inplace = True, drop=True)


        return [train_df, valid_df, test_df]



class FogDataNPZ(): 

    def __init__(self, dataframe, map_structure:str, 
                                predictor_names: list, 
                                lead_time_pred: int, 
                                mean_std_dict: dict,
                                point_geolocation_dic: list):

        self.dataframe            = dataframe
        self.map_structure        = map_structure
        self.predictor_names      = predictor_names
        self.lead_time_pred       = lead_time_pred
        self.mean_std_dict        = mean_std_dict 

        self.point_geolocation_dic  = point_geolocation_dic

        #self.dataframe       = self.binarize_onehot_label_df()
        self.label_onehot_df = pd.get_dummies(self.dataframe.vis_category)
    
    def return_data_dict(self):
        len = self.__len__()
        output_dict = {}
        for idx in range(len):
            sample = self.__getitem__(idx)
            date_cycletime = sample['date_cycletime']
            output_dict[date_cycletime] = sample

        return output_dict
    

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
            timeseries_predictors_matrix   = timeseries_predictors_matrix#[:, :, :, :]
            timeseries_predictors_matrix   = torch.as_tensor(timeseries_predictors_matrix, dtype = torch.float32)
            #timeseries_predictors_matrix   = timeseries_predictors_matrix.permute(3, 1, 2, 0)

        #=============================================== reading the target visibility =======================================#
        onehotlabel = self.label_onehot_df.loc[idx]
        onehotlabel = torch.as_tensor(onehotlabel, dtype=torch.long)

        label = self.dataframe.loc[idx]['vis_class'] 
        label = torch.as_tensor(label, dtype=torch.long)#dtype = torch.float32
        #label = torch.unsqueeze(label, dim=0)
        #=============================================== return the sample =======================================#
        sample = {"input": timeseries_predictors_matrix, 
                                        "onehotlabel":onehotlabel, 
                                        "label_class": label, 
                                        "date_time": date_time, 
                                        "round_time": round_time, 
                                        "date_cycletime": date_cycletime,  
                                        "vis": visibility}

        return sample
        #return  nam_timeseries_predictor_matrix, label

    
    def gen_combined_predictor(self, timeseries_nam_predictor_matrix, 
                               timeseries_st_predictor_matrix, 
                               timeseries_mur_predictor_matrix):
        
        tem2m_index = 0
        dpt2m_index = 76
        t = 93

        timeseries_nam_predictor_matrix_out = None
        for m in range(timeseries_st_predictor_matrix.shape[0]): 
            upsampled_timeseries_st_predictor_matrix   = scipy.ndimage.zoom(timeseries_st_predictor_matrix[m, :, :, 0], 11.75, order=3)

            surface_index      = np.where(timeseries_mur_predictor_matrix[m, :, :, 0] == -32768)
            
            listOfCoordinates  = list(zip(surface_index[0], surface_index[1]))
            
            
            for l in listOfCoordinates:

                timeseries_mur_predictor_matrix[m, l[0] , l[1], :] = upsampled_timeseries_st_predictor_matrix[l[0] , l[1]]
            
            downsampled_timeseries_mur_predictor_matrix = scipy.ndimage.zoom(timeseries_mur_predictor_matrix[m, :, :, 0], 0.0851, order=3)
            
            mur_predictor_mean  = self.mean_std_dict['analysed_sst'][0] 
            mur_predictor_std   = self.mean_std_dict['analysed_sst'][1] 
            
            SST = (downsampled_timeseries_mur_predictor_matrix - mur_predictor_mean)/mur_predictor_std

            
            NETCDF_TMPDPT      = np.subtract(timeseries_nam_predictor_matrix[m, :, :, tem2m_index], timeseries_nam_predictor_matrix[m, :, :, dpt2m_index])
            NETCDF_TMPSST      = np.subtract(timeseries_nam_predictor_matrix[m, :, :, tem2m_index], SST)
            NETCDF_DPTSST      = np.subtract(timeseries_nam_predictor_matrix[m, :, :, dpt2m_index], SST)


            this_nam_predictor_matrix = timeseries_nam_predictor_matrix[m, :, :, :]
            this_nam_predictor_matrix = np.insert(this_nam_predictor_matrix, t, SST, axis = 2)
            this_nam_predictor_matrix = np.insert(this_nam_predictor_matrix, t+1, NETCDF_TMPDPT, axis = 2)
            this_nam_predictor_matrix = np.insert(this_nam_predictor_matrix, t+2, NETCDF_TMPSST, axis = 2)
            this_nam_predictor_matrix = np.insert(this_nam_predictor_matrix, t+3, NETCDF_DPTSST, axis = 2)


            this_nam_predictor_matrix = np.expand_dims(this_nam_predictor_matrix, axis = 0)

            if timeseries_nam_predictor_matrix_out is None: 
                timeseries_nam_predictor_matrix_out = this_nam_predictor_matrix
            else: 
                timeseries_nam_predictor_matrix_out = np.concatenate((timeseries_nam_predictor_matrix_out, this_nam_predictor_matrix), axis = 0)

            #t += 93
            #tem2m_index += 93
            #dpt2m_index += 93


        return timeseries_nam_predictor_matrix_out

    def read_nc_nam_maps(self, netcdf_file_root_names, netcdf_mur_root_names): #
        # since I have added SST to the list through normalization I have to remove it here: 
        #self.predictor_names.remove('analysed_sst')
        

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

        timeseries_predictor_matrix = None
        timeseries_surfacetemp_matrix = None
        timeseries_mur_matrix = None

        for idx, nc_file_name in enumerate(netcdf_file_names_list):
            predictor_names_updated = self.predictor_names
            dataset_object = netCDF4.Dataset(nc_file_name)

            nam_predictor_matrix = None
            
            for this_predictor_name in predictor_names_updated:

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
            

            if self.map_structure == '1D':

                time_steps = lead_time_steps[idx]
                this_timeseries_predictor_matrix = self.return_tabular_data(nam_predictor_matrix, time_steps) 

                if timeseries_predictor_matrix is None:
                    timeseries_predictor_matrix = this_timeseries_predictor_matrix

                else:
                    timeseries_predictor_matrix = pd.concat((
                        timeseries_predictor_matrix, this_timeseries_predictor_matrix), axis=1)

            else:

                if timeseries_predictor_matrix is None:
                    timeseries_predictor_matrix = nam_predictor_matrix + 0.

                else:
                    timeseries_predictor_matrix = np.concatenate(
                        (timeseries_predictor_matrix, nam_predictor_matrix), axis=0
                    )

                if timeseries_surfacetemp_matrix is None:
                    timeseries_surfacetemp_matrix = surfacetemp_matrix + 0.

                else:
                    timeseries_surfacetemp_matrix = np.concatenate(
                        (timeseries_surfacetemp_matrix, surfacetemp_matrix), axis=0
                    )
        
        # Preparing MUR Dataset: 

            netcdf_mur_024_names   = netcdf_mur_root_names[0:57] + '009_input.nc'
            mur_dataset_object     = netCDF4.Dataset(netcdf_mur_024_names)
            mur_predictor_matrix   = np.array(
                mur_dataset_object.variables['analysed_sst'][:], dtype=float
            )

            mur_predictor_matrix = np.expand_dims(
                mur_predictor_matrix, axis=-1)

            if self.map_structure == '4D':

                if timeseries_mur_matrix is None:
                    timeseries_mur_matrix = mur_predictor_matrix + 0.

                else:
                    timeseries_mur_matrix = np.concatenate(
                        (timeseries_mur_matrix, mur_predictor_matrix), axis=0
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





















plt.rcParams.update({'axes.titlesize': 14})
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('axes', labelsize=14)     # fontsize of the x and y labels 


DEFAULT_IMAGE_DIR_NAME    = '/data1/fog-data/fog-maps/'
DEFAULT_TARGET_DIR_NAME   = '/data1/fog/Dataset/TARGET'

NAM_coords = {'P1':[23, 6],
         'P2':[22, 8]} 

'''NAM_coords = {'P1':[23, 6],
         'P2':[22, 8],
         'P3':[21, 10],
         'P4':[19, 12]} '''

# Variable names.
NETCDF_X = 'x'
NETCDF_Y = 'y'
MUR_LATITUDE = 'lat'
MUR_LONGITUDE = 'lon'
NETCDF_LATITUDE = 'latitude'
NETCDF_LONGITUDE = 'longitude'
NETCDF_TIME = 'time'


NETCDF_UGRD_10m   = 'UGRD_10maboveground'
#NETCDF_UGRD_1000mb= 'UGRD_1000mb'
NETCDF_UGRD_975mb = 'UGRD_975mb'
NETCDF_UGRD_950mb = 'UGRD_950mb'
NETCDF_UGRD_925mb = 'UGRD_925mb'
NETCDF_UGRD_900mb = 'UGRD_900mb'
NETCDF_UGRD_875mb = 'UGRD_875mb'
NETCDF_UGRD_850mb = 'UGRD_850mb'
NETCDF_UGRD_825mb = 'UGRD_825mb'
NETCDF_UGRD_800mb = 'UGRD_800mb'
NETCDF_UGRD_775mb = 'UGRD_775mb'
NETCDF_UGRD_750mb = 'UGRD_750mb'
NETCDF_UGRD_725mb = 'UGRD_725mb'
NETCDF_UGRD_700mb = 'UGRD_700mb'
NETCDF_VGRD_10m   = 'VGRD_10maboveground'
#NETCDF_VGRD_1000mb= 'VGRD_1000mb'
NETCDF_VGRD_975mb = 'VGRD_975mb'
NETCDF_VGRD_950mb = 'VGRD_950mb'
NETCDF_VGRD_925mb = 'VGRD_925mb'
NETCDF_VGRD_900mb = 'VGRD_900mb'
NETCDF_VGRD_875mb = 'VGRD_875mb'
NETCDF_VGRD_850mb = 'VGRD_850mb'
NETCDF_VGRD_825mb = 'VGRD_825mb'
NETCDF_VGRD_800mb = 'VGRD_800mb'
NETCDF_VGRD_775mb = 'VGRD_775mb'
NETCDF_VGRD_750mb = 'VGRD_750mb'
NETCDF_VGRD_725mb = 'VGRD_725mb'
NETCDF_VGRD_700mb = 'VGRD_700mb'
#NETCDF_VVEL_1000mb= 'VVEL_1000mb'
NETCDF_VVEL_975mb = 'VVEL_975mb'
NETCDF_VVEL_950mb = 'VVEL_950mb'
NETCDF_VVEL_925mb = 'VVEL_925mb'
NETCDF_VVEL_900mb = 'VVEL_900mb'
NETCDF_VVEL_875mb = 'VVEL_875mb'
NETCDF_VVEL_850mb = 'VVEL_850mb'
NETCDF_VVEL_825mb = 'VVEL_825mb'
NETCDF_VVEL_800mb = 'VVEL_800mb'
NETCDF_VVEL_775mb = 'VVEL_775mb'
NETCDF_VVEL_750mb = 'VVEL_750mb'
NETCDF_VVEL_725mb = 'VVEL_725mb'
NETCDF_VVEL_700mb = 'VVEL_700mb'
#NETCDF_TKE_1000mb = 'TKE_1000mb'
NETCDF_TKE_975mb = 'TKE_975mb'
NETCDF_TKE_950mb = 'TKE_950mb'
NETCDF_TKE_925mb = 'TKE_925mb'
NETCDF_TKE_900mb = 'TKE_900mb'
NETCDF_TKE_875mb = 'TKE_875mb'
NETCDF_TKE_850mb = 'TKE_850mb'
NETCDF_TKE_825mb = 'TKE_825mb'
NETCDF_TKE_800mb = 'TKE_800mb'
NETCDF_TKE_775mb = 'TKE_775mb'
NETCDF_TKE_750mb = 'TKE_750mb'
NETCDF_TKE_725mb = 'TKE_725mb'
NETCDF_TKE_700mb = 'TKE_700mb'
NETCDF_TMP_SFC  = 'TMP_surface'
NETCDF_TMP_2m    = 'TMP_2maboveground'
#NETCDF_TMP_1000mb= 'TMP_1000mb'
NETCDF_TMP_975mb = 'TMP_975mb'
NETCDF_TMP_950mb = 'TMP_950mb'
NETCDF_TMP_925mb = 'TMP_925mb'
NETCDF_TMP_900mb = 'TMP_900mb'
NETCDF_TMP_875mb = 'TMP_875mb'
NETCDF_TMP_850mb = 'TMP_850mb'
NETCDF_TMP_825mb = 'TMP_825mb'
NETCDF_TMP_800mb   = 'TMP_800mb'
NETCDF_TMP_775mb   = 'TMP_775mb'
NETCDF_TMP_750mb   = 'TMP_750mb'
NETCDF_TMP_725mb   = 'TMP_725mb'
NETCDF_TMP_700mb   = 'TMP_700mb'
#NETCDF_RH_1000mb = 'RH_1000mb'
NETCDF_RH_975mb    = 'RH_975mb'
NETCDF_RH_950mb    = 'RH_950mb'
NETCDF_RH_925mb    = 'RH_925mb'
NETCDF_RH_900mb    = 'RH_900mb'
NETCDF_RH_875mb    = 'RH_875mb'
NETCDF_RH_850mb    = 'RH_850mb'
NETCDF_RH_825mb    = 'RH_825mb'
NETCDF_RH_800mb    = 'RH_800mb'
NETCDF_RH_775mb    = 'RH_775mb'
NETCDF_RH_750mb    = 'RH_750mb'
NETCDF_RH_725mb    = 'RH_725mb'
NETCDF_RH_700mb    = 'RH_700mb'
NETCDF_DPT_2m      = 'DPT_2maboveground'
NETCDF_FRICV       = 'FRICV_surface'
NETCDF_VIS         = 'VIS_surface'
NETCDF_RH_2m       = 'RH_2maboveground'
#
NETCDF_Q975        = 'Q_975mb'
NETCDF_Q950        = 'Q_950mb'
NETCDF_Q925        = 'Q_925mb'
NETCDF_Q900        = 'Q_900mb'
NETCDF_Q875        = 'Q_875mb'
NETCDF_Q850        = 'Q_850mb'
NETCDF_Q825        = 'Q_825mb'
NETCDF_Q800        = 'Q_800mb'
NETCDF_Q775        = 'Q_775mb'
NETCDF_Q750        = 'Q_750mb'
NETCDF_Q725        = 'Q_725mb'
NETCDF_Q700        = 'Q_700mb'
NETCDF_Q           = 'Q_surface'
#NETCDF_DQDZ1000SFC = 'DQDZ1000SFC'
NETCDF_DQDZ975SFC  = 'DQDZ975SFC'
NETCDF_DQDZ950975  = 'DQDZ950975'
NETCDF_DQDZ925950  = 'DQDZ925950'
NETCDF_DQDZ900925  = 'DQDZ900925'
NETCDF_DQDZ875900  = 'DQDZ875900'
NETCDF_DQDZ850875  = 'DQDZ850875'
NETCDF_DQDZ825850  = 'DQDZ825850'
NETCDF_DQDZ800825  = 'DQDZ800825'
NETCDF_DQDZ775800  = 'DQDZ775800'
NETCDF_DQDZ750775  = 'DQDZ750775'
NETCDF_DQDZ725750  = 'DQDZ725750'
NETCDF_DQDZ700725  = 'DQDZ700725'
NETCDF_LCLT        = 'LCLT'

#+++++++++++++++
NETCDF_SST    = 'analysed_sst'
NETCDF_TMPDPT = 'TMP-DPT'
NETCDF_TMPSST = 'TMP-SST'
NETCDF_DPTSST = 'DPT-SST'
PREDICTOR_NAMES_KEY  = 'predictor_names'
PREDICTOR_MATRIX_KEY = 'predictor_matrix'
CUBE_NAMES_KEY       = 'cube_name'
SST_MATRIX_KEY       = 'sst_matrix'
SST_NAME_KEY         = 'sst_name'

NETCDF_PREDICTOR_NAMES = {
    'All': [NETCDF_TMP_2m, NETCDF_TMP_975mb, NETCDF_TMP_950mb, NETCDF_TMP_925mb, NETCDF_TMP_900mb, NETCDF_TMP_875mb, NETCDF_TMP_850mb, 
    NETCDF_TMP_825mb, NETCDF_TMP_800mb, NETCDF_TMP_775mb, NETCDF_TMP_750mb, NETCDF_TMP_725mb, NETCDF_TMP_700mb, 
    NETCDF_UGRD_10m, NETCDF_VGRD_10m, NETCDF_FRICV, NETCDF_UGRD_975mb, NETCDF_VGRD_975mb, NETCDF_TKE_975mb,
    NETCDF_UGRD_950mb, NETCDF_VGRD_950mb, NETCDF_TKE_950mb, NETCDF_UGRD_925mb, NETCDF_VGRD_925mb, NETCDF_TKE_925mb, NETCDF_UGRD_900mb, NETCDF_VGRD_900mb,
    NETCDF_TKE_900mb, NETCDF_UGRD_875mb, NETCDF_VGRD_875mb, NETCDF_TKE_875mb, NETCDF_UGRD_850mb, NETCDF_VGRD_850mb, NETCDF_TKE_850mb, NETCDF_UGRD_825mb,
    NETCDF_VGRD_825mb, NETCDF_TKE_825mb, NETCDF_UGRD_800mb, NETCDF_VGRD_800mb, NETCDF_TKE_800mb, NETCDF_UGRD_775mb, NETCDF_VGRD_775mb,
    NETCDF_TKE_775mb, NETCDF_UGRD_750mb, NETCDF_VGRD_750mb, NETCDF_TKE_750mb, NETCDF_UGRD_725mb, NETCDF_VGRD_725mb, NETCDF_TKE_725mb,
    NETCDF_UGRD_700mb,  NETCDF_VGRD_700mb, NETCDF_TKE_700mb, NETCDF_Q975, NETCDF_Q950, NETCDF_Q925, NETCDF_Q900, NETCDF_Q875, NETCDF_Q850,
    NETCDF_Q825, NETCDF_Q800, NETCDF_Q775,NETCDF_Q750, NETCDF_Q725, NETCDF_Q700, NETCDF_RH_975mb, NETCDF_RH_950mb, NETCDF_RH_925mb,NETCDF_RH_900mb, 
    NETCDF_RH_875mb, NETCDF_RH_850mb, NETCDF_RH_825mb, NETCDF_RH_800mb, NETCDF_RH_775mb, NETCDF_RH_750mb, NETCDF_RH_725mb, NETCDF_RH_700mb, 
    NETCDF_DPT_2m, NETCDF_Q, NETCDF_RH_2m, NETCDF_LCLT, NETCDF_VIS, NETCDF_VVEL_975mb, NETCDF_VVEL_950mb, NETCDF_VVEL_925mb, NETCDF_VVEL_900mb, 
    NETCDF_VVEL_875mb, NETCDF_VVEL_850mb, NETCDF_VVEL_825mb, NETCDF_VVEL_800mb, NETCDF_VVEL_775mb, NETCDF_VVEL_750mb, NETCDF_VVEL_725mb, NETCDF_VVEL_700mb],


    'Five_Top': [NETCDF_VVEL_750mb, NETCDF_VVEL_925mb, NETCDF_VIS, NETCDF_VGRD_775mb, NETCDF_TMP_2m],
    'Five_Top_2': [NETCDF_VVEL_925mb, NETCDF_TMP_2m, NETCDF_VGRD_775mb, NETCDF_VIS, NETCDF_VVEL_750mb],

    'Physical_G1':[NETCDF_FRICV, NETCDF_UGRD_10m, NETCDF_UGRD_975mb, NETCDF_UGRD_950mb, NETCDF_UGRD_925mb, NETCDF_UGRD_900mb,
    NETCDF_UGRD_875mb, NETCDF_UGRD_850mb, NETCDF_UGRD_825mb, NETCDF_UGRD_800mb, NETCDF_UGRD_775mb,  NETCDF_UGRD_750mb,
    NETCDF_UGRD_725mb, NETCDF_UGRD_700mb, NETCDF_VGRD_10m, NETCDF_VGRD_975mb, NETCDF_VGRD_950mb, NETCDF_VGRD_925mb,
    NETCDF_VGRD_900mb, NETCDF_VGRD_875mb, NETCDF_VGRD_850mb, NETCDF_VGRD_825mb, NETCDF_VGRD_800mb, NETCDF_VGRD_775mb, NETCDF_VGRD_750mb,
    NETCDF_VGRD_725mb, NETCDF_VGRD_700mb],
    'Physical_G2':[NETCDF_TKE_975mb, NETCDF_TKE_950mb, NETCDF_TKE_925mb, NETCDF_TKE_900mb, NETCDF_TKE_875mb, NETCDF_TKE_850mb, NETCDF_TKE_825mb,
    NETCDF_TKE_800mb, NETCDF_TKE_775mb, NETCDF_TKE_750mb, NETCDF_TKE_725mb, NETCDF_TKE_700mb, NETCDF_Q975, NETCDF_Q950, NETCDF_Q925, NETCDF_Q900,
    NETCDF_Q875, NETCDF_Q850, NETCDF_Q825, NETCDF_Q800, NETCDF_Q775,NETCDF_Q750, NETCDF_Q725, NETCDF_Q700],
    'Physical_G3':[NETCDF_TMP_2m, NETCDF_TMP_975mb, NETCDF_TMP_950mb, NETCDF_TMP_925mb, NETCDF_TMP_900mb, NETCDF_TMP_875mb, NETCDF_TMP_850mb,
    NETCDF_TMP_825mb, NETCDF_TMP_800mb, NETCDF_TMP_775mb, NETCDF_TMP_750mb, NETCDF_TMP_725mb, NETCDF_TMP_700mb, NETCDF_DPT_2m, NETCDF_RH_2m,
    NETCDF_RH_975mb, NETCDF_RH_950mb, NETCDF_RH_925mb,NETCDF_RH_900mb, NETCDF_RH_875mb, NETCDF_RH_850mb, NETCDF_RH_825mb, NETCDF_RH_800mb,
    NETCDF_RH_775mb, NETCDF_RH_750mb, NETCDF_RH_725mb, NETCDF_RH_700mb],
    'Physical_G4':[NETCDF_Q, NETCDF_LCLT, NETCDF_VIS, NETCDF_VVEL_975mb, NETCDF_VVEL_950mb, NETCDF_VVEL_925mb, NETCDF_VVEL_900mb, NETCDF_VVEL_875mb, NETCDF_VVEL_850mb, NETCDF_VVEL_825mb,
    NETCDF_VVEL_800mb, NETCDF_VVEL_775mb, NETCDF_VVEL_750mb, NETCDF_VVEL_725mb, NETCDF_VVEL_700mb], 
    'Mixed':[NETCDF_SST, NETCDF_TMPDPT, NETCDF_TMPSST, NETCDF_DPTSST], 
    'MUR': [NETCDF_SST], 
    'Generated': [NETCDF_TMPDPT, NETCDF_TMPSST, NETCDF_DPTSST], 
    'TMP': [NETCDF_TMP_SFC, NETCDF_TMP_2m, NETCDF_DPT_2m]
    }
NETCDF_MUR_NAMES   = [NETCDF_SST]
NETCDF_TMP_NAMES   = [NETCDF_TMP_SFC, NETCDF_TMP_2m, NETCDF_DPT_2m]
NETCDF_MIXED_NAMES = [NETCDF_SST, NETCDF_TMPDPT, NETCDF_TMPSST, NETCDF_DPTSST]
NETCDF_GEN_NAMES   = [NETCDF_TMPDPT, NETCDF_TMPSST, NETCDF_DPTSST]