import numpy as np
import os
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json
#from sklearn.metrics import classification_report, confusion_matrix
import warnings
from TransMAP.models import configs
warnings.simplefilter(action='ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"



from src import dataloader
from models import transformers, engine



def Eval1D(image_data_path: str, 
                target_data_path: str,
                start_date : str, 
                finish_date : str, 
                lead_time_pred : int, 
                predictor_names : dict, 
                data_split_dict : dict,  
                visibility_threshold: int, 
                point_geolocation_dic: dict, 
                model_config_dict : dict, 
                Exp_name: str,
):
    save_dir = '/data1/fog/TransMAP/EXPs/' + Exp_name
    dict_name = save_dir + '/mean_std_' + Exp_name + '.json' 


    dataset = dataloader.input_dataframe_generater(img_path = None, 
                                                target_path = None, 
                                                first_date_string = start_date, 
                                                last_date_string = finish_date, 
                                                target_binarizing_thre = visibility_threshold).dataframe_generation()

    # split the data into train, validation and test:
    train_df, valid_df, test_df = dataloader.split_data_train_valid_test(dataset, year_split_dict = data_split_dict)
    _ = engine.print_report(train_df, valid_df, test_df)
    # calculating the mean and std of training variables: 
    #start_time = time.time()
    isDictExists = os.path.isfile(dict_name)
    if not isDictExists:
        norm_mean_std_dict = dataloader.return_train_variable_mean_std(train_df, 
                                                    predictor_names = predictor_names, 
                                                    lead_time_pred = lead_time_pred).return_mean_std_dict()
        #print("--- Normalize data: %s seconds ---" % (time.time() - start_time))
        with open(dict_name, "w") as outfile:
            json.dump(norm_mean_std_dict, outfile)
    else: 
        with open(dict_name, 'r') as file:
            norm_mean_std_dict = json.load(file)

    train_dataset = dataloader.DataAdopter(train_df, 
                                            map_structure         = '1D', 
                                            predictor_names       = predictor_names, 
                                            lead_time_pred        = lead_time_pred, 
                                            mean_std_dict         = norm_mean_std_dict,
                                            point_geolocation_dic = point_geolocation_dic)

    valid_dataset = dataloader.DataAdopter(valid_df, 
                                            map_structure         = '1D', 
                                            predictor_names       = predictor_names, 
                                            lead_time_pred        = lead_time_pred, 
                                            mean_std_dict         = norm_mean_std_dict,
                                            point_geolocation_dic = point_geolocation_dic)

    test_dataset = dataloader.DataAdopter(test_df, 
                                            map_structure         = '1D', 
                                            predictor_names       = predictor_names, 
                                            lead_time_pred        = lead_time_pred, 
                                            mean_std_dict         = norm_mean_std_dict,
                                            point_geolocation_dic = point_geolocation_dic)

    data_loader_training = torch.utils.data.DataLoader(train_dataset, batch_size= model_config_dict['batch_size'], 
                                                    shuffle=False,  num_workers=8) 
    data_loader_validate = torch.utils.data.DataLoader(valid_dataset, batch_size= model_config_dict['batch_size'], 
                                                    shuffle=False,  num_workers=8)
    data_loader_testing = torch.utils.data.DataLoader(test_dataset, batch_size= model_config_dict['batch_size'], 
                                                    shuffle=False,  num_workers=8)



    model = transformers.Transformer1d(
                                d_model        = 744, 
                                nhead          = model_config_dict['nhead'], 
                                dim_feedforward= model_config_dict['dim_feedforward'], 
                                n_classes      = model_config_dict['num_classes'], 
                                dropout        = model_config_dict['dropout'], 
                                activation     = 'relu',
                                verbose        = True)
    parallel_net = nn.DataParallel(model, device_ids = [0,1,2, 3])
    parallel_net = parallel_net.to(0)



    train_output, valid_output, test_output = engine.eval_1d(parallel_net, 
                                                            data_loader_training, 
                                                            data_loader_validate, 
                                                            data_loader_testing, 
                                                            Exp_name = Exp_name,)


if __name__ == "__main__":

    DEFAULT_IMAGE_DIR_NAME    = '/data1/fog-data/fog-maps/'
    DEFAULT_TARGET_DIR_NAME   = '/data1/fog/Dataset/TARGET'
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


    Eval1D(image_data_path        = DEFAULT_IMAGE_DIR_NAME, 
            target_data_path      = DEFAULT_TARGET_DIR_NAME,
            start_date            = year_information['2009'][0], 
            finish_date           = year_information['2020'][1], 
            lead_time_pred        = 24, 
            predictor_names       = dataloader.NETCDF_PREDICTOR_NAMES['All'], 
            data_split_dict       = {'train': ['2013', '2014', '2015', '2016', '2017'], 'valid': ['2009', '2010', '2011'], 'test': ['2018', '2019', '2020']},  
            visibility_threshold  = 1, 
            point_geolocation_dic = dataloader.NAM_coords, 
            model_config_dict     = configs.config_dictionary_1D,
            Exp_name              = 'EX001_1D_All')
