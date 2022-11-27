import numpy as np
import os
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json

import warnings
warnings.simplefilter(action='ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Check if there is GPU(s): {torch.cuda.is_available()}")


from src import dataloader
from models import transformers, engine, configs






def train_vit(image_data_path: str, 
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
    isExist  = os.path.isdir(save_dir)

    if not isExist:
        os.mkdir(save_dir)

    best_model_name      = save_dir + '/best_model_' + Exp_name + '.pth'
    loss_fig_name        = save_dir + '/loss_' + Exp_name + '.png'
    loss_df_name         = save_dir + '/loss_' + Exp_name + '.csv' 
    dict_name            = save_dir + '/mean_std_' + Exp_name + '.json' 

    # creating the entire data: 
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
                                            map_structure         = '2D', 
                                            predictor_names       = predictor_names, 
                                            lead_time_pred        = lead_time_pred, 
                                            mean_std_dict         = norm_mean_std_dict,
                                            point_geolocation_dic = point_geolocation_dic)

    valid_dataset = dataloader.DataAdopter(valid_df, 
                                            map_structure         = '2D', 
                                            predictor_names       = predictor_names, 
                                            lead_time_pred        = lead_time_pred, 
                                            mean_std_dict         = norm_mean_std_dict,
                                            point_geolocation_dic = point_geolocation_dic)

    test_dataset = dataloader.DataAdopter(test_df, 
                                            map_structure         = '2D', 
                                            predictor_names       = predictor_names, 
                                            lead_time_pred        = lead_time_pred, 
                                            mean_std_dict         = norm_mean_std_dict,
                                            point_geolocation_dic = point_geolocation_dic)

    data_loader_training = torch.utils.data.DataLoader(train_dataset, batch_size= model_config_dict['batch_size'], 
                                                    shuffle=True,  num_workers=8) 
    data_loader_validate = torch.utils.data.DataLoader(valid_dataset, batch_size= model_config_dict['batch_size'], 
                                                    shuffle=False,  num_workers=8)
    data_loader_testing = torch.utils.data.DataLoader(test_dataset, batch_size= model_config_dict['batch_size'], 
                                                    shuffle=False,  num_workers=8)
    model_type = 'ViT-L_32'
    config = transformers.CONFIGS[model_type]
    model = transformers.VisionTransformer(config, img_size=32, num_classes=2,)
    #parallel_net = nn.DataParallel(model, device_ids = [0, 1, 2, 3])
    parallel_net = model.to(0)

    # train and test
    optimizer = optim.Adam(parallel_net.parameters(), lr = model_config_dict['lr'], weight_decay = model_config_dict['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss() # BCEWithLogitsLoss()  #  torch.nn.NLLLoss()

    loss_stats = {'train': [],"val": []}

    best_val_loss = 100000 # initial dummy value
    early_stopping = engine.EarlyStopping(tolerance = model_config_dict['early_stop_tolerance'], min_delta=50)
    step  = 0
    #==============================================================================================================#
    #==============================================================================================================#
    epochs = model_config_dict['epochs']
    for epoch in range(1, epochs+1):
        training_start_time = time.time()
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc  = 0
        parallel_net.train()

        for batch_idx, sample in enumerate(data_loader_training):
            
            input_train      = sample['input'].to(0)
            train_label_true = sample['label_class'].to(0)

            #logits = parallel_net(input_train)[0]
            #print(logits)
            optimizer.zero_grad()

            train_loss  = parallel_net(input_train, train_label_true) #
            train_loss.backward()
            optimizer.step()
            step += 1

            train_epoch_loss += train_loss.item()

            #train_acc = engine.binary_acc(train_out, train_class_true.unsqueeze(1))
            #train_epoch_acc += train_acc.item()

        #scheduler.step(_)
        # VALIDATION    
        with torch.no_grad():
            
            val_epoch_loss = 0
            #val_epoch_acc = 0
            parallel_net.eval()
            for batch, sample in enumerate(data_loader_validate):
                
                input_val      = sample['input'].to(0)
                label_true_val = sample['label_class'].to(0)

                #logits = parallel_net(input_val)
            
                val_loss       = parallel_net(input_val, label_true_val) #loss_func(logits, label_true_val)
                val_epoch_loss += val_loss.item()

                
        training_duration_time = (time.time() - training_start_time)

        loss_stats['train'].append(train_epoch_loss/len(data_loader_training))
        loss_stats['val'].append(val_epoch_loss/len(data_loader_validate))
        print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss/len(data_loader_training):.4f} | Val Loss: {val_epoch_loss/len(data_loader_validate):.4f} | Time(s): {training_duration_time:.3f}') 
        #   | Train HSS: {train_epoch_acc/len(data_loader_training):.4f} | Val HSS: {val_epoch_acc/len(data_loader_validate):.4f}
        if (val_epoch_loss/len(data_loader_validate)) < best_val_loss or epoch==0:
                    
            best_val_loss=(val_epoch_loss/len(data_loader_validate))
            torch.save(parallel_net.state_dict(), best_model_name)
            
            status = True

            print(f'Best model Saved! Val Loss: {best_val_loss:.4f}')
        else:
            print(f'Model is not saved! Current Val Loss: {(val_epoch_loss/len(data_loader_validate)):.4f}') 
                
            status = False

        # early stopping
        early_stopping(status)
        if early_stopping.early_stop:
            print("Epoch:", epoch)
            break
    _ = engine.save_loss_df(loss_stats, loss_df_name, loss_fig_name)


    train_output, valid_output, test_output = engine.predict(parallel_net, 
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
                                '2020':['20200101', '20201215']}


    train_vit(image_data_path     = DEFAULT_IMAGE_DIR_NAME, 
            target_data_path      = DEFAULT_TARGET_DIR_NAME,
            start_date            = year_information['2009'][0], 
            finish_date           = year_information['2020'][1], 
            lead_time_pred        = 24, 
            predictor_names       = dataloader.NETCDF_PREDICTOR_NAMES['Five_Top'], 
            data_split_dict       = {'train': ['2013', '2014', '2015', '2016', '2017'], 'valid': ['2009', '2010', '2011'], 'test': ['2018', '2019', '2020']}, #{'train': ['2020'], 'valid': ['2020'], 'test': ['2020']}, 
            visibility_threshold  = 1, 
            point_geolocation_dic = dataloader.NAM_coords, 
            model_config_dict     = configs.config_dictionary_1D,
            Exp_name              = 'EX002_2D_FiveTop'
            )

