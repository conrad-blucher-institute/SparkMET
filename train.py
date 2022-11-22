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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())


from src import dataloader
from models import transformers, engine


DEFAULT_IMAGE_DIR_NAME    = ('/data1/fog-data/fog-maps/')
DEFAULT_TARGET_DIR_NAME   = ('/data1/fog/Dataset/TARGET')
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



def train_1d(start_date = None, 
                finish_date = None, 
                lead_time_pred = None, 
                predictor_names = None, 
                data_split_dict = None,  
                visibility_threshold = None, 
                point_geolocation_dic = None, 
                lr                    = 1e-3,
                weight_decay          = 1e-4, 
                dropout               = 0.3,
                nhead                 = 8,
                dim_feedforward       = 512, 
                batch_size            = 2,
                early_stop_tolerance  = 50,
                epochs                = 50,
                Exp_name              = None,
):


    save_dir = './EXPs/' + Exp_name
    isExist  = os.path.isdir(save_dir)

    if not isExist:
        os.mkdir(save_dir)

    best_model_name      = save_dir + '/best_model' + Exp_name + '.pth'
    loss_fig_name        = save_dir + '/loss' + Exp_name + '.png'
    loss_df_name         = save_dir + '/loss' + Exp_name + '.csv' 
    dict_name            = save_dir + '/mean_std_' + Exp_name + '.json' 




    # creating the entire data: 

    dataset = dataloader.input_dataframe_generater(img_path = None, 
                                                target_path = None, 
                                                first_date_string = start_date, 
                                                last_date_string = finish_date).dataframe_generation()
                                             


    # split the data into train, validation and test:
    train_df, valid_df, test_df = dataloader.split_data_train_valid_test(dataset, year_split_dict = data_split_dict)
    print(f"=============   train input size: {train_df.shape} | validation input size: {valid_df.shape} | test input size{test_df.shape} ==============")
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
                                            vis_threshold         = visibility_threshold, 
                                            map_structure         = '1D', 
                                            predictor_names       = predictor_names, 
                                            lead_time_pred        = lead_time_pred, 
                                            mean_std_dict         = norm_mean_std_dict,
                                            point_geolocation_dic = point_geolocation_dic)

    valid_dataset = dataloader.DataAdopter(valid_df, 
                                            vis_threshold         = visibility_threshold, 
                                            map_structure         = '1D', 
                                            predictor_names       = predictor_names, 
                                            lead_time_pred        = lead_time_pred, 
                                            mean_std_dict         = norm_mean_std_dict,
                                            point_geolocation_dic = point_geolocation_dic)


    data_loader_training = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, 
                                                    shuffle=True,  num_workers=8) 
    data_loader_validate = torch.utils.data.DataLoader(valid_dataset, batch_size= batch_size, 
                                                    shuffle=False,  num_workers=8)

    


    model = transformers.Transformer1d(
                                d_model        = 744, 
                                nhead          = nhead, 
                                dim_feedforward= dim_feedforward, 
                                n_classes      = 2, 
                                dropout        = dropout, 
                                activation     = 'relu',
                                verbose        = True)
    model.to(device)

    # train and test
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.NLLLoss() # BCEWithLogitsLoss()  # torch.nn.CrossEntropyLoss()


    loss_stats = {'train': [],"val": []}

    best_val_loss = 100000 # initial dummy value
    early_stopping = engine.EarlyStopping(tolerance = early_stop_tolerance, min_delta=50)
    step  = 0
    #==============================================================================================================#
    #==============================================================================================================#
    for epoch in range(1, epochs+1):
        
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc  = 0
        model.train()

        for batch_idx, sample in enumerate(data_loader_training):
            
            input_train      = sample['input'].to(device)
            train_class_true = sample['onehotlabel'].to(device)
            train_label_true = sample['label_class'].to(device)

            _, train_out, _ = model(input_train)
            optimizer.zero_grad()

            train_loss  = loss_func(train_out, train_label_true) #
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
            val_epoch_acc = 0
            model.eval()
            for batch, sample in enumerate(data_loader_validate):
                
                input_val      = sample['input'].to(device)
                class_true_val = sample['onehotlabel'].to(device)
                label_true_val = sample['label_class'].to(device)

                _, pred_val, _ = model(input_val)
            
                val_loss       = loss_func(pred_val, label_true_val)
                val_epoch_loss += val_loss.item()

                #valid_acc = engine.binary_acc(train_out, train_class_true.unsqueeze(1))
                #val_epoch_acc += valid_acc.item()
                

        loss_stats['train'].append(train_epoch_loss/len(data_loader_training))
        loss_stats['val'].append(val_epoch_loss/len(data_loader_validate))
        print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss/len(data_loader_training):.4f} | Val Loss: {val_epoch_loss/len(data_loader_validate):.4f}') 
        #   | Train HSS: {train_epoch_acc/len(data_loader_training):.4f} | Val HSS: {val_epoch_acc/len(data_loader_validate):.4f}
        if (val_epoch_loss/len(data_loader_validate)) < best_val_loss or epoch==0:
                    
            best_val_loss=(val_epoch_loss/len(data_loader_validate))
            torch.save(model.state_dict(), best_model_name)
            
            status = True

            print(f'Best model Saved! Val Loss: {best_val_loss:.4f}')
        else:
            print(f'Model is not saved! Current val Loss: {(val_epoch_loss/len(data_loader_validate)):.4f}') 
                
            status = False

        # early stopping
        early_stopping(status)
        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            break
    _ = engine.save_loss_df(loss_stats, loss_df_name, loss_fig_name)


if __name__ == "__main__":
    train_1d(start_date           = year_information['2009'][0], 
            finish_date           = year_information['2020'][1], 
            lead_time_pred        = 24, 
            predictor_names       = dataloader.NETCDF_PREDICTOR_NAMES['All'], 
            data_split_dict       = {'train': ['2019'], 'valid': ['2018'], 'test':['2020']},  
            visibility_threshold  = 1, 
            point_geolocation_dic = dataloader.NAM_coords, 
            lr                    = 1e-3,
            weight_decay          = 1e-4, 
            dropout               = 0.3,
            nhead                 = 8,
            dim_feedforward       = 512, 
            batch_size            = 64,
            early_stop_tolerance  = 50,
            epochs                = 50,
            Exp_name              = 'EX002_1D'
            )

