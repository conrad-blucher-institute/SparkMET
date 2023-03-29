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
warnings.simplefilter(action='ignore')


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Check if there is GPU(s): {torch.cuda.is_available()}")


from src import dataloader
from models import transformers, engine, configs





def train_1d(image_data_path: str, 
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



    save_dir = '/data1/fog/SparkMET/EXPs/' + Exp_name
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
                                                    shuffle=True,  num_workers=8) 
    data_loader_validate = torch.utils.data.DataLoader(valid_dataset, batch_size= model_config_dict['batch_size'], 
                                                    shuffle=False,  num_workers=8)
    data_loader_testing = torch.utils.data.DataLoader(test_dataset, batch_size= model_config_dict['batch_size'], 
                                                    shuffle=False,  num_workers=8)
    


    model = transformers.Transformer1d(
                                d_model        = model_config_dict['dim_model'], 
                                nhead          = model_config_dict['nhead'], 
                                dim_feedforward= model_config_dict['dim_feedforward'], 
                                n_classes      = model_config_dict['num_classes'], 
                                dropout        = model_config_dict['dropout'], 
                                activation     = 'relu',
                                verbose        = True)

    parallel_net = nn.DataParallel(model, device_ids = [0,1,2, 3])
    parallel_net = parallel_net.to(0)

    # train and test
    optimizer = optim.Adam(parallel_net.parameters(), lr = model_config_dict['lr'], weight_decay = model_config_dict['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.NLLLoss() # BCEWithLogitsLoss()  # torch.nn.CrossEntropyLoss()


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
            #train_class_true = sample['onehotlabel'].to(0)
            train_label_true = sample['label_class'].to(0)

            _, train_out, _ = parallel_net(input_train)
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
            #val_epoch_acc = 0
            parallel_net.eval()
            for batch, sample in enumerate(data_loader_validate):
                
                input_val      = sample['input'].to(0)
                class_true_val = sample['onehotlabel'].to(0)
                label_true_val = sample['label_class'].to(0)

                _, pred_val, _ = parallel_net(input_val)
            
                val_loss       = loss_func(pred_val, label_true_val)
                val_epoch_loss += val_loss.item()

                #valid_acc = engine.binary_acc(train_out, train_class_true.unsqueeze(1))
                #val_epoch_acc += valid_acc.item()
                
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


    train_output, valid_output, test_output = engine.eval_1d(parallel_net, 
                                                            data_loader_training, 
                                                            data_loader_validate, 
                                                            data_loader_testing, 
                                                            Exp_name = Exp_name,)

if __name__ == "__main__":

    train_1d(image_data_path      = configs.DEFAULT_IMAGE_DIR_NAME, 
            target_data_path      = configs.DEFAULT_TARGET_DIR_NAME,
            start_date            = configs.year_information['2009'][0], 
            finish_date           = configs.year_information['2020'][1], 
            lead_time_pred        = 24, 
            predictor_names       = configs.NETCDF_PREDICTOR_NAMES['All'], 
            data_split_dict       = {'train': ['2013', '2014', '2015', '2016', '2017'], 'valid': ['2009', '2010', '2011'], 'test': ['2018', '2019', '2020']},  
            visibility_threshold  = 1, 
            point_geolocation_dic = configs.NAM_coords, 
            model_config_dict     = configs.config_dictionary_1D,
            Exp_name              = 'EX001_1D'
            )

