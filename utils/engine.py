import numpy as np
import pandas as pd
import torch 
import random
import glob

seed = 1987
torch.manual_seed(seed) # important 
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from einops import rearrange, repeat, reduce
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
#from torch.cuda.amp import autocast, GradScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fog_dataloader import FogDataloader

# device ="cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda:0")


from fog_dataloader import FogDataloader as fog
dict_ = fog.NETCDF_PREDICTOR_NAMES
from models import multiview_vit as vit




def return_variable_index(category: str):

    if category == 'v1':

        indices_G1 = [dict_['All'].index(item) for item in dict_['Physical_G1'] if item in dict_['All']]
        # indices_tensor_G1 = torch.tensor(indices_G1).to(device)
        timeseries_indices_G1 = indices_G1 + [item + 97 for item in indices_G1] + [item + 194 for item in indices_G1] + [item + 291 for item in indices_G1]
        timeseries_indices_tensor_G1 = torch.tensor(timeseries_indices_G1).to(device)

        indices_G2 = [dict_['All'].index(item) for item in dict_['Physical_G2'] if item in dict_['All']]
        # indices_tensor_G2 = torch.tensor(indices_G2).to(device)
        timeseries_indices_G2 = indices_G2 + [item + 97 for item in indices_G2] + [item + 194 for item in indices_G2] + [item + 291 for item in indices_G2]
        timeseries_indices_tensor_G2 = torch.tensor(timeseries_indices_G2).to(device)


        indices_G3 = [dict_['All'].index(item) for item in dict_['Physical_G3'] if item in dict_['All']]
        # indices_tensor_G3 = torch.tensor(indices_G3).to(device)
        timeseries_indices_G3 = indices_G3 + [item + 97 for item in indices_G3] + [item + 194 for item in indices_G3] + [item + 291 for item in indices_G3]
        timeseries_indices_tensor_G3 = torch.tensor(timeseries_indices_G3).to(device)


        indices_G4 = [dict_['All'].index(item) for item in dict_['Physical_G4'] if item in dict_['All']]
        # indices_tensor_G4 = torch.tensor(indices_G4).to(device)
        timeseries_indices_G4 = indices_G4 + [item + 97 for item in indices_G4] + [item + 194 for item in indices_G4] + [item + 291 for item in indices_G4]
        timeseries_indices_tensor_G4 = torch.tensor(timeseries_indices_G4).to(device)

        indices_G5 = [93, 94, 95, 96]
        # indices_tensor_G5 = torch.tensor(indices_G5).to(device)
        timeseries_indices_G5 = indices_G5 + [item + 97 for item in indices_G5] + [item + 194 for item in indices_G5] + [item + 291 for item in indices_G5]
        timeseries_indices_tensor_G5 = torch.tensor(timeseries_indices_G5).to(device)

        return [timeseries_indices_G1, 
                timeseries_indices_G2, 
                timeseries_indices_G3, 
                timeseries_indices_G4, 
                timeseries_indices_G5], [timeseries_indices_tensor_G1, 
                                     timeseries_indices_tensor_G2, 
                                     timeseries_indices_tensor_G3, 
                                     timeseries_indices_tensor_G4, 
                                     timeseries_indices_tensor_G5
                                     ]
    elif category == 'v2':

        indices_G1_V2 = [dict_['All'].index(item) for item in dict_['Physical_G1_V2'] if item in dict_['All']]
        # indices_tensor_G1_V2 = torch.tensor(indices_G1_V2).to(device)
        timeseries_indices_G1_V2 = indices_G1_V2 + [item + 97 for item in indices_G1_V2] + [item + 194 for item in indices_G1_V2] + [item + 291 for item in indices_G1_V2]
        timeseries_indices_tensor_G1_V2 = torch.tensor(timeseries_indices_G1_V2).to(device)

        indices_G2_V2 = [dict_['All'].index(item) for item in dict_['Physical_G2_V2'] if item in dict_['All']]
        # indices_tensor_G2_V2 = torch.tensor(indices_G2_V2).to(device)
        timeseries_indices_G2_V2 = indices_G2_V2 + [item + 97 for item in indices_G2_V2] + [item + 194 for item in indices_G2_V2] + [item + 291 for item in indices_G2_V2]
        timeseries_indices_tensor_G2_V2 = torch.tensor(timeseries_indices_G2_V2).to(device)

        indices_G3_V2 = [dict_['All'].index(item) for item in dict_['Physical_G3_V2'] if item in dict_['All']]
        # indices_tensor_G3_V2 = torch.tensor(indices_G3_V2).to(device)
        timeseries_indices_G3_V2 = indices_G3_V2 + [item + 97 for item in indices_G3_V2] + [item + 194 for item in indices_G3_V2] + [item + 291 for item in indices_G3_V2]
        timeseries_indices_tensor_G3_V2 = torch.tensor(timeseries_indices_G3_V2).to(device)

        indices_G4_V2 = [dict_['All'].index(item) for item in dict_['Physical_G4_V2'] if item in dict_['All']]
        indices_G4_V2.append(94)
        # indices_tensor_G4_V2 = torch.tensor(indices_G4_V2).to(device)
        timeseries_indices_G4_V2 = indices_G4_V2 + [item + 97 for item in indices_G4_V2] + [item + 194 for item in indices_G4_V2] + [item + 291 for item in indices_G4_V2]
        timeseries_indices_tensor_G4_V2 = torch.tensor(timeseries_indices_G4_V2).to(device)

        indices_G5_V2 = [93, 95, 96]
        # indices_tensor_G5_V2 = torch.tensor(indices_G5).to(device)
        timeseries_indices_G5_V2 = indices_G5_V2 + [item + 97 for item in indices_G5_V2] + [item + 194 for item in indices_G5_V2] + [item + 291 for item in indices_G5_V2]
        timeseries_indices_tensor_G5_V2 = torch.tensor(timeseries_indices_G5_V2).to(device)

        return [timeseries_indices_G1_V2, 
                timeseries_indices_G2_V2, 
                timeseries_indices_G3_V2, 
                timeseries_indices_G4_V2, 
                timeseries_indices_G5_V2], [timeseries_indices_tensor_G1_V2, 
                                        timeseries_indices_tensor_G2_V2, 
                                        timeseries_indices_tensor_G3_V2, 
                                        timeseries_indices_tensor_G4_V2, 
                                        timeseries_indices_tensor_G5_V2
                                        ]

    elif category == 'var':
        indices_temp = [dict_['All'].index(item) for item in dict_['temp'] if item in dict_['All']]
        # indices_tensor_temp = torch.tensor(indices_temp).to(device)
        timeseries_indices_temp = indices_temp + [item + 97 for item in indices_temp] + [item + 194 for item in indices_temp] + [item + 291 for item in indices_temp]
        timeseries_indices_tensor_temp = torch.tensor(timeseries_indices_temp).to(device)

        indices_u10 = [dict_['All'].index(item) for item in dict_['U10'] if item in dict_['All']]
        # indices_tensor_u10 = torch.tensor(indices_u10).to(device)
        timeseries_indices_u10 = indices_u10 + [item + 97 for item in indices_u10] + [item + 194 for item in indices_u10] + [item + 291 for item in indices_u10]
        timeseries_indices_tensor_u10 = torch.tensor(timeseries_indices_u10).to(device)

        indices_v10 = [dict_['All'].index(item) for item in dict_['V10'] if item in dict_['All']]
        # indices_tensor_v10 = torch.tensor(indices_v10).to(device)
        timeseries_indices_v10 = indices_v10 + [item + 97 for item in indices_v10] + [item + 194 for item in indices_v10] + [item + 291 for item in indices_v10]
        timeseries_indices_tensor_v10 = torch.tensor(timeseries_indices_v10).to(device)

        indices_TKE = [dict_['All'].index(item) for item in dict_['TKE'] if item in dict_['All']]
        # indices_tensor_TKE = torch.tensor(indices_TKE).to(device)
        timeseries_indices_TKE = indices_TKE + [item + 97 for item in indices_TKE] + [item + 194 for item in indices_TKE] + [item + 291 for item in indices_TKE]
        timeseries_indices_tensor_TKE = torch.tensor(timeseries_indices_TKE).to(device)

        indices_Q = [dict_['All'].index(item) for item in dict_['Q'] if item in dict_['All']]
        # indices_tensor_Q = torch.tensor(indices_Q).to(device)
        timeseries_indices_Q = indices_Q + [item + 97 for item in indices_Q] + [item + 194 for item in indices_Q] + [item + 291 for item in indices_Q]
        timeseries_indices_tensor_Q = torch.tensor(timeseries_indices_Q).to(device)

        indices_RH = [dict_['All'].index(item) for item in dict_['RH'] if item in dict_['All']]
        # indices_tensor_RH = torch.tensor(indices_RH).to(device)
        timeseries_indices_RH = indices_RH + [item + 97 for item in indices_RH] + [item + 194 for item in indices_RH] + [item + 291 for item in indices_RH]
        timeseries_indices_tensor_RH = torch.tensor(timeseries_indices_RH).to(device)

        indices_VVEL = [dict_['All'].index(item) for item in dict_['VVEL'] if item in dict_['All']]
        # indices_tensor_VVEL = torch.tensor(indices_VVEL).to(device)
        timeseries_indices_VVEL = indices_VVEL + [item + 97 for item in indices_VVEL] + [item + 194 for item in indices_VVEL] + [item + 291 for item in indices_VVEL]
        timeseries_indices_tensor_VVEL = torch.tensor(timeseries_indices_VVEL).to(device)

        indices_VIS = [dict_['All'].index(item) for item in dict_['VIS'] if item in dict_['All']]
        # indices_tensor_VIS = torch.tensor(indices_VIS).to(device)
        timeseries_indices_VIS = indices_VIS + [item + 97 for item in indices_VIS] + [item + 194 for item in indices_VIS] + [item + 291 for item in indices_VIS]
        timeseries_indices_tensor_VIS = torch.tensor(timeseries_indices_VIS).to(device)

        indices_SST = [93, 94, 95, 96]
        # indices_tensor_SST = torch.tensor(indices_SST).to(device)
        timeseries_indices_SST = indices_SST + [item + 97 for item in indices_SST] + [item + 194 for item in indices_SST] + [item + 291 for item in indices_SST]
        timeseries_indices_tensor_SST = torch.tensor(timeseries_indices_SST).to(device)

        return [timeseries_indices_temp, 
                timeseries_indices_u10, 
                timeseries_indices_v10, 
                timeseries_indices_TKE, 
                timeseries_indices_Q,
                timeseries_indices_RH, 
                timeseries_indices_VVEL, 
                timeseries_indices_VIS, 
                timeseries_indices_SST], [timeseries_indices_tensor_temp, 
                                            timeseries_indices_tensor_u10, 
                                            timeseries_indices_tensor_v10,
                                            timeseries_indices_tensor_TKE,
                                            timeseries_indices_tensor_Q, 
                                        timeseries_indices_tensor_RH, 
                                        timeseries_indices_tensor_VVEL, 
                                        timeseries_indices_tensor_VIS, 
                                        timeseries_indices_tensor_SST
                                        ]

class EarlyStopping():
    def __init__(self, tolerance=30, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, status):


        if status is True:
            self.counter = 0
        elif status is False: 
            self.counter +=1

        print(f"count: {self.counter}")
        if self.counter >= self.tolerance:  
                self.early_stop = True

def save_loss_df(loss_stat, loss_df_name, loss_fig_name):

    df = pd.DataFrame.from_dict(loss_stat).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    df.to_csv(loss_df_name) 
    plt.figure(figsize=(12,8))
    sns.lineplot(data=df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.ylim(0, df['value'].max())
    plt.savefig(loss_fig_name, dpi = 300)

def train(model, optimizer, loss_func, data_loader_training, data_loader_validate, 
          epochs=100, early_stop_tolerance= 50, SaveDir = None, Exp_name = None):

    
    save_dir = SaveDir + Exp_name
 
    best_model_name      = save_dir + '/best_model_' + Exp_name + '.pth'
    last_epoch_model     = save_dir + '/last_epoch_' + Exp_name + '.pth'
    loss_fig_name        = save_dir + '/loss_' + Exp_name + '.png'
    loss_df_name         = save_dir + '/loss_' + Exp_name + '.csv' 

    loss_stats = {'train': [],"val": []}

    best_val_loss  = 100000 
    early_stopping = EarlyStopping(tolerance = early_stop_tolerance, min_delta=50)
    step           = 0
    #==============================================================================================================#
    #==============================================================================================================#
    #scaler = GradScaler()
    inputs, targets, p, ces, ls = [], [], [], [], []
    for epoch in range(1, epochs+1):
        
        e_in, e_tar, e_p, e_ces, e_ls = [], [], [], [], []

        training_start_time = time.time()
        # TRAINING
        train_epoch_loss = 0
        model.train()
        for i, sample in enumerate(data_loader_training):

            input_train        = sample['input'].to(device) 
            train_label_true   = sample['label_class'].to(device).float() 
            # train_visibility = sample['vis'].long()
            train_visibility_indices = train_label_true.long() # (train_visibility - 0.5)

            optimizer.zero_grad()
            # train_out = model(input_train,  g = True) 
            train_out = model(input_train, train_visibility_indices.to(device))#torch.LongTensor()
            train_loss = loss_func(train_out[:, 1], train_label_true) 
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()


            # e_in.append(l[0].detach().cpu().numpy())
            # e_tar.append(l[1].detach().cpu().numpy())
            # e_p.append(l[2].detach().cpu().numpy())
            # e_ces.append(l[3].detach().cpu().numpy())
            # e_ls.append(l[4].detach().cpu().numpy())

        # model.eval()
        with torch.no_grad():
            val_epoch_loss = 0
            for batch, sample in enumerate(data_loader_validate):
                
                input_val = sample['input'].to(device) 
                label_true_val = sample['label_class'].to(device).float() 
                # val_vis = sample['vis']
                val_visibility_indices = label_true_val.long() #(val_vis - 0.5)
                # pred_val = model(input_val,  g = False)
                pred_val = model(input_val, val_visibility_indices.to(device))
            
                val_loss = loss_func(pred_val[:, 1], label_true_val)#torch.LongTensor()
                val_epoch_loss += val_loss.item()
        
        training_duration_time = (time.time() - training_start_time)

        loss_stats['train'].append(train_epoch_loss/len(data_loader_training))
        loss_stats['val'].append(val_epoch_loss/len(data_loader_validate))
        print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss/len(data_loader_training):.4f} | Val Loss: {val_epoch_loss/len(data_loader_validate):.4f} | Time(s): {training_duration_time:.3f}') 

        if (val_epoch_loss/len(data_loader_validate)) < best_val_loss or epoch==0:
            best_val_loss=(val_epoch_loss/len(data_loader_validate))
            torch.save(model.state_dict(), best_model_name)
            
            status = True
            print(f'Best model Saved! Val Loss: {best_val_loss:.4f}')
    
        else:
            print(f'Model is not saved! Current Val Loss: {(val_epoch_loss/len(data_loader_validate)):.4f}') 
                
            status = False

        if epoch==30:
            torch.save(model.state_dict(), last_epoch_model)   
        # early stopping
        early_stopping(status)
        if early_stopping.early_stop:
            print("Epoch:", epoch)
            break
    
    #     inputs.append(np.concatenate(e_in))
    #     targets.append(np.concatenate(e_tar))
    #     p.append(np.concatenate(e_p))
    #     ces.append(np.concatenate(e_ces))
    #     ls.append(np.concatenate(e_ls))

    # inputs = np.concatenate(inputs)
    # targets = np.concatenate(targets)
    # p = np.concatenate(p)
    # ces = np.concatenate(ces)
    # ls = np.concatenate(ls)

    # prob_loss = pd.DataFrame()

    # prob_loss['logits'] = inputs
    # prob_loss['targets'] = targets
    # prob_loss['p'] = p
    # prob_loss['ce'] = ces
    # prob_loss['loss'] = ls

    # prob_loss.to_csv(save_dir + '/prob_loss_' + Exp_name + '.csv' )

    _ = save_loss_df(loss_stats, loss_df_name, loss_fig_name)

    return model, loss_stats

def predict(model, data_loader_training, data_loader_testing, config, by: str, SaveDir = None, Exp_name = None,):

    save_dir =  SaveDir + Exp_name
    best_model_name      = save_dir + '/best_model_' + Exp_name + '.pth'
    # last_epoch_model     = save_dir + '/last_epoch_' + Exp_name + '.pth'

    train_csv_file = save_dir + '/train_prob_' + Exp_name + '.csv'
    test_csv_file  = save_dir + '/test_prob_' + Exp_name + '.csv'
    figure_name_root = save_dir + '/' +  Exp_name  

    full_report  = save_dir + '/full_report_' + Exp_name + '.txt'


    model = vit.multi_view_models(config, cond = False).to(device)

    model.load_state_dict(torch.load(best_model_name))

    #==============================================================================================================#
    # model.eval()
    with torch.no_grad():
        
        train_date_times, train_round_times, train_cycletimes, train_visibilitys, train_label_trues, train_fog_preds, train_nonfog_preds= [], [], [], [], [], [], []
        for batch_idx, sample in enumerate(data_loader_training):

            train_date_time  = sample['date_time']
            train_date_times.append(train_date_time)

            train_round_time = sample['round_time']
            train_round_times.append(train_round_time)

            train_cycletime  = sample['date_cycletime']
            train_cycletimes.append(train_cycletime)

            train_vis = sample['vis'] 
            train_visibilitys.append(train_vis)

            train_label_true = sample['label_class']
            train_label_trues.append(train_label_true)

            input_train      = sample['input'].to(device)

            # train_attention_scores, train_out = model(input_train)
            # train_out = model(input_train,  g = False)

            train_visibility_indices = (train_vis - 0.5).long() 
            train_out = model(input_train,  train_visibility_indices.to(device))#torch.LongTensor()
            
            #============================================================================================
            # Output
            softmax            = torch.nn.Softmax(dim=1)
            train_out          = softmax(train_out)
            # train_out          = torch.sigmoid(train_out)
            train_out          = train_out.detach().cpu().numpy()

            train_fog_preds.append(train_out[:, 1]) #.append(train_out) 
            train_nonfog_preds.append(train_out[:, 0]) #.append(1 - train_out) 

        train_date_times   = np.concatenate(train_date_times)
        train_round_times  = np.concatenate(train_round_times)
        train_cycletimes   = np.concatenate(train_cycletimes)
        train_visibilitys  = np.concatenate(train_visibilitys)
        train_label_trues  = np.concatenate(train_label_trues)
        train_fog_preds    = np.concatenate(train_fog_preds)
        train_nonfog_preds = np.concatenate(train_nonfog_preds)

        # train_ypred = np.empty([train_fog_preds.shape[0], 2], dtype = float)
        # train_ypred[:,1] = train_fog_preds
        # train_ypred[:,0] = train_nonfog_preds

        train_output = pd.DataFrame()
        train_output['date_time'] = train_date_times
        train_output['round_time'] = train_round_times
        train_output['date_cycletime'] = train_cycletimes
        train_output['vis'] = train_visibilitys
        train_output['ytrue'] = train_label_trues
        train_output['fog_prob'] = train_fog_preds
        train_output['nonfog_prob'] = train_nonfog_preds

        train_output.to_csv(train_csv_file)


        test_date_times, test_round_times,test_cycletimes, test_visibilitys, test_label_trues, test_fog_preds, test_nonfog_preds = [], [], [], [], [], [], []
        for batch, sample in enumerate(data_loader_testing):

            test_date_time  = sample['date_time']
            test_date_times.append(test_date_time)

            test_round_time = sample['round_time']
            test_round_times.append(test_round_time)

            test_cycletime  = sample['date_cycletime']
            test_cycletimes.append(test_cycletime)

            test_vis = sample['vis'] 
            test_visibilitys.append(test_vis)

            test_label_true = sample['label_class']
            test_label_trues.append(test_label_true)

            input_test      = sample['input'].to(device)
            # test_attention_scores, pred_test = model(input_test)
            # pred_test = model(input_test,  g = False)
            test_visibility_indices = (test_vis - 0.5).long() 
            pred_test = model(input_test,  test_visibility_indices.to(device))#torch.LongTensor()

            softmax            = torch.nn.Softmax(dim=1)
            pred_test          = softmax(pred_test)
            # pred_test = pred_test.detach().cpu().numpy()
            # pred_test          = torch.sigmoid(pred_test)
            pred_test          = pred_test.detach().cpu().numpy()
            test_fog_preds.append(pred_test[:, 1]) #
            test_nonfog_preds.append(pred_test[:, 0]) # 

        test_date_times   = np.concatenate(test_date_times)
        test_round_times  = np.concatenate(test_round_times)
        test_cycletimes   = np.concatenate(test_cycletimes)
        test_visibilitys  = np.concatenate(test_visibilitys)
        test_label_trues  = np.concatenate(test_label_trues)
        test_fog_preds    = np.concatenate(test_fog_preds)
        test_nonfog_preds = np.concatenate(test_nonfog_preds)

        test_output = pd.DataFrame()
        test_output['date_time'] = test_date_times
        test_output['round_time'] = test_round_times
        test_output['date_cycletime'] = test_cycletimes
        test_output['vis'] = test_visibilitys
        test_output['ytrue'] = test_label_trues
        test_output['fog_prob'] = test_fog_preds
        test_output['nonfog_prob'] = test_nonfog_preds

        test_output.to_csv(test_csv_file)

    E = Evaluation(train_output, test_output, by = by, report_file_name = full_report, figure_name = figure_name_root)
    E.return_report()
    E.roc_curve()
    E.reliability_diagram()

    return [train_output, test_output]

def inference(model, data_loader, emb_type:str, SaveDir = None, Exp_name = None,):

    #'/data1/fog/SparkMET/EXPs/'
    save_dir =  SaveDir + Exp_name
    best_model_name      = save_dir + '/best_model_' + Exp_name + '.pth'
    last_epoch_model     = save_dir + '/last_epoch_' + Exp_name + '.pth'
    model.load_state_dict(torch.load(best_model_name))

    att_scores = save_dir + '/att_scores_' + Exp_name + '.npz'

    #==============================================================================================================#
    #==============================================================================================================#
    with torch.no_grad():
        date_times, round_times, cycletimes, visibilitys, label_trues, fog_preds, nonfog_preds= [], [], [], [], [], [], []
        attention_outputs = []
        input_maps = []
        attention_outputs_var, attention_outputs_pa = [], []

        for batch_idx, sample in enumerate(data_loader):

            date_time  = sample['date_time']
            date_times.append(date_time)

            round_time = sample['round_time']
            round_times.append(round_time)

            cycletime  = sample['date_cycletime']
            cycletimes.append(cycletime)

            visibility = sample['vis'] 
            visibilitys.append(visibility)

            label_true = sample['label_class']
            label_trues.append(label_true)

            input = sample['input'].to(device)

            attention_scores, pred = model(input)

            softmax = torch.nn.Softmax(dim=1)
            pred_ = softmax(pred)
            pred_ = pred_.detach().cpu().numpy()
            fog_preds.append(pred_[:, 1])

            # print(f"len: {len(attention_scores)} | shape: {attention_scores[0].shape}")
            # 
            #============================================================================================
            if emb_type == 'VVT':

                input_maps.append(input.detach().cpu().numpy())


                layer_attention_outputs = None
                for layer_output in attention_scores:
                    layer_output = layer_output[:, :, 1:, 1:].detach().cpu().numpy()
                    layer_output = np.expand_dims(layer_output, axis  = 2)
                    
                    if layer_attention_outputs is None: 
                        layer_attention_outputs = layer_output
                    else: 
                        layer_attention_outputs = np.concatenate((layer_attention_outputs, layer_output), axis = 2)
                attention_outputs.append(layer_attention_outputs) 

            elif emb_type == 'STT':

                # input_maps.append(input.detach().cpu().numpy())

                layer_attention_outputs = None
                for layer_output in attention_scores:
                    layer_output = layer_output[:, :, 1:, 1:].detach().cpu().numpy()
                    layer_output = np.expand_dims(layer_output, axis  = 2)
                    
                    if layer_attention_outputs is None: 
                        layer_attention_outputs = layer_output
                    else: 
                        layer_attention_outputs = np.concatenate((layer_attention_outputs, layer_output), axis = 2)
                attention_outputs.append(layer_attention_outputs) 


            elif emb_type == 'UVT':
                layer_attention_outputs = None

                att = attention_scores[-1][0, -1, 1:, 1:].detach().cpu().numpy()
                att_copy = att.copy()
                np.fill_diagonal(att_copy, 0)

                column_means = att_copy.mean(axis=0)
                column_means = np.expand_dims(column_means, axis  = 0) 

                if layer_attention_outputs is None: 
                    layer_attention_outputs = column_means
                else: 
                    layer_attention_outputs =  np.concatenate((layer_attention_outputs, column_means), axis = 0) 
                attention_outputs.append(layer_attention_outputs) 

    date_times   = np.concatenate(date_times)
    round_times  = np.concatenate(round_times)
    cycletimes   = np.concatenate(cycletimes)
    visibilitys  = np.concatenate(visibilitys)
    label_trues  = np.concatenate(label_trues)
    fog_preds    = np.concatenate(fog_preds)
    df = pd.DataFrame()
    df['date_time']  = date_times
    df['round_time'] = round_times
    df['date_cycletime'] = cycletimes
    df['vis'] = visibilitys
    df['ytrue'] = label_trues
    df['fog_prob'] = fog_preds

    attention_outputs   = np.concatenate(attention_outputs, axis = 0)
    if (emb_type == 'VVT') or (emb_type == 'STT'):
        input_maps        = input.detach().cpu().numpy()  
        return df, attention_outputs, input_maps
    else: 
        return df, attention_outputs

class Evaluation(): 
    def __init__(self, train_df, test_df, by: str, report_file_name = None, figure_name = None):
        self.train_df           = train_df
        self.test_df            = test_df
        self.by                 = by
        self.report_file_name   = report_file_name 
        self.figure_name        = figure_name


    def return_report(self):

        train_accuray_list, test_accuracy_list, threshold = self._return_test_train_eval()
        hit_cases, miss_cases, false_alarms               = self._return_hit_miss_fa_cases(threshold)
        
        with open(self.report_file_name, "a") as f: 
            print(f"************************************************************************", file=f)
            print(f"******************************* Report *********************************", file=f)
            print(f"************************************************************************", file=f)
            print(file=f)
            print(f"                              Training Performance                      ", file=f)
            print(f"The optimal threshold: {train_accuray_list[0]}", file=f)
            print(f"Hit: {train_accuray_list[1]}", file=f)
            print(f"Miss: {train_accuray_list[2]}", file=f)
            print(f"FA: {train_accuray_list[3]}", file=f)
            print(f"CR: {train_accuray_list[4]}", file=f)
            print(f"POD: {train_accuray_list[5]}", file=f)
            print(f"F: {train_accuray_list[6]}", file=f)
            print(f"FAR: {train_accuray_list[7]}", file=f)
            print(f"CSI: {train_accuray_list[8]}", file=f)
            print(f"PSS: {train_accuray_list[9]}", file=f)
            print(f"HSS: {train_accuray_list[10]}", file=f)
            print(f"ORSS: {train_accuray_list[11]}", file=f)
            print(f"CSS: {train_accuray_list[12]}", file=f)
            print(file=f)
            print(f"                               Test Performance                        ", file=f)
            print(f"Hit: {test_accuracy_list[0]}", file=f)
            print(f"Miss: {test_accuracy_list[1]}", file=f)
            print(f"FA: {test_accuracy_list[2]}", file=f)
            print(f"CR: {test_accuracy_list[3]}", file=f)
            print(f"POD: {test_accuracy_list[4]}", file=f)
            print(f"F: {test_accuracy_list[5]}", file=f)
            print(f"FAR: {test_accuracy_list[6]}", file=f)
            print(f"CSI: {test_accuracy_list[7]}", file=f)
            print(f"PSS: {test_accuracy_list[8]}", file=f)
            print(f"HSS: {test_accuracy_list[9]}", file=f)
            print(f"ORSS: {test_accuracy_list[10]}", file=f)
            print(f"CSS: {test_accuracy_list[11]}", file=f)
            print(file=f)
            print(f"Hit Cases (means, there is a fog event and the model correctly detect it): ", file=f)
            for idx, row in hit_cases.iterrows():
                print(f"Day {row['date_cycletime']}: visibility = {row['vis']} | prediction probability = {row['fog_prob']:.2f} ", file=f)
            print(file=f)
            print(f"Miss Cases:", file=f)
            for idx, row in miss_cases.iterrows():
                print(f"Day {row['date_cycletime']}: visibility = {row['vis']} | prediction probability = {row['fog_prob']:.2f} ", file=f)
            print(file=f)
            print(f"False Alarm Cases:", file=f)
            for idx, row in false_alarms.iterrows():
                print(f"Day {row['date_cycletime']}: visibility = {row['vis']} | prediction probability = {row['fog_prob']:.2f} ", file=f)

    def _return_hit_miss_fa_cases(self, threshold):
        # Hits are cases where the true value is 1 and the predicted probability is greater than or equal to the threshold
        hit_cases = self.test_df[(self.test_df['ytrue'] == 1) & (self.test_df['fog_prob'] >= threshold)]

        # Misses are cases where the true value is 1 but the predicted probability is less than the threshold
        miss_cases = self.test_df[(self.test_df['ytrue'] == 1) & (self.test_df['fog_prob'] < threshold)]

        # False Alarms are cases where the true value is 0 but the predicted probability is greater than or equal to the threshold
        false_alarms = self.test_df[(self.test_df['ytrue'] == 0) & (self.test_df['fog_prob'] >= threshold)]

        return hit_cases, miss_cases, false_alarms

    def _return_test_train_eval(self): 

        optimal_threshold, train_accuray_list = self._return_train_performance_maximized_by(self.train_df.ytrue, self.train_df.fog_prob, by = self.by)
        pred_classes      = self.test_df.fog_prob[:, None] >= optimal_threshold
        CR, FA, MISS, Hit = confusion_matrix(self.test_df.ytrue, pred_classes).ravel()
        
        POD  = Hit/(Hit + MISS)
        F    = FA/(FA+CR)
        FAR  = FA/(Hit+FA)
        CSI  = Hit/(Hit+FA+MISS)
        PSS  = ((Hit*CR)-(FA*MISS))/((FA+CR)*(Hit+MISS))
        HSS  = (2*((Hit*CR)-(FA*MISS)))/(((Hit+MISS)*(MISS+CR))+((Hit+FA)*(FA+CR)))
        ORSS = ((Hit*CR)-(FA*MISS))/((Hit*CR)+(FA*MISS))
        CSS  = ((Hit*CR)-(FA*MISS))/((Hit+FA)*(MISS+CR))
        
        test_accuracy_list = [Hit, MISS, FA, CR, POD, F, FAR, CSI, PSS, HSS, ORSS, CSS]

        return train_accuray_list, test_accuracy_list, optimal_threshold

    def _return_train_performance_maximized_by(self, ytrue, ypred, by = None):

        # calculate the results based on the range of threshold:
        result_list = self._return_results_on_thresholds(ytrue, ypred)

        # Dictionaries for indices
        metric_indices = {'PSS': 9, 'HSS': 10, 'CSS': 12}

        # Check if 'by' is one of the metrics
        if by in metric_indices:
            metric = result_list[:, metric_indices[by]]
            valid_metric = metric[~np.isnan(metric)]
            # Get the index of the best performance
            best_index = np.argmax(valid_metric)
            # Extract the best accuracy and threshold
            accuracy_list = result_list[best_index, :]
            optimal_threshold = result_list[best_index, 0]
            return optimal_threshold, accuracy_list
        # If 'by' is not provided or not in the metric
        raise ValueError("The 'by' parameter must be one of 'PSS', 'HSS', or 'CSS'.")
    
    def _return_results_on_thresholds(self, ytrue, ypred): 

        thresholds = np.arange(0, 1, 0.001)  # 100 thresholds from 0 to 1
        results    = np.empty((thresholds.size, 13), dtype='float')

        pred_classes = ypred[:, None] >= thresholds  # This creates a boolean array for all thresholds at once

        for i, th in enumerate(thresholds):
            pred_class = pred_classes[:, i]

            metrics_list = self._confusion_matrix(ytrue, pred_class)

            results[i] = np.concatenate(([th], metrics_list))
        
        return results
    
    def _confusion_matrix(self, ytrue, pred_class):
            
        CR, FA, MISS, Hit = confusion_matrix(ytrue, pred_class).ravel()

        POD = Hit/(Hit+MISS)
        F   = FA/(FA+CR)
        FAR  = FA/(Hit+FA)
        CSI = Hit/(Hit+FA+MISS)
        PSS = ((Hit*CR)-(FA*MISS))/((FA+CR)*(Hit+MISS))
        HSS = (2*((Hit*CR)-(FA*MISS)))/(((Hit+MISS)*(MISS+CR))+((Hit+FA)*(FA+CR)))
        ORSS = ((Hit*CR)-(FA*MISS))/((Hit*CR)+(FA*MISS))
        CSS = ((Hit*CR)-(FA*MISS))/((Hit+FA)*(MISS+CR))


        return [Hit, MISS, FA, CR, POD, F, FAR, CSI, PSS, HSS, ORSS, CSS]

    def calculate_pod_far(self, y_true, y_prob):
        # Initialize variables
        pods = []
        fars = []
        thresholds = np.arange(0, 1, 0.001)
        for i in thresholds:
            pred_classes = y_prob[:, None] >= i
            CR, FA, MISS, Hit = confusion_matrix(y_true, pred_classes).ravel()
            POD  = Hit/(Hit+MISS)
            F    = FA/(FA+CR)
            pods.append(POD)
            fars.append(F)

        return pods, fars

    def roc_curve(self):

        PODs, FARs = self.calculate_pod_far(self.test_df.ytrue, self.test_df.fog_prob)
        roc_data = {"POD": PODs, "FAR": FARs}
        roc_df_logit = pd.DataFrame(roc_data).sort_values(by = ["POD", "FAR"])

        ROC_AUC = np.trapz(roc_df_logit["POD"], roc_df_logit["FAR"])

        POD_arr = np.array(roc_df_logit["POD"]) 
        FAR_arr = np.array(roc_df_logit["FAR"])


        FAR_cutoff = 0.1
        far_idx = np.where(np.array(roc_df_logit["FAR"]) > FAR_cutoff)[0].min()
        far_idx +=1

        pAUCc, pAUC, pAUCx, pAUCcn, pAUCn, pAUCxn, sPA = concordant_partial_AUC(FAR_arr[:far_idx], 
                                                                                POD_arr[:far_idx])

        POD_cutoff = 0.7
        idx = np.where(POD_arr > POD_cutoff)[0].min()
        idx+=1
        PAI = auc(FAR_arr[idx:], POD_arr[idx:]) -  POD_arr[idx]*(1-FAR_arr[idx])
        PAI = PAI/(1-POD_arr[idx])

        fig, ax = plt.subplots(1, 1, figsize=(10,6))

        ax.plot(roc_df_logit["FAR"], roc_df_logit["POD"], linewidth=3,)
        ax.plot([0,1], [0,1],  linestyle='--', color='k')
        # ax.hlines(POD_cutoff, 0, 1, linestyle='--', color='k')
        ax.fill_between(roc_df_logit["FAR"][:far_idx], 
                        roc_df_logit["POD"][:far_idx], 
                        color='#73C6B6', 
                        label=f"pAUC: {pAUC:0.3f} (@FAR: {FAR_cutoff:0.2f})")
        
        # ax.fill_between(roc_df_logit["FAR"][idx:], roc_df_logit["POD"][idx:], 
        #                 [roc_df_logit["POD"][idx]]*len(roc_df_logit["FAR"][idx:]), 
        #                 color='r', 
        #                 label=f'PAI: {PAI:0.3f} (@POD: {POD_cutoff:0.2f})')


        ax.grid()
        ax.legend(loc='lower right')
        ax.set_xlim([-0.005, 1.0])
        ax.set_ylim([0.0, 1.005])
        ax.set_xlabel('Probability of false detection (FAR)',  fontsize=14)
        ax.set_ylabel('Probability of detection (POD)', fontsize=14)
        ax.set_title(f'AUC: {ROC_AUC:0.3f} | pAUC: {pAUC:0.3f} (@FAR: {FAR_cutoff:0.2f}) | PAI: {PAI:0.3f} (@POD: {POD_cutoff:0.2f})')
        # ax.set_title(f'AUC: {ROC_AUC:0.3f} | pAUC: {pAUC:0.3f} (@FAR: {FAR_cutoff:0.2f})')
        plt.savefig(self.figure_name + "_roc.png", dpi = 300, bbox_inches="tight", pad_inches=0.2)

    def reliability_diagram(self):
        # labels and probabilities
        observed_labels = self.test_df.ytrue.values
        forecast_probabilities = self.test_df.fog_prob.values
        # check the format and input size
        assert np.all(np.logical_or(
            observed_labels == 0, observed_labels == 1
        ))

        assert np.all(np.logical_and(
            forecast_probabilities >= 0, forecast_probabilities <= 1
        ))
        num_bins = 10
        assert num_bins > 1

        # inputs_to_bins = _get_histogram(
        #     input_values=forecast_probabilities, num_bins=num_bins, min_value=0.,
        #     max_value=1.)
        
        bin_cutoffs = np.linspace(0., 1., num=num_bins + 1)

        inputs_to_bins = np.digitize(
            forecast_probabilities, bin_cutoffs, right=False
        ) - 1

        inputs_to_bins[inputs_to_bins < 0] = 0
        inputs_to_bins[inputs_to_bins > num_bins - 1] = num_bins - 1

        mean_forecast_probs = np.full(num_bins, np.nan)
        mean_event_frequencies = np.full(num_bins, np.nan)
        num_examples_by_bin = np.full(num_bins, -1, dtype=int)

        for k in range(num_bins):
            these_example_indices = np.where(inputs_to_bins == k)[0]
            num_examples_by_bin[k] = len(these_example_indices)

            mean_forecast_probs[k] = np.mean(
                forecast_probabilities[these_example_indices])

            mean_event_frequencies[k] = np.mean(
                observed_labels[these_example_indices].astype(float)
            )

        fig, axes_object = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 6), dpi=300, 
                            gridspec_kw={"height_ratios": [4, 1]})

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.03)

        perfect_x_coords = np.array([0, 1], dtype=float)
        perfect_y_coords = perfect_x_coords + 0.
        axes_object[0].plot(
            perfect_x_coords, perfect_y_coords, color='gray',
            linestyle='dashed', linewidth=3)

        real_indices = np.where(np.invert(np.logical_or(
            np.isnan(mean_forecast_probs), np.isnan(mean_event_frequencies)
        )))[0]

        axes_object[0].plot(
            mean_forecast_probs[real_indices], mean_event_frequencies[real_indices],
            color='b',
            linestyle='solid', linewidth=1)

        accuracies = mean_event_frequencies
        confidences = mean_forecast_probs
        counts = num_examples_by_bin
        bin_cutoffs = np.linspace(0, 1, num=10 + 1)
        bins = bin_cutoffs

        bin_size = 1.0 / len(counts)
        positions = bins[:-1] + bin_size/2.0


        colors = np.zeros((10, 4))
        colors[:, 0] = 240 / 255.
        colors[:, 1] = 60 / 255.
        colors[:, 2] = 60 / 255.
        colors[:, 3] = 0.2

        gap_plt = axes_object[0].bar(positions, np.abs(accuracies - confidences), 
                            bottom=np.minimum(accuracies, confidences), width=0.1,
                            edgecolor=colors, color=colors, linewidth=1, label="Gap")

        acc_plt = axes_object[0].bar(positions, 0, bottom=accuracies, width=0.1,
                            edgecolor="black", color="black", alpha=1.0, linewidth=3,
                            label="Accuracy")

        # axes_object[0].set_xlabel('Forecast probability')
        axes_object[0].set_ylabel('Expected frequency')
        axes_object[0].set_xlim(0., 1.)
        axes_object[0].set_ylim(0., 1.)
        axes_object[0].legend(handles=[gap_plt, acc_plt])

        orig_counts = counts
        bin_data = -orig_counts
        axes_object[1].bar(positions, bin_data, width=bin_size * 0.9)

        axes_object[1].set_xlim(0, 1)
        # axes_object[1].set_yscale('log')
        axes_object[1].set_xlabel('Forecast probability')
        axes_object[1].set_ylabel('Count')

        # Also negate the ticks for the upside-down histogram.
        new_ticks = np.abs(axes_object[1].get_yticks()).astype(np.int)
        axes_object[1].set_yticklabels(new_ticks)  

        plt.savefig(self.figure_name + "_reliability_diagram.png", dpi = 300, bbox_inches="tight", pad_inches=0.2)

def concordant_partial_AUC(pfpr, ptpr):
    ''' Computes the concordant partial area under the curve and alternatives, given arrays of \n
    "partial fpr" and "partial tpr" values.  These arrays only contain points on the partial curve \n
    and the trapezoidal rule is used to compute areas with these points. \n
    \n
    pAUCc:      the concordant partial area under the curve \n
    pAUC:       the (vertical) partial area under the curve \n
    pAUCx:      the horizontal partial area under the curve \n
    pAUCc_norm: the concordant partial area under the curve normalized \n
    pAUC_norm:  the (vertical) partial area under the curve normalized \n
    pAUCx_norm: the horizontal partial area under the curve normalized \n
    '''

    # xrange [a,b]
    a    = float(pfpr[0])
    b    = float(pfpr[-1])
    delx = b - a
    vertical_stripe_area = (1 * delx)

    # yrange [f,g]
    f    = float(ptpr[0])
    g    = float(ptpr[-1])
    dely = g - f
    horizontal_stripe_area = (dely * 1)

    if delx == 0:
        print("Warning: For pAUC and pAUCc the width (delx) of the vertical column is zero.")
        pAUC  = 0
        pAUCn = 0
        sPA   = 0
    else:
        # Compute the partial AUC mathematically defined in (Dodd and Pepe, 2003) and conceptually defined in
        #   (McClish, 1989; Thompson and Zucchini, 1989). Use the trapezoidal rule to compute the integral.
        pAUC  = np.trapz(ptpr, pfpr)  # trapz is y,x
        pAUCn = pAUC / vertical_stripe_area
    #endif

    if dely == 0:
        print("Warning: For pAUCx and pAUCc the height (dely) of the horizontal stripe is zero.")
        pAUCx  = 0
        pAUCxn = 0
    else:
        # Compute the horizontal partial AUC (pAUCx) defined in (Carrington et al, 2020) and as
        # suggested by (Walter, 2005) and similar to the partial area index (PAI)
        # (Nishikawa et al, ?) although PAI has a fixed right boundary instead and a slightly
        # different mathematical definition.
        #
        # Normally we would compute the area to the right of the curve, the horizontal integral,
        # as follows:
        #   1. swap the axes
        #   2. flip the direction of the new vertical
        #   3. compute the (vertical) integral
        #
        tempX  = ptpr                      # swap the axes
        tempY  = list(1 - np.array(pfpr))  # flip the direction
        pAUCx  = np.trapz(tempY, tempX)    # trapz is y,x
        pAUCxn = pAUCx / horizontal_stripe_area
    #endif

    total_for_norm = vertical_stripe_area + horizontal_stripe_area
    if total_for_norm == 0:
        pAUCc  = 0
        pAUCcn = 0
        print('Warning: Zero length partial curve specified.')
    else:
        pAUCc  = (1/2)*pAUCx + (1/2)*pAUC  # the complexity is in the derivation, meaning/generalization
        #   and relation to the c and partial c statistic, not the
        #   formula which looks like a simple average
        #pAUCcn= (pAUCx + pAUC) / total_for_norm
        if vertical_stripe_area > 0 and horizontal_stripe_area > 0: # NEW part-wise normalization
            pAUCcn = (1 / 2) * (pAUC  / vertical_stripe_area) + (1 / 2) * (pAUCx / horizontal_stripe_area)
        elif pAUCn == 1:
            pAUCcn = 1 # best case scenario
        elif vertical_stripe_area   == 0:
            pAUCcn = (1 / 2) * (pAUCx / horizontal_stripe_area)
        elif horizontal_stripe_area == 0:
            pAUCcn = (1 / 2) * (pAUC  / vertical_stripe_area)
        # endif
    #endif
    
    sPA = 0.5*(1+ (pAUC - (pfpr[-1]**2)/2)/ (pfpr[-1] - (pfpr[-1]**2)/2))  # One the use of partial AUC. Hua Ma et all
    
    return pAUCc, pAUC, pAUCx, pAUCcn, pAUCn, pAUCxn, sPA






























# def reliability_diagram(test_df):
#     # labels and probabilities
#     observed_labels = test_df.ytrue.values
#     forecast_probabilities = test_df.fog_prob.values
#     # check the format and input size
#     assert np.all(np.logical_or(
#         observed_labels == 0, observed_labels == 1
#     ))

#     assert np.all(np.logical_and(
#         forecast_probabilities >= 0, forecast_probabilities <= 1
#     ))
#     num_bins = 10
#     assert num_bins > 1

#     # inputs_to_bins = _get_histogram(
#     #     input_values=forecast_probabilities, num_bins=num_bins, min_value=0.,
#     #     max_value=1.)
    
#     bin_cutoffs = np.linspace(0., 1., num=num_bins + 1)

#     inputs_to_bins = np.digitize(
#         forecast_probabilities, bin_cutoffs, right=False
#     ) - 1

#     inputs_to_bins[inputs_to_bins < 0] = 0
#     inputs_to_bins[inputs_to_bins > num_bins - 1] = num_bins - 1

#     mean_forecast_probs = np.full(num_bins, np.nan)
#     mean_event_frequencies = np.full(num_bins, np.nan)
#     num_examples_by_bin = np.full(num_bins, -1, dtype=int)

#     for k in range(num_bins):
#         these_example_indices = np.where(inputs_to_bins == k)[0]
#         num_examples_by_bin[k] = len(these_example_indices)

#         mean_forecast_probs[k] = np.mean(
#             forecast_probabilities[these_example_indices])

#         mean_event_frequencies[k] = np.mean(
#             observed_labels[these_example_indices].astype(float)
#         )

#     fig, axes_object = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 6), dpi=300, 
#                            gridspec_kw={"height_ratios": [4, 1]})

#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.03)

#     perfect_x_coords = np.array([0, 1], dtype=float)
#     perfect_y_coords = perfect_x_coords + 0.
#     axes_object[0].plot(
#         perfect_x_coords, perfect_y_coords, color='gray',
#         linestyle='dashed', linewidth=3)

#     real_indices = np.where(np.invert(np.logical_or(
#         np.isnan(mean_forecast_probs), np.isnan(mean_event_frequencies)
#     )))[0]

#     axes_object[0].plot(
#         mean_forecast_probs[real_indices], mean_event_frequencies[real_indices],
#         color='b',
#         linestyle='solid', linewidth=1)

#     accuracies = mean_event_frequencies
#     confidences = mean_forecast_probs
#     counts = num_examples_by_bin
#     bin_cutoffs = np.linspace(0, 1, num=10 + 1)
#     bins = bin_cutoffs

#     bin_size = 1.0 / len(counts)
#     positions = bins[:-1] + bin_size/2.0


#     colors = np.zeros((10, 4))
#     colors[:, 0] = 240 / 255.
#     colors[:, 1] = 60 / 255.
#     colors[:, 2] = 60 / 255.
#     colors[:, 3] = 0.2

#     gap_plt = axes_object[0].bar(positions, np.abs(accuracies - confidences), 
#                         bottom=np.minimum(accuracies, confidences), width=0.1,
#                         edgecolor=colors, color=colors, linewidth=1, label="Gap")

#     acc_plt = axes_object[0].bar(positions, 0, bottom=accuracies, width=0.1,
#                         edgecolor="black", color="black", alpha=1.0, linewidth=3,
#                         label="Accuracy")

#     # axes_object[0].set_xlabel('Forecast probability')
#     axes_object[0].set_ylabel('Expected frequency')
#     axes_object[0].set_xlim(0., 1.)
#     axes_object[0].set_ylim(0., 1.)
#     axes_object[0].legend(handles=[gap_plt, acc_plt])

#     orig_counts = counts
#     bin_data = -orig_counts
#     axes_object[1].bar(positions, bin_data, width=bin_size * 0.9)

#     axes_object[1].set_xlim(0, 1)
#     # axes_object[1].set_yscale('log')
#     axes_object[1].set_xlabel('Forecast probability')
#     axes_object[1].set_ylabel('Count')

#     # Also negate the ticks for the upside-down histogram.
#     new_ticks = np.abs(axes_object[1].get_yticks()).astype(np.int)
#     axes_object[1].set_yticklabels(new_ticks)  

#     # plt.savefig(self.figure_name + "_reliability_diagram.png", dpi = 300, bbox_inches="tight", pad_inches=0.2)
        
# def _get_histogram(input_values, num_bins, min_value, max_value):
#     """Creates histogram with uniform bin-spacing.
#     E = number of input values
#     B = number of bins
#     :param input_values: length-E numpy array of values to bin.
#     :param num_bins: Number of bins (B).
#     :param min_value: Minimum value.  Any input value < `min_value` will be
#         assigned to the first bin.
#     :param max_value: Max value.  Any input value > `max_value` will be
#         assigned to the last bin.
#     :return: inputs_to_bins: length-E numpy array of bin indices (integers).
#     """

#     bin_cutoffs = np.linspace(min_value, max_value, num=num_bins + 1)

#     inputs_to_bins = np.digitize(
#         input_values, bin_cutoffs, right=False
#     ) - 1

#     inputs_to_bins[inputs_to_bins < 0] = 0
#     inputs_to_bins[inputs_to_bins > num_bins - 1] = num_bins - 1

#     return inputs_to_bins

# def get_points_in_relia_curve(
#         observed_labels, forecast_probabilities, num_bins):
#     """Creates points for reliability curve.
#     The reliability curve is the main component of the attributes diagram.
#     E = number of examples
#     B = number of bins
#     :param observed_labels: length-E numpy array of class labels (integers in
#         0...1).
#     :param forecast_probabilities: length-E numpy array with forecast
#         probabilities of label = 1.
#     :param num_bins: Number of bins for forecast probability.
#     :return: mean_forecast_probs: length-B numpy array of mean forecast
#         probabilities.
#     :return: mean_event_frequencies: length-B numpy array of conditional mean
#         event frequencies.  mean_event_frequencies[j] = frequency of label 1
#         when forecast probability is in the [j]th bin.
#     :return: num_examples_by_bin: length-B numpy array with number of examples
#         in each forecast bin.
#     """

#     assert np.all(np.logical_or(
#         observed_labels == 0, observed_labels == 1
#     ))

#     assert np.all(np.logical_and(
#         forecast_probabilities >= 0, forecast_probabilities <= 1
#     ))

#     assert num_bins > 1

#     # inputs_to_bins = _get_histogram(
#     #     input_values=forecast_probabilities, num_bins=num_bins, min_value=0.,
#     #     max_value=1.)
    
#     bin_cutoffs = np.linspace(0., 1., num=num_bins + 1)

#     inputs_to_bins = np.digitize(
#         forecast_probabilities, bin_cutoffs, right=False
#     ) - 1

#     inputs_to_bins[inputs_to_bins < 0] = 0
#     inputs_to_bins[inputs_to_bins > num_bins - 1] = num_bins - 1

#     mean_forecast_probs = np.full(num_bins, np.nan)
#     mean_event_frequencies = np.full(num_bins, np.nan)
#     num_examples_by_bin = np.full(num_bins, -1, dtype=int)

#     for k in range(num_bins):
#         these_example_indices = np.where(inputs_to_bins == k)[0]
#         num_examples_by_bin[k] = len(these_example_indices)

#         mean_forecast_probs[k] = np.mean(
#             forecast_probabilities[these_example_indices])

#         mean_event_frequencies[k] = np.mean(
#             observed_labels[these_example_indices].astype(float)
#         )

#     return mean_forecast_probs, mean_event_frequencies, num_examples_by_bin

# def plot_reliability_curve(
#         observed_labels, forecast_probabilities, num_bins=10,
#         axes_object=None):
#     """Plots reliability curve.
#     E = number of examples
#     :param observed_labels: length-E numpy array of class labels (integers in
#         0...1).
#     :param forecast_probabilities: length-E numpy array with forecast
#         probabilities of label = 1.
#     :param num_bins: Number of bins for forecast probability.
#     :param axes_object: Will plot on these axes (instance of
#         `matplotlib.axes._subplots.AxesSubplot`).  If `axes_object is None`,
#         will create new axes.
#     :return: mean_forecast_probs: See doc for `get_points_in_relia_curve`.
#     :return: mean_event_frequencies: Same.
#     :return: num_examples_by_bin: Same.
#     """

#     # mean_forecast_probs, mean_event_frequencies, num_examples_by_bin = (
#     #     get_points_in_relia_curve(
#     #         observed_labels=observed_labels,
#     #         forecast_probabilities=forecast_probabilities, num_bins=num_bins)
#     # )

#     assert np.all(np.logical_or(
#         observed_labels == 0, observed_labels == 1
#     ))

#     assert np.all(np.logical_and(
#         forecast_probabilities >= 0, forecast_probabilities <= 1
#     ))

#     assert num_bins > 1

#     # inputs_to_bins = _get_histogram(
#     #     input_values=forecast_probabilities, num_bins=num_bins, min_value=0.,
#     #     max_value=1.)
    
#     bin_cutoffs = np.linspace(0., 1., num=num_bins + 1)

#     inputs_to_bins = np.digitize(
#         forecast_probabilities, bin_cutoffs, right=False
#     ) - 1

#     inputs_to_bins[inputs_to_bins < 0] = 0
#     inputs_to_bins[inputs_to_bins > num_bins - 1] = num_bins - 1

#     mean_forecast_probs = np.full(num_bins, np.nan)
#     mean_event_frequencies = np.full(num_bins, np.nan)
#     num_examples_by_bin = np.full(num_bins, -1, dtype=int)

#     for k in range(num_bins):
#         these_example_indices = np.where(inputs_to_bins == k)[0]
#         num_examples_by_bin[k] = len(these_example_indices)

#         mean_forecast_probs[k] = np.mean(
#             forecast_probabilities[these_example_indices])

#         mean_event_frequencies[k] = np.mean(
#             observed_labels[these_example_indices].astype(float)
#         )

#     fig, axes_object = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 6), dpi=300, 
#                            gridspec_kw={"height_ratios": [4, 1]})

#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.03)

#     perfect_x_coords = np.array([0, 1], dtype=float)
#     perfect_y_coords = perfect_x_coords + 0.
#     axes_object[0].plot(
#         perfect_x_coords, perfect_y_coords, color='gray',
#         linestyle='dashed', linewidth=3)

#     real_indices = np.where(np.invert(np.logical_or(
#         np.isnan(mean_forecast_probs), np.isnan(mean_event_frequencies)
#     )))[0]

#     axes_object[0].plot(
#         mean_forecast_probs[real_indices], mean_event_frequencies[real_indices],
#         color='b',
#         linestyle='solid', linewidth=1)

#     accuracies = mean_event_frequencies
#     confidences = mean_forecast_probs
#     counts = num_examples_by_bin
#     bin_cutoffs = np.linspace(0, 1, num=10 + 1)
#     bins = bin_cutoffs

#     bin_size = 1.0 / len(counts)
#     positions = bins[:-1] + bin_size/2.0


#     colors = np.zeros((10, 4))
#     colors[:, 0] = 240 / 255.
#     colors[:, 1] = 60 / 255.
#     colors[:, 2] = 60 / 255.
#     colors[:, 3] = 0.2

#     gap_plt = axes_object[0].bar(positions, np.abs(accuracies - confidences), 
#                         bottom=np.minimum(accuracies, confidences), width=0.1,
#                         edgecolor=colors, color=colors, linewidth=1, label="Gap")

#     acc_plt = axes_object[0].bar(positions, 0, bottom=accuracies, width=0.1,
#                         edgecolor="black", color="black", alpha=1.0, linewidth=3,
#                         label="Accuracy")

#     # axes_object[0].set_xlabel('Forecast probability')
#     axes_object[0].set_ylabel('Expected frequency')
#     axes_object[0].set_xlim(0., 1.)
#     axes_object[0].set_ylim(0., 1.)
#     axes_object[0].legend(handles=[gap_plt, acc_plt])

#     orig_counts = counts
#     bin_data = -orig_counts
#     axes_object[1].bar(positions, bin_data, width=bin_size * 0.9)

#     axes_object[1].set_xlim(0, 1)
#     # axes_object[1].set_yscale('log')
#     axes_object[1].set_xlabel('Forecast probability')
#     axes_object[1].set_ylabel('Count')

#     # Also negate the ticks for the upside-down histogram.
#     new_ticks = np.abs(axes_object[1].get_yticks()).astype(np.int)
#     axes_object[1].set_yticklabels(new_ticks)  
        
#     # return mean_forecast_probs, mean_event_frequencies, num_examples_by_bin

# def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
#     """Collects predictions into bins used to draw a reliability diagram.

#     Arguments:
#         true_labels: the true labels for the test examples
#         pred_labels: the predicted labels for the test examples
#         confidences: the predicted confidences for the test examples
#         num_bins: number of bins

#     The true_labels, pred_labels, confidences arguments must be NumPy arrays;
#     pred_labels and true_labels may contain numeric or string labels.

#     For a multi-class model, the predicted label and confidence should be those
#     of the highest scoring class.

#     Returns a dictionary containing the following NumPy arrays:
#         accuracies: the average accuracy for each bin
#         confidences: the average confidence for each bin
#         counts: the number of examples in each bin
#         bins: the confidence thresholds for each bin
#         avg_accuracy: the accuracy over the entire test set
#         avg_confidence: the average confidence over the entire test set
#         expected_calibration_error: a weighted average of all calibration gaps
#         max_calibration_error: the largest calibration gap across all bins
#     """
#     assert(len(confidences) == len(pred_labels))
#     assert(len(confidences) == len(true_labels))
#     assert(num_bins > 0)

#     bin_size = 1.0 / num_bins
#     bins = np.linspace(0.0, 1.0, num_bins + 1)
#     indices = np.digitize(confidences, bins, right=True)

#     bin_accuracies = np.zeros(num_bins, dtype=np.float)
#     bin_confidences = np.zeros(num_bins, dtype=np.float)
#     bin_counts = np.zeros(num_bins, dtype=np.int)

#     for b in range(num_bins):
#         selected = np.where(indices == b + 1)[0]
#         if len(selected) > 0:
#             bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
#             bin_confidences[b] = np.mean(confidences[selected])
#             bin_counts[b] = len(selected)

#     avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
#     avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

#     gaps = np.abs(bin_accuracies - bin_confidences)
#     ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
#     mce = np.max(gaps)

#     return { "accuracies": bin_accuracies, 
#              "confidences": bin_confidences, 
#              "counts": bin_counts, 
#              "bins": bins,
#              "avg_accuracy": avg_acc,
#              "avg_confidence": avg_conf,
#              "expected_calibration_error": ece,
#              "max_calibration_error": mce }

# def _reliability_diagram_subplot(ax, bin_data, 
#                                  draw_ece=True, 
#                                  draw_bin_importance=False,
#                                  title="Reliability Diagram", 
#                                  xlabel="Confidence", 
#                                  ylabel="Expected Accuracy"):
#     """Draws a reliability diagram into a subplot."""
#     accuracies = bin_data["accuracies"]
#     confidences = bin_data["confidences"]
#     counts = bin_data["counts"]
#     bins = bin_data["bins"]

#     bin_size = 1.0 / len(counts)
#     positions = bins[:-1] + bin_size/2.0

#     widths = bin_size
#     alphas = 0.3
#     min_count = np.min(counts)
#     max_count = np.max(counts)
#     normalized_counts = (counts - min_count) / (max_count - min_count)

#     if draw_bin_importance == "alpha":
#         alphas = 0.2 + 0.8*normalized_counts
#     elif draw_bin_importance == "width":
#         widths = 0.1*bin_size + 0.9*bin_size*normalized_counts

#     colors = np.zeros((len(counts), 4))
#     colors[:, 0] = 240 / 255.
#     colors[:, 1] = 60 / 255.
#     colors[:, 2] = 60 / 255.
#     colors[:, 3] = alphas

#     gap_plt = ax.bar(positions, np.abs(accuracies - confidences), 
#                      bottom=np.minimum(accuracies, confidences), width=widths,
#                      edgecolor=colors, color=colors, linewidth=1, label="Gap")

#     acc_plt = ax.bar(positions, 0, bottom=accuracies, width=widths,
#                      edgecolor="black", color="black", alpha=1.0, linewidth=3,
#                      label="Accuracy")

#     ax.set_aspect("equal")
#     ax.plot([0,1], [0,1], linestyle = "--", color="gray")
    
#     if draw_ece:
#         ece = (bin_data["expected_calibration_error"] * 100)
#         ax.text(0.98, 0.02, "ECE=%.2f" % ece, color="black", 
#                 ha="right", va="bottom", transform=ax.transAxes)

#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     #ax.set_xticks(bins)

#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     ax.legend(handles=[gap_plt, acc_plt])

# def _confidence_histogram_subplot(ax, bin_data, 
#                                   draw_averages=True,
#                                   title="Examples per bin", 
#                                   xlabel="Confidence",
#                                   ylabel="Count"):
#     """Draws a confidence histogram into a subplot."""
#     counts = bin_data["counts"]
#     bins = bin_data["bins"]

#     bin_size = 1.0 / len(counts)
#     positions = bins[:-1] + bin_size/2.0

#     ax.bar(positions, counts, width=bin_size * 0.9)
   
#     ax.set_xlim(0, 1)
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     if draw_averages:
#         acc_plt = ax.axvline(x=bin_data["avg_accuracy"], ls="solid", lw=3, 
#                              c="black", label="Accuracy")
#         conf_plt = ax.axvline(x=bin_data["avg_confidence"], ls="dotted", lw=3, 
#                               c="#444", label="Avg. confidence")
#         ax.legend(handles=[acc_plt, conf_plt])

# def _reliability_diagram_combined(bin_data, 
#                                   draw_ece, draw_bin_importance, draw_averages, 
#                                   title, figsize, dpi, return_fig):
#     """Draws a reliability diagram and confidence histogram using the output
#     from compute_calibration()."""
#     figsize = (figsize[0], figsize[0] * 1.4)

#     fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi, 
#                            gridspec_kw={"height_ratios": [4, 1]})

#     plt.tight_layout()
#     plt.subplots_adjust(hspace=-0.1)

#     _reliability_diagram_subplot(ax[0], bin_data, draw_ece, draw_bin_importance, 
#                                  title=title, xlabel="")

#     # Draw the confidence histogram upside down.
#     orig_counts = bin_data["counts"]
#     bin_data["counts"] = -bin_data["counts"]
#     _confidence_histogram_subplot(ax[1], bin_data, draw_averages, title="")
#     bin_data["counts"] = orig_counts

#     # Also negate the ticks for the upside-down histogram.
#     new_ticks = np.abs(ax[1].get_yticks()).astype(np.int)
#     ax[1].set_yticklabels(new_ticks)    

#     plt.show()

#     if return_fig: return fig

# def reliability_diagram(true_labels, pred_labels, confidences, num_bins=10,
#                         draw_ece=True, draw_bin_importance=False, 
#                         draw_averages=True, title="Reliability Diagram", 
#                         figsize=(6, 6), dpi=72, return_fig=False):
#     """Draws a reliability diagram and confidence histogram in a single plot.
    
#     First, the model's predictions are divided up into bins based on their
#     confidence scores.

#     The reliability diagram shows the gap between average accuracy and average 
#     confidence in each bin. These are the red bars.

#     The black line is the accuracy, the other end of the bar is the confidence.

#     Ideally, there is no gap and the black line is on the dotted diagonal.
#     In that case, the model is properly calibrated and we can interpret the
#     confidence scores as probabilities.

#     The confidence histogram visualizes how many examples are in each bin. 
#     This is useful for judging how much each bin contributes to the calibration
#     error.

#     The confidence histogram also shows the overall accuracy and confidence. 
#     The closer these two lines are together, the better the calibration.
    
#     The ECE or Expected Calibration Error is a summary statistic that gives the
#     difference in expectation between confidence and accuracy. In other words,
#     it's a weighted average of the gaps across all bins. A lower ECE is better.

#     Arguments:
#         true_labels: the true labels for the test examples
#         pred_labels: the predicted labels for the test examples
#         confidences: the predicted confidences for the test examples
#         num_bins: number of bins
#         draw_ece: whether to include the Expected Calibration Error
#         draw_bin_importance: whether to represent how much each bin contributes
#             to the total accuracy: False, "alpha", "widths"
#         draw_averages: whether to draw the overall accuracy and confidence in
#             the confidence histogram
#         title: optional title for the plot
#         figsize: setting for matplotlib; height is ignored
#         dpi: setting for matplotlib
#         return_fig: if True, returns the matplotlib Figure object
#     """
#     bin_data = compute_calibration(true_labels, pred_labels, confidences, num_bins)
#     return _reliability_diagram_combined(bin_data, draw_ece, draw_bin_importance,
#                                          draw_averages, title, figsize=figsize, 
#                                          dpi=dpi, return_fig=return_fig)

# def reliability_diagrams(results, num_bins=10,
#                          draw_ece=True, draw_bin_importance=False, 
#                          num_cols=4, dpi=72, return_fig=False):
#     """Draws reliability diagrams for one or more models.
    
#     Arguments:
#         results: dictionary where the key is the model name and the value is
#             a dictionary containing the true labels, predicated labels, and
#             confidences for this model
#         num_bins: number of bins
#         draw_ece: whether to include the Expected Calibration Error
#         draw_bin_importance: whether to represent how much each bin contributes
#             to the total accuracy: False, "alpha", "widths"
#         num_cols: how wide to make the plot
#         dpi: setting for matplotlib
#         return_fig: if True, returns the matplotlib Figure object
#     """
#     ncols = num_cols
#     nrows = (len(results) + ncols - 1) // ncols
#     figsize = (ncols * 4, nrows * 4)

#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, 
#                            figsize=figsize, dpi=dpi, constrained_layout=True)

#     for i, (plot_name, data) in enumerate(results.items()):
#         y_true = data["true_labels"]
#         y_pred = data["pred_labels"]
#         y_conf = data["confidences"]
        
#         bin_data = compute_calibration(y_true, y_pred, y_conf, num_bins)
        
#         row = i // ncols
#         col = i % ncols
#         _reliability_diagram_subplot(ax[row, col], bin_data, draw_ece, 
#                                      draw_bin_importance, 
#                                      title="\n".join(plot_name.split()),
#                                      xlabel="Confidence" if row == nrows - 1 else "",
#                                      ylabel="Expected Accuracy" if col == 0 else "")

#     for i in range(i + 1, nrows * ncols):
#         row = i // ncols
#         col = i % ncols        
#         ax[row, col].axis("off")
        
#     plt.show()

#     if return_fig: return fig

# class Evaluation2(): 
#     def __init__(self, ytrue, ypred, report_file_name = None, figure_name = None):
#         self.ytrue     = ytrue
#         self.ypred     = ypred
#         self.report_file_name  = report_file_name 
#         self.figure_name = figure_name

#     def confusion_matrix_calc(self): 

#         ypred_class = np.argmax(self.ypred, axis = 1)

#         CR, FA, miss, Hit = confusion_matrix(self.ytrue, ypred_class).ravel()
#         POD   = Hit/(Hit+miss)
#         F     = FA/(FA+CR)
#         FAR   = FA/(Hit+FA)
#         CSI   = Hit/(Hit+FA+miss)
#         PSS   = ((Hit*CR)-(FA*miss))/((FA+CR)*(Hit+miss))
#         HSS   = (2*((Hit*CR)-(FA*miss)))/(((Hit+miss)*(miss+CR))+((Hit+FA)*(FA+CR)))
#         ORSS  = ((Hit*CR)-(FA*miss))/((Hit*CR)+(FA*miss))
#         CSS   = ((Hit*CR)-(FA*miss))/((Hit+FA)*(miss+CR))

#         #SEDI = (log(F) - log(POD) - log(1-F) + log(1-POD))/(log(F) + log(POD) + log(1-F) + log(1-POD))

#         output = [Hit, miss, FA, CR, POD, F, FAR, CSI, PSS, HSS, ORSS, CSS]

#         with open(self.report_file_name, "a") as f: 
#             # for m in output:
#             #     name = str(m) 
#             #output= [
#                 #'output = ['CSS', 'PSS' …]
#                 #for m in output:
#                 #print f'm: {eval{m)}', 
#             #     print(f"{name}: {m}", file=f)
#             print(f"Hit: {Hit}", file=f)
#             print(f"Miss: {miss}", file=f)
#             print(f"FA: {FA}", file=f)
#             print(f"CR: {CR}", file=f)
#             print(f"POD: {POD}", file=f)
#             print(f"F: {F}", file=f)
#             print(f"FAR: {FAR}", file=f)
#             print(f"CSI: {CSI}", file=f)
#             print(f"PSS: {PSS}", file=f)
#             print(f"HSS: {HSS}", file=f)
#             print(f"ORSS: {ORSS}", file=f)
#             print(f"CSS: {CSS}", file=f)
#             #print(f"SEDI: {SEDI}", file=f)

#         return output
        
#     def ruc_curve_plot(self): 
#         ypred_fog = self.ypred[:, 1] 
#         fpr, tpr, thresholds = roc_curve(self.ytrue, ypred_fog)

#         ROC_AUC = auc(fpr, tpr)     
#         fig, ax = plt.subplots(figsize = (8, 6))
        
#         ax.plot(fpr, tpr, linewidth=3, color = 'red')
#         ax.plot([0, 1], [0, 1], 'k--')
#         ax.set_xlim([-0.005, 1.0])
#         ax.set_ylim([0.0, 1.005])
#         ax.set_xlabel('FAR (probability of false detection)',  fontsize=16)
#         ax.set_ylabel('POD (probability of detection)', fontsize=16)
#         title_string = 'ROC curve (AUC = {0:.3f})'.format(ROC_AUC)
#         ax.set_title(title_string, fontsize=16)
#         plt.savefig(self.figure_name, dpi = 300)

#     def BS_BSS_calc(self): 

#         ytrue_rev    = ytrue.copy()
#         indices_one  = ytrue_rev == 1
#         indices_zero = ytrue_rev == 0
#         ytrue_rev[indices_one] = 0 # replacing 1s with 0s
#         ytrue_rev[indices_zero] = 1 # replacing 0s with 1s
        
        
#         P_c = np.mean(ytrue_rev)
#         bs_init = 0
#         bss_init = 0
#         for e in range(len(ytrue_rev)):
#             bss_init  = bs_init + (P_c - ytrue_rev[e])**2 # average value of fog accurence 
            
#             if ytrue_rev[e] == 0:
#                 prob = ypred[e, 1]
#                 bs_init  = bs_init + (prob - 0)**2
                
#             elif ytrue_rev[e] == 1:
#                 prob = ypred[e, 0]
#                 bs_init  = bs_init + (prob - 1)**2
                
#         BS     = bs_init/len(ytrue_rev)
#         BS_ref = bss_init/len(ytrue_rev)
#         BSS    = (1-BS)/BS_ref 
        
#         return BS, BSS



# def attention_map_visualize(df, inputs, attention_maps, variable:int, date:str): 

#     idx = df[df['date_cycletime'] == date].index[0]
#     num_heads  = attention_maps.shape[1]
#     num_layers = attention_maps.shape[4]

#     fig, axs = plt.subplots(4, 8, figsize = (24, 12))

#     img1 = inputs[idx, variable, ::-1, :]
#     min_ = np.min(attention_maps[idx, :, :, :, :])

#     for i in range(num_layers):
#         for j in range(num_heads):
#             img2 = attention_maps[idx, j, ::-1, :, i]
#             vis1 = axs[i, j].imshow(img1, cmap = 'gray')
#             vis2 = axs[i, j].imshow(img2, alpha=0.4)
            
#             vis2.set_clim(0.0589, 0.062)

#             if j == num_heads -1: 
#                 divider = make_axes_locatable(axs[i, j])
#                 cax = divider.append_axes("right", size="5%", pad=0.1)
#                 cbar1 = fig.colorbar(vis2, cax=cax)
#             axs[i, j].set_xticks([])
#             axs[i, j].set_yticks([])

#     for i in range(num_layers): 
#         axs[i, 0].set_ylabel(r'Layer {0}'.format(i+1))
#     for j in range(num_heads):
#         axs[0, j].set_title(r'Head {0}'.format(j+1))

    
#     plt.show()

# def single_spatial_att_map(df, inputs, attention_maps, global_case: str, local_date:str, head_avg = False): 


#     if local_date: 
#         idx = df[df['date_cycletime'] == local_date].index[0]
#         map = inputs[idx, -1, :, :, 93]

#         att_map = _return_spatial_scores(idx, attention_maps, head = -1, head_avg = head_avg)

#     else:
#         if global_case == 'Hit':
#             # Hits are cases where the true value is 1 and the predicted probability is greater than or equal to the threshold
#             cases         = df[(df['ytrue'] == 1) & (df['fog_prob'] >= 0.51)]
#             cases_indices = cases.index
#             sub_attention_maps = attention_maps[cases_indices, :]
#             att_map = _return_spatial_scores_global(sub_attention_maps)

            
#         elif global_case == 'Miss':
#             # Misses are cases where the true value is 1 but the predicted probability is less than the threshold
#             cases = df[(df['ytrue'] == 1) & (df['fog_prob'] < 0.51)]
#             cases_indices  = cases.index
#             sub_attention_maps = attention_maps[cases_indices, :]
#             att_map = _return_spatial_scores_global(sub_attention_maps)

#         elif global_case == 'FA':
#             # False Alarms are cases where the true value is 0 but the predicted probability is greater than or equal to the threshold
#             cases = df[(df['ytrue'] == 0) & (df['fog_prob'] >= 0.51)]
#             cases_indices = cases.index
#             sub_attention_maps = attention_maps[cases_indices, :]
#             att_map = _return_spatial_scores_global(sub_attention_maps)

#         elif global_case == 'CR':
#             # Corrected Rejection are cases where the true value is 0 and the predicted probability is less than the threshold
#             cases = df[(df['ytrue'] == 0) & (df['fog_prob'] < 0.51)]
#             cases_indices = cases.index
#             sub_attention_maps = attention_maps[cases_indices, :]
#             att_map = _return_spatial_scores_global(sub_attention_maps)

#         map = inputs[0, -1, :, :, 93]



#     fig, axs = plt.subplots(1, 3, figsize = (21, 7))
#     axs[0].imshow(map, cmap = 'gray')
    
#     vis2 = axs[1].imshow(att_map)
#     divider = make_axes_locatable(axs[1])
#     cax = divider.append_axes("right", size="5%", pad=0.1)
#     cbar1 = fig.colorbar(vis2, cax=cax)
    
#     axs[2].imshow(map, cmap = 'gray')
#     axs[2].imshow(att_map, alpha=0.4)


#     plt.show()

# def sp_att_map(df, inputs, attention_maps, global_case: str, local_date:str, threshold: int): 

#     if local_date: 
#         idx = df[df['date_cycletime'] == local_date].index[0]
#         map = inputs[0, -1, :, :, 93]

#         att_maps = _return_sp_scores(attention_maps, local_idx = idx)

#     else:
#         if global_case == 'Hit':
#             # Hits are cases where the true value is 1 and the predicted probability is greater than or equal to the threshold
#             cases         = df[(df['ytrue'] == 1) & (df['fog_prob'] >= threshold)]
#             cases_indices = cases.index
#             sub_attention_maps = attention_maps[cases_indices, :]
#             att_maps = _return_sp_scores(sub_attention_maps, local_idx = None)

#         elif global_case == 'Miss':
#             # Misses are cases where the true value is 1 but the predicted probability is less than the threshold
#             cases = df[(df['ytrue'] == 1) & (df['fog_prob'] < threshold)]
#             cases_indices  = cases.index
#             sub_attention_maps = attention_maps[cases_indices, :]
#             att_maps = _return_sp_scores(sub_attention_maps, local_idx = None)

#         elif global_case == 'FA':
#             # False Alarms are cases where the true value is 0 but the predicted probability is greater than or equal to the threshold
#             cases = df[(df['ytrue'] == 0) & (df['fog_prob'] >= threshold)]
#             cases_indices = cases.index
#             sub_attention_maps = attention_maps[cases_indices, :]
#             att_maps = _return_sp_scores(sub_attention_maps, local_idx = None)

#         elif global_case == 'CR':
#             # Corrected Rejection are cases where the true value is 0 and the predicted probability is less than the threshold
#             cases = df[(df['ytrue'] == 0) & (df['fog_prob'] < threshold)]
#             cases_indices = cases.index
#             sub_attention_maps = attention_maps[cases_indices, :]
#             att_maps = _return_sp_scores(sub_attention_maps, local_idx = None)

#         map = inputs[0, -1, :, :, 93]

#     # Get the indices that would sort the array in descending order
#     time_scores = np.array((np.mean(att_maps[0]), np.mean(att_maps[1]), np.mean(att_maps[2]), np.mean(att_maps[3])))
#     sorted_indices = np.argsort(time_scores)[::-1]

#     # Get the top ten values and their corresponding indices
#     top_values = time_scores[sorted_indices]
#     top_indices = sorted_indices
#     for i, (value, index) in enumerate(zip(top_values, top_indices)):
#             print(f"Rank {i+1}: Value = {value}, time_step = {index}")
        

#     fig, axs = plt.subplots(1, 4, figsize = (24, 6), sharey= True)
#     for i in range(4):
#         axs[i].imshow(map, cmap = 'gray')
#         vis_0 = axs[i].imshow(att_maps[i], alpha=0.4)
#         vis_0.set_clim(0.01, 0.030)
#         if i == 3: 
#             divider = make_axes_locatable(axs[i])
#             cax = divider.append_axes("right", size="5%", pad=0.1)
#             cbar1 = fig.colorbar(vis_0, cax=cax)


#     plt.show()

# def _return_sp_scores(attention_maps, local_idx: str):

#     if local_idx: 
#         att_st = attention_maps[local_idx, -1, -1, :, :]
#     else: 
#         att_st = attention_maps[:, -1, -1, :, :]
#         att_st = np.mean(att_st, axis = 0)

#     att_copy = att_st.copy()
#     np.fill_diagonal(att_copy, 0)
#     column_sums = att_copy.mean(axis=0)

#     map_t0 = column_sums[:16]
#     map_t1 = column_sums[16:32]
#     map_t2 = column_sums[32:48]
#     map_t3 = column_sums[48:]

#     expanded_matrix_0 = np.zeros((32, 32))
#     expanded_matrix_1 = np.zeros((32, 32))
#     expanded_matrix_2 = np.zeros((32, 32))
#     expanded_matrix_3 = np.zeros((32, 32))


#     # Fill each value into an 8x8 block
#     for i, value in enumerate(map_t0):
#         # Calculate the starting row and column for the current block
#         start_row = (i // 4) * 8
#         start_col = (i % 4) * 8
        
#         # Fill the 8x8 block with the current value
#         expanded_matrix_0[start_row:start_row+8, start_col:start_col+8] = value

#     for i, value in enumerate(map_t1):
#         # Calculate the starting row and column for the current block
#         start_row = (i // 4) * 8
#         start_col = (i % 4) * 8
        
#         # Fill the 8x8 block with the current value
#         expanded_matrix_1[start_row:start_row+8, start_col:start_col+8] = value

#     for i, value in enumerate(map_t2):
#         # Calculate the starting row and column for the current block
#         start_row = (i // 4) * 8
#         start_col = (i % 4) * 8
        
#         # Fill the 8x8 block with the current value
#         expanded_matrix_2[start_row:start_row+8, start_col:start_col+8] = value

#     for i, value in enumerate(map_t3):
#         # Calculate the starting row and column for the current block
#         start_row = (i // 4) * 8
#         start_col = (i % 4) * 8
        
#         # Fill the 8x8 block with the current value
#         expanded_matrix_3[start_row:start_row+8, start_col:start_col+8] = value

#     return [expanded_matrix_0, expanded_matrix_1, expanded_matrix_2, expanded_matrix_3]

# def _return_spatial_scores(idx, attention_maps, head:int, head_avg = False):

#     if head_avg: 
#         att = attention_maps[idx, :, -1, :, :].mean(axis = 0)
#     else: 
#         if head is None:
#             head = -1
#         att = attention_maps[idx, head, -1, :, :]

#     att_copy = att.copy()
#     np.fill_diagonal(att_copy, 0)

#     column_sums = att_copy.mean(axis=0)

#     expanded_matrix = np.zeros((32, 32))

#     # Fill each value into an 8x8 block
#     for i, value in enumerate(column_sums):
#         # Calculate the starting row and column for the current block
#         start_row = (i // 4) * 8
#         start_col = (i % 4) * 8
        
#         # Fill the 8x8 block with the current value
#         expanded_matrix[start_row:start_row+8, start_col:start_col+8] = value


#     return expanded_matrix

# def _return_spatial_scores_global(attention_maps,):

#     att = attention_maps[:, -1, -1, :, :]
#     att = np.mean(att, axis = 0)

#     att_copy = att.copy()
#     np.fill_diagonal(att_copy, 0)

#     column_sums = att_copy.mean(axis=0)

#     expanded_matrix = np.zeros((32, 32))

#     # Fill each value into an 8x8 block
#     for i, value in enumerate(column_sums):
#         # Calculate the starting row and column for the current block
#         start_row = (i // 4) * 8
#         start_col = (i % 4) * 8
        
#         # Fill the 8x8 block with the current value
#         expanded_matrix[start_row:start_row+8, start_col:start_col+8] = value

#     return expanded_matrix

# def _return_variable_scores(df, attention_maps, global_case: str, local_idx :str):


#     if local_idx:
#         att = attention_maps[local_idx, ...]
#         att_means_flat = att.flatten()
#         # Get the indices that would sort the array in descending order
#         sorted_indices = np.argsort(att_means_flat)[::-1]
#         # Get the top ten values and their corresponding indices
#         top_ten_values = att_means_flat[sorted_indices][:10]
#         top_ten_indices = sorted_indices[:10]

#         print(f"{df.iloc[local_idx]['date_cycletime']}: vis = {df.iloc[local_idx]['vis']} | fog probability = {df.iloc[local_idx]['fog_prob']}")
#         # Print the top ten values and their indices
#         for i, (value, index) in enumerate(zip(top_ten_values, top_ten_indices)):
#             print(f"Rank {i+1}: Value = {value}, Index = {index}")

#     else: 
#         if global_case == 'Hit':
#             # Hits are cases where the true value is 1 and the predicted probability is greater than or equal to the threshold
#             cases         = df[(df['ytrue'] == 1) & (df['fog_prob'] >= 0.51)]
#             cases_indices = cases.index
#             sub_attention_maps = attention_maps[cases_indices, :]

#             sub_attention_maps = sub_attention_maps.mean(axis=0)
#             sub_attention_maps = sub_attention_maps.flatten()
            
#         elif global_case == 'Miss':
#             # Misses are cases where the true value is 1 but the predicted probability is less than the threshold
#             cases = df[(df['ytrue'] == 1) & (df['fog_prob'] < 0.51)]
#             cases_indices  = cases.index
#             sub_attention_maps = attention_maps[cases_indices, :]

#             sub_attention_maps = sub_attention_maps.mean(axis=0)
#             sub_attention_maps = sub_attention_maps.flatten()

#         elif global_case == 'FA':
#             # False Alarms are cases where the true value is 0 but the predicted probability is greater than or equal to the threshold
#             cases = df[(df['ytrue'] == 0) & (df['fog_prob'] >= 0.51)]
#             cases_indices = cases.index
#             sub_attention_maps = attention_maps[cases_indices, :]

#             sub_attention_maps = sub_attention_maps.mean(axis=0)
#             sub_attention_maps = sub_attention_maps.flatten()
#         elif global_case == 'CR':
#             # Corrected Rejection are cases where the true value is 0 and the predicted probability is less than the threshold
#             cases = df[(df['ytrue'] == 0) & (df['fog_prob'] < 0.51)]
#             cases_indices = cases.index
#             sub_attention_maps = attention_maps[cases_indices, :]

#             sub_attention_maps = sub_attention_maps.mean(axis=0)
#             sub_attention_maps = sub_attention_maps.flatten()

        
#         # Get the indices that would sort the array in descending order
#         sorted_indices = np.argsort(sub_attention_maps)[::-1]

#         # Get the top ten values and their corresponding indices
#         top_ten_values = sub_attention_maps[sorted_indices][:10]
#         top_ten_indices = sorted_indices[:10]

#         # Print the top ten values and their indices
#         for i, (value, index) in enumerate(zip(top_ten_values, top_ten_indices)):
#             time_series_number, var = _return_variable_name_by_index(index)
#             print(f"Rank {i+1}: Value = {value}, variable = {var}, time_step = {time_series_number}")
        
# def _return_variable_name_by_index(idx):

#     reduced_index = idx % 97
#     time_series_number = idx // 97

#     var = FogDataloader.var_idx[reduced_index]

#     return time_series_number, var

# def _return_mean_prob_cvs(exp_dir: str):

#     # train_pattern = '/data1/fog/SparkMET/EXPs/Nov11_Cv2_VVT_Final/*/train*.csv'
#     # train_prob_files = sorted(glob.glob(train_pattern, recursive= True))

#     test_pattern = exp_dir + '*/test_prob_last*.csv'
#     test_prob_files = sorted(glob.glob(test_pattern, recursive= True))

#     list_fog_prob_column =[]

#     for file in test_prob_files:
#         df = pd.read_csv(file)
#         list_fog_prob_column.append(df['fog_prob'])

#     #
#     sample_df = pd.read_csv(test_prob_files[0])
#     mean_df = pd.DataFrame()
#     mean_df['date_time']      = sample_df['date_time']
#     mean_df['round_time']     = sample_df['date_time']
#     mean_df['date_cycletime'] = sample_df['date_cycletime']
#     mean_df['vis']            = sample_df['vis']
#     mean_df['ytrue']          = sample_df['ytrue']
#     mean_df['fog_prob']       = np.mean(list_fog_prob_column, axis  = 0)

#     return mean_df

# ===============================================================
# valid_date_times, valid_round_times, valid_cycletimes, valid_visibilitys, valid_label_trues, valid_fog_preds, valid_nonfog_preds = [], [], [], [], [], [], []
# for batch, sample in enumerate(data_loader_validate):
    
#     valid_date_time  = sample['date_time']
#     valid_date_times.append(valid_date_time)

#     valid_round_time = sample['round_time']
#     valid_round_times.append(valid_round_time)

#     valid_cycletime  = sample['date_cycletime']
#     valid_cycletimes.append(valid_cycletime)

#     valid_visibility = sample['vis'] 
#     valid_visibilitys.append(valid_visibility)

#     valid_label_true = sample['label_class']
#     valid_label_trues.append(valid_label_true)

#     input_val            = sample['input'].to(device)
#     valid_attention_scores, pred_val = model(input_val)

#     # pred_val = torch.exp(pred_val)
#     softmax           = torch.nn.Softmax(dim=1)
#     pred_val          = softmax(pred_val)
#     pred_val = pred_val.detach().cpu().numpy()
#     valid_fog_preds.append(pred_val[:, 1])
#     valid_nonfog_preds.append(pred_val[:, 0])

# valid_date_times   = np.concatenate(valid_date_times)
# valid_round_times  = np.concatenate(valid_round_times)
# valid_cycletimes   = np.concatenate(valid_cycletimes)
# valid_visibilitys  = np.concatenate(valid_visibilitys)
# valid_label_trues  = np.concatenate(valid_label_trues)
# valid_fog_preds    = np.concatenate(valid_fog_preds)
# valid_nonfog_preds = np.concatenate(valid_nonfog_preds)

# valid_ypred = np.empty([valid_fog_preds.shape[0], 2], dtype = float)
# valid_ypred[:,1] = valid_fog_preds
# valid_ypred[:,0] = valid_nonfog_preds


# valid_output = pd.DataFrame()
# valid_output['date_time'] = valid_date_times
# valid_output['round_time'] = valid_round_times
# valid_output['date_cycletime'] = valid_cycletimes
# valid_output['vis'] = valid_visibilitys
# valid_output['ytrue'] = valid_label_trues
# valid_output['fog_prob'] = valid_fog_preds
# valid_output['nonfog_prob'] = valid_nonfog_preds

# valid_output.to_csv(valid_csv_file)

# valid_eval_obj = Evaluation2(valid_label_trues, valid_ypred, report_file_name = valid_report, figure_name = valid_roc_curve_fig)
# valid_evaluation_metrics = valid_eval_obj.confusion_matrix_calc()
# _ = valid_eval_obj.ruc_curve_plot()