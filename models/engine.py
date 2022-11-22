import numpy as np
import pandas as pd
import os 
import json
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.set(font_scale=1.5)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
from src import dataloader
from models import transformers, engine
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import math
from math import log



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

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    confusion_vector = y_pred_tag / y_test
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = torch.sum(confusion_vector == 1).item()
 
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = torch.sum(confusion_vector == float('inf')).item()
 
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = torch.sum(torch.isnan(confusion_vector)).item()

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = torch.sum(confusion_vector == 0).item()
    #print(f"{TP}|{FP}|{TN}|{FN}")
    if (((TP+FN)*(FN+TN))+((TP+FP)*(FP+TN))) == 0:
        HSS = torch.as_tensor(0, dtype = torch.float32)
    else:
        HSS  = torch.as_tensor((2*((TP*TN)-(FP*FN)))/(((TP+FN)*(FN+TN))+((TP+FP)*(FP+TN))), dtype = torch.float32)
    HSS = torch.round(HSS)

    return HSS

def save_loss_df(loss_stat, loss_df_name, loss_fig_name):

    df = pd.DataFrame.from_dict(loss_stat).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    df.to_csv(loss_df_name) 
    plt.figure(figsize=(12,8))
    sns.lineplot(data=df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.ylim(0, df['value'].max())
    plt.savefig(loss_fig_name, dpi = 300)

class Evaluation(): 
    def __init__(self, ytrue, ypred, category = None):
        self.ytrue     = ytrue
        self.ypred     = ypred
        self.category = category 


    def confusion_matrix_calc(self): 

        ypred_class = np.argmax(self.ypred, axis = 1)

        Hit, miss, FA, CR = confusion_matrix(self.ytrue, ypred_class).ravel()
        POD   = Hit/(Hit+miss)
        F     = FA/(FA+CR)
        FAR   = FA/(Hit+FA)
        CSI   = Hit/(Hit+FA+miss)
        PSS   = ((Hit*CR)-(FA*miss))/((FA+CR)*(Hit+miss))
        HSS   = (2*((Hit*CR)-(FA*miss)))/(((Hit+miss)*(miss+CR))+((Hit+FA)*(FA+CR)))
        ORSS  = ((Hit*CR)-(FA*miss))/((Hit*CR)+(FA*miss))
        CSS   = ((Hit*CR)-(FA*miss))/((Hit+FA)*(miss+CR))

        
        if POD == 1.0:
            POD = 0.999  
        if F <= 0:
            F = 0.009 
        
        SEDI = (log(F) - log(POD) - log(1-F) + log(1-POD))/(log(F) + log(POD) + log(1-F) + log(1-POD))

        output = [Hit, miss, FA, CR, POD, F, FAR, CSI, PSS, HSS, ORSS, CSS, SEDI]
    
        return output
        
    def ruc_curve_plot(self): 
        ypred_fog = self.ypred[:, -1] 
        fpr, tpr, thresholds = roc_curve(self.ytrue, ypred_fog, pos_label=2)

        ROC_AUC = auc(fpr, tpr)     
        
        plt.plot(fpr, tpr, linewidth=3, color = 'red')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.005, 1.0])
        plt.ylim([0.0, 1.005])
        plt.xlabel('FAR (probability of false detection)',  fontsize=20)
        plt.ylabel('POD (probability of detection)', fontsize=20)
        title_string = 'ROC curve (AUC = {0:.3f})'.format(ROC_AUC)
        plt.title(title_string, fontsize=20)

    def BS_BSS_calc(self): 

        ytrue_rev    = ytrue.copy()
        indices_one  = ytrue_rev == 1
        indices_zero = ytrue_rev == 0
        ytrue_rev[indices_one] = 0 # replacing 1s with 0s
        ytrue_rev[indices_zero] = 1 # replacing 0s with 1s
        
        
        P_c = np.mean(ytrue_rev)
        bs_init = 0
        bss_init = 0
        for e in range(len(ytrue_rev)):
            bss_init  = bs_init + (P_c - ytrue_rev[e])**2 # average value of fog accurence 
            
            if ytrue_rev[e] == 0:
                prob = ypred[e, 1]
                bs_init  = bs_init + (prob - 0)**2
                
            elif ytrue_rev[e] == 1:
                prob = ypred[e, 0]
                bs_init  = bs_init + (prob - 1)**2
                
        BS     = bs_init/len(ytrue_rev)
        BS_ref = bss_init/len(ytrue_rev)
        BSS    = (1-BS)/BS_ref 
        
        return BS, BSS



def eval_1d(start_date = None, 
                finish_date = None, 
                lead_time_pred = None, 
                predictor_names = None, 
                data_split_dict = None,  
                visibility_threshold = None, 
                point_geolocation_dic = None, 
                dropout               = 0.3,
                nhead                 = 8,
                dim_feedforward       = 512, 
                batch_size            = 64,
                Exp_name              = None,
):


    save_dir = './EXPs/' + Exp_name
    isExist  = os.path.isdir(save_dir)

    if not isExist:
        os.mkdir(save_dir)

    best_model_name      = save_dir + '/best_model' + Exp_name + '.pth'
    dict_name            = save_dir + '/mean_std_' + Exp_name + '.json' 




    # creating the entire data: 
    dataset = dataloader.input_dataframe_generater(img_path = None, 
                                                target_path = None, 
                                                first_date_string = start_date, 
                                                last_date_string = finish_date).dataframe_generation()
                                             
    # split the data into train, validation and test:
    train_df, valid_df, test_df = dataloader.split_data_train_valid_test(dataset, year_split_dict = data_split_dict)
    # calculating the mean and std of training variables: 
    isDictExists = os.path.isfile(dict_name)
    if not isDictExists:
        norm_mean_std_dict = dataloader.return_train_variable_mean_std(train_df, 
                                                    predictor_names = predictor_names, 
                                                    lead_time_pred = lead_time_pred).return_mean_std_dict()
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

    test_dataset = dataloader.DataAdopter(test_df, 
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
    data_loader_testing = torch.utils.data.DataLoader(test_dataset, batch_size= batch_size, 
                                                    shuffle=False,  num_workers=8)

    


    model = transformers.Transformer1d(
                                d_model        = dataset.shape[1], 
                                nhead          = nhead, 
                                dim_feedforward= dim_feedforward, 
                                n_classes      = 2, 
                                dropout        = dropout, 
                                activation     = 'relu',
                                verbose        = True).to(device)


    model.load_state_dict(torch.load(best_model_name))

    #==============================================================================================================#
    #==============================================================================================================#
    with torch.no_grad():
        model.eval()
        train_date_times, train_round_times, train_cycletimes, train_visibilitys, train_label_trues, train_label_preds = [], [], [], [], [], []
        for batch_idx, sample in enumerate(data_loader_training):

            train_date_time  = sample['date_time']
            train_date_times.append(train_date_time)

            train_round_time = sample['round_time']
            train_round_times.append(train_round_time)

            train_cycletime  = sample['date_cycletime']
            train_cycletimes.append(train_cycletime)

            train_visibility = sample['vis'] 
            train_visibilitys.append(train_visibility)

            train_label_true = sample['label_class']
            train_label_trues.append(train_label_true)

            input_train      = sample['input'].to(device)
            _, train_out, _ = model(input_train)
            train_label_preds.append(train_out)

        train_date_times = np.concatenate(train_date_times)
        train_round_times = np.concatenate(train_round_times)
        train_cycletimes = np.concatenate(train_cycletimes)
        train_visibilitys = np.concatenate(train_visibilitys)
        train_label_trues = np.concatenate(train_label_trues)
        train_label_preds = np.concatenate(train_label_preds)

        train_output = pd.DataFrame()
        train_output['date_time'] = train_date_times
        train_output['round_time'] = train_round_times
        train_output['date_cycletime'] = train_cycletimes
        train_output['vis'] = train_visibilitys
        train_output['ytrue'] = train_label_trues
        train_output['ypred'] = train_label_preds

        train_output.to_csv()

        train_eval_obj = Evaluation(train_label_trues, train_label_preds, category = 'train')
        train_evaluation_metrics = train_eval_obj.confusion_matrix_calc()
        train_plot_roc_curve     = train_eval_obj.ruc_curve_plot()



        valid_date_times, valid_round_times, valid_cycletimes, valid_visibilitys, valid_label_trues, valid_label_preds = [], [], [], [], [], []
        for batch, sample in enumerate(data_loader_validate):
            
            valid_date_time  = sample['date_time']
            valid_date_times.append(valid_date_time)

            valid_round_time = sample['round_time']
            valid_round_times.append(valid_round_time)

            valid_cycletime  = sample['date_cycletime']
            valid_cycletimes.append(valid_cycletime)

            valid_visibility = sample['vis'] 
            valid_visibilitys.append(valid_visibility)

            valid_label_true = sample['label_class']
            valid_label_trues.append(valid_label_true)

            input_val      = sample['input'].to(device)
            _, pred_val, _ = model(input_val)
            valid_label_preds.append(pred_val)


        valid_date_times  = np.concatenate(valid_date_times)
        valid_round_times = np.concatenate(valid_round_times)
        valid_cycletimes  = np.concatenate(valid_cycletimes)
        valid_visibilitys = np.concatenate(valid_visibilitys)
        valid_label_trues = np.concatenate(valid_label_trues)
        valid_label_preds = np.concatenate(valid_label_preds)

        valid_output = pd.DataFrame()
        valid_output['date_time'] = valid_date_times
        valid_output['round_time'] = valid_round_times
        valid_output['date_cycletime'] = valid_cycletimes
        valid_output['vis'] = valid_visibilitys
        valid_output['ytrue'] = valid_label_trues
        valid_output['ypred'] = valid_label_preds

        valid_output.to_csv()
        valid_eval_obj = Evaluation(valid_label_trues, valid_label_preds, category = 'valid')
        valid_evaluation_metrics = valid_eval_obj.confusion_matrix_calc()
        valid_plot_roc_curve     = valid_eval_obj.ruc_curve_plot()

        test_date_times, test_round_times,test_cycletimes, test_visibilitys, test_label_trues, test_label_preds = [], [], [], [], [], []
        for batch, sample in enumerate(data_loader_testing):

            test_date_time  = sample['date_time']
            test_date_times.append(test_date_time)

            test_round_time = sample['round_time']
            test_round_times.append(test_round_time)

            test_cycletime  = sample['date_cycletime']
            test_cycletimes.append(test_cycletime)

            test_visibility = sample['vis'] 
            test_visibilitys.append(test_visibility)

            test_label_true = sample['label_class']
            test_label_trues.append(test_label_true)
            


            input_test      = sample['input'].to(device)
            _, pred_test, _ = model(input_test)
            test_label_preds.append(pred_test)


        test_date_times  = np.concatenate(test_date_times)
        test_round_times = np.concatenate(test_round_times)
        test_cycletimes  = np.concatenate(test_cycletimes)
        test_visibilitys = np.concatenate(test_visibilitys)
        test_label_trues = np.concatenate(test_label_trues)
        test_label_preds = np.concatenate(test_label_preds)

        test_output = pd.DataFrame()
        test_output['date_time'] = test_date_times
        test_output['round_time'] = test_round_times
        test_output['date_cycletime'] = test_cycletimes
        test_output['vis'] = test_visibilitys
        test_output['ytrue'] = test_label_trues
        test_output['ypred'] = test_label_preds

        test_output.to_csv()
        test_eval_obj = Evaluation(test_label_trues, test_label_preds, category = 'valid')
        test_evaluation_metrics = test_eval_obj.confusion_matrix_calc()
        test_plot_roc_curve     = test_eval_obj.ruc_curve_plot()





