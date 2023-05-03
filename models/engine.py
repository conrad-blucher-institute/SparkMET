import numpy as np
import pandas as pd
import torch 
device ="cuda" if torch.cuda.is_available() else "cpu"
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import math
from math import log
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib.patches import Circle





def print_report(train_df, valid_df, test_df):
    train_fog_cases = train_df['vis_category'].value_counts()['fog']
    valid_fog_cases = valid_df['vis_category'].value_counts()['fog']
    test_fog_cases = test_df['vis_category'].value_counts()['fog']
    print("#================================ Summary of Dataset ==================#")
    print(f"number of training samples:   {train_df.shape[0]} | number of training fog cases:   {train_fog_cases}")
    print(f"number of validation samples: {valid_df.shape[0]} | number of validation fog cases: {valid_fog_cases}")
    print(f"number of test samples:       {test_df.shape[0]} | number of test fog cases:       {test_fog_cases}")
    print("#======================================================================#")

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
    def __init__(self, ytrue, ypred, report_file_name = None, figure_name = None):
        self.ytrue     = ytrue
        self.ypred     = ypred
        self.report_file_name  = report_file_name 
        self.figure_name = figure_name


    def confusion_matrix_calc(self): 

        ypred_class = np.argmax(self.ypred, axis = 1)

        CR, FA, miss, Hit= confusion_matrix(self.ytrue, ypred_class).ravel()
        POD   = Hit/(Hit+miss)
        F     = FA/(FA+CR)
        FAR   = FA/(Hit+FA)
        CSI   = Hit/(Hit+FA+miss)
        PSS   = ((Hit*CR)-(FA*miss))/((FA+CR)*(Hit+miss))
        HSS   = (2*((Hit*CR)-(FA*miss)))/(((Hit+miss)*(miss+CR))+((Hit+FA)*(FA+CR)))
        ORSS  = ((Hit*CR)-(FA*miss))/((Hit*CR)+(FA*miss))
        CSS   = ((Hit*CR)-(FA*miss))/((Hit+FA)*(miss+CR))

        #SEDI = (log(F) - log(POD) - log(1-F) + log(1-POD))/(log(F) + log(POD) + log(1-F) + log(1-POD))

        output = [Hit, miss, FA, CR, POD, F, FAR, CSI, PSS, HSS, ORSS, CSS]

        with open(self.report_file_name, "a") as f: 
            # for m in output:
            #     name = str(m) 
            #output= [
                #'output = ['CSS', 'PSS' …]
                #for m in output:
                #print f'm: {eval{m)}', 
            #     print(f"{name}: {m}", file=f)
            print(f"Hit: {Hit}", file=f)
            print(f"Miss: {miss}", file=f)
            print(f"FA: {FA}", file=f)
            print(f"CR: {CR}", file=f)
            print(f"POD: {POD}", file=f)
            print(f"F: {F}", file=f)
            print(f"FAR: {FAR}", file=f)
            print(f"CSI: {CSI}", file=f)
            print(f"PSS: {PSS}", file=f)
            print(f"HSS: {HSS}", file=f)
            print(f"ORSS: {ORSS}", file=f)
            print(f"CSS: {CSS}", file=f)
            #print(f"SEDI: {SEDI}", file=f)

        return output
        
    def ruc_curve_plot(self): 
        ypred_fog = self.ypred[:, 1] 
        fpr, tpr, thresholds = roc_curve(self.ytrue, ypred_fog)

        ROC_AUC = auc(fpr, tpr)     
        fig, ax = plt.subplots(figsize = (8, 6))
        
        ax.plot(fpr, tpr, linewidth=3, color = 'red')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([-0.005, 1.0])
        ax.set_ylim([0.0, 1.005])
        ax.set_xlabel('FAR (probability of false detection)',  fontsize=16)
        ax.set_ylabel('POD (probability of detection)', fontsize=16)
        title_string = 'ROC curve (AUC = {0:.3f})'.format(ROC_AUC)
        ax.set_title(title_string, fontsize=16)
        plt.savefig(self.figure_name, dpi = 300)

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

def predict(model, data_loader_training, data_loader_validate, data_loader_testing, Exp_name = None,):


    save_dir = '/data1/fog/SparkMET/EXPs/' + Exp_name
    best_model_name      = save_dir + '/best_model_' + Exp_name + '.pth'

    train_csv_file = save_dir + '/train_prob_' + Exp_name + '.csv'
    valid_csv_file = save_dir + '/valid_prob_' + Exp_name + '.csv'
    test_csv_file  = save_dir + '/test_prob_' + Exp_name + '.csv'

    train_roc_curve_fig = save_dir + '/train_roc_' + Exp_name + '.png'
    valid_roc_curve_fig = save_dir + '/valid_roc_' + Exp_name + '.png'
    test_roc_curve_fig  = save_dir + '/test_roc_' + Exp_name + '.png'

    train_report = save_dir + '/train_report_' + Exp_name + '.txt'
    valid_report = save_dir + '/valid_report_' + Exp_name + '.txt'
    test_report  = save_dir + '/test_report_' + Exp_name + '.txt'

    model.load_state_dict(torch.load(best_model_name))

    #==============================================================================================================#
    #==============================================================================================================#
    with torch.no_grad():
        model.eval()
        train_date_times, train_round_times, train_cycletimes, train_visibilitys, train_label_trues, train_fog_preds, train_nonfog_preds= [], [], [], [], [], [], []
        train_attention_outputs, valid_attention_outputs, test_attention_outputs = [], [], []
        train_attention_map_outputs, valid_attention_map_outputs, test_attention_map_outputs = [], [], []
        train_inputs, valid_inputs, test_inputs = [], [], []
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

            input_train      = sample['input'].to(0)

            train_attention_scores, train_out = model(input_train)
            #train_attention_scores, train_out = model(input_train)

            #============================================================================================
            # Output
            train_out = torch.exp(train_out)
            train_out = train_out.detach().cpu().numpy()
            train_fog_preds.append(train_out[:, 1])
            train_nonfog_preds.append(train_out[:, 0])

            # Attention scores: 
            layer_attention_outputs = None
            for layer_output in train_attention_scores:
                layer_output = np.expand_dims(layer_output.detach().cpu().numpy(), axis  = -1)
                layer_output = layer_output[:, :, 1:, 1:, :]
                if layer_attention_outputs is None: 
                    layer_attention_outputs = layer_output
                else: 
                    layer_attention_outputs = np.concatenate((layer_attention_outputs, layer_output), axis = -1)
            train_attention_outputs.append(layer_attention_outputs) 
            



            # inputs
            train_inputs.append(input_train.detach().cpu().numpy())

        train_attention_outputs  = np.concatenate(train_attention_outputs, axis = 0)
        train_attention_output_maps, train_attention_output_scores  = attention_map_vis(train_attention_outputs).return_all_attention_maps()

        train_inputs             = np.concatenate(train_inputs, axis = 0)


        
        train_date_times   = np.concatenate(train_date_times)
        train_round_times  = np.concatenate(train_round_times)
        train_cycletimes   = np.concatenate(train_cycletimes)
        train_visibilitys  = np.concatenate(train_visibilitys)
        train_label_trues  = np.concatenate(train_label_trues)
        train_fog_preds    = np.concatenate(train_fog_preds)
        train_nonfog_preds = np.concatenate(train_nonfog_preds)

        train_ypred = np.empty([train_fog_preds.shape[0], 2], dtype = float)
        train_ypred[:,1] = train_fog_preds
        train_ypred[:,0] = train_nonfog_preds

        train_output = pd.DataFrame()
        train_output['date_time'] = train_date_times
        train_output['round_time'] = train_round_times
        train_output['date_cycletime'] = train_cycletimes
        train_output['vis'] = train_visibilitys
        train_output['ytrue'] = train_label_trues
        train_output['fog_prob'] = train_fog_preds
        train_output['nonfog_prob'] = train_nonfog_preds

        train_output.to_csv(train_csv_file)

        train_eval_obj = Evaluation(train_label_trues, train_ypred, report_file_name = train_report, figure_name = train_roc_curve_fig)
        train_evaluation_metrics = train_eval_obj.confusion_matrix_calc()
        _ = train_eval_obj.ruc_curve_plot()

        #===============================================================
        valid_date_times, valid_round_times, valid_cycletimes, valid_visibilitys, valid_label_trues, valid_fog_preds, valid_nonfog_preds = [], [], [], [], [], [], []
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

            input_val            = sample['input'].to(0)
            valid_attention_scores, pred_val = model(input_val)
            #valid_attention_scores, pred_val = model(input_val)
            #m = nn.Softmax(dim=1)
            #pred_val = m(logits)
            pred_val = torch.exp(pred_val)

            pred_val = pred_val.detach().cpu().numpy()
            valid_fog_preds.append(pred_val[:, 1])
            valid_nonfog_preds.append(pred_val[:, 0])

            # Attention scores: 
            layer_attention_outputs_v = None
            for layer_output_v in valid_attention_scores:
                layer_output_v = np.expand_dims(layer_output_v.detach().cpu().numpy(), axis  = -1)
                layer_output_v = layer_output_v[:, :, 1:, 1:, :]
                if layer_attention_outputs_v is None: 
                    layer_attention_outputs_v = layer_output_v
                else: 
                    layer_attention_outputs_v = np.concatenate((layer_attention_outputs_v, layer_output_v), axis = -1)

            valid_attention_outputs.append(layer_attention_outputs_v)   
            valid_inputs.append(input_val.detach().cpu().numpy())



        valid_attention_outputs  = np.concatenate(valid_attention_outputs, axis = 0)
        valid_attention_output_maps, valid_attention_output_scores = attention_map_vis(valid_attention_outputs).return_all_attention_maps()
        valid_inputs = np.concatenate(valid_inputs, axis = 0)




        valid_date_times   = np.concatenate(valid_date_times)
        valid_round_times  = np.concatenate(valid_round_times)
        valid_cycletimes   = np.concatenate(valid_cycletimes)
        valid_visibilitys  = np.concatenate(valid_visibilitys)
        valid_label_trues  = np.concatenate(valid_label_trues)
        valid_fog_preds    = np.concatenate(valid_fog_preds)
        valid_nonfog_preds = np.concatenate(valid_nonfog_preds)

        valid_ypred = np.empty([valid_fog_preds.shape[0], 2], dtype = float)
        valid_ypred[:,1] = valid_fog_preds
        valid_ypred[:,0] = valid_nonfog_preds


        valid_output = pd.DataFrame()
        valid_output['date_time'] = valid_date_times
        valid_output['round_time'] = valid_round_times
        valid_output['date_cycletime'] = valid_cycletimes
        valid_output['vis'] = valid_visibilitys
        valid_output['ytrue'] = valid_label_trues
        valid_output['fog_prob'] = valid_fog_preds
        valid_output['nonfog_prob'] = valid_nonfog_preds

        valid_output.to_csv(valid_csv_file)

        valid_eval_obj = Evaluation(valid_label_trues, valid_ypred, report_file_name = valid_report, figure_name = valid_roc_curve_fig)
        valid_evaluation_metrics = valid_eval_obj.confusion_matrix_calc()
        _ = valid_eval_obj.ruc_curve_plot()

        test_date_times, test_round_times,test_cycletimes, test_visibilitys, test_label_trues, test_fog_preds, test_nonfog_preds = [], [], [], [], [], [], []
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

            input_test      = sample['input'].to(0)
            test_attention_scores, pred_test = model(input_test)
            #test_attention_scores, pred_test = model(input_test)

            pred_test = torch.exp(pred_test)
            pred_test = pred_test.detach().cpu().numpy()
            test_fog_preds.append(pred_test[:, 1])
            test_nonfog_preds.append(pred_test[:, 0])

            # Attention scores: 
            layer_attention_outputs_test = None
            for layer_output_test in test_attention_scores:
                layer_output_test = np.expand_dims(layer_output_test.detach().cpu().numpy(), axis  = -1)
                layer_output_test = layer_output_test[:, :, 1:, 1:, :]
                if layer_attention_outputs_test is None: 
                    layer_attention_outputs_test = layer_output_test
                else: 
                    layer_attention_outputs_test = np.concatenate((layer_attention_outputs_test, layer_output_test), axis = -1)

            test_attention_outputs.append(layer_attention_outputs_test)    
            test_inputs.append(input_test.detach().cpu().numpy())

        test_inputs            = np.concatenate(test_inputs, axis = 0)

        test_attention_outputs = np.concatenate(test_attention_outputs, axis = 0)
        test_attention_output_maps, test_attention_output_scores = attention_map_vis(test_attention_outputs).return_all_attention_maps()

        test_date_times   = np.concatenate(test_date_times)
        test_round_times  = np.concatenate(test_round_times)
        test_cycletimes   = np.concatenate(test_cycletimes)
        test_visibilitys  = np.concatenate(test_visibilitys)
        test_label_trues  = np.concatenate(test_label_trues)
        test_fog_preds    = np.concatenate(test_fog_preds)
        test_nonfog_preds = np.concatenate(test_nonfog_preds)

        test_ypred = np.empty([test_fog_preds.shape[0], 2], dtype = float)
        test_ypred[:,1] = test_fog_preds
        test_ypred[:,0] = test_nonfog_preds

        test_output = pd.DataFrame()
        test_output['date_time'] = test_date_times
        test_output['round_time'] = test_round_times
        test_output['date_cycletime'] = test_cycletimes
        test_output['vis'] = test_visibilitys
        test_output['ytrue'] = test_label_trues
        test_output['fog_preb'] = test_fog_preds
        test_output['nonfog_preb'] = test_nonfog_preds

        test_output.to_csv(test_csv_file)

        test_eval_obj = Evaluation(test_label_trues, test_ypred, report_file_name = test_report, figure_name = test_roc_curve_fig)
        test_evaluation_metrics = test_eval_obj.confusion_matrix_calc()
        _ = test_eval_obj.ruc_curve_plot()

    predictions = [train_output, valid_output, test_output]
    inputs      = [train_inputs, valid_inputs, test_inputs]
    raw_attention_scores = [train_attention_outputs, valid_attention_outputs, test_attention_outputs]
    att_maps    = [train_attention_output_maps, valid_attention_output_maps, test_attention_output_maps]
    att_scores  = [train_attention_output_scores, valid_attention_output_scores, test_attention_output_scores]

    return predictions, inputs, raw_attention_scores, att_maps, att_scores #

def train(model, optimizer, loss_func, training_config_dict, data_loader_training, data_loader_validate, Exp_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = '/data1/fog/SparkMET/EXPs/' + Exp_name
 
    best_model_name      = save_dir + '/best_model_' + Exp_name + '.pth'
    loss_fig_name        = save_dir + '/loss_' + Exp_name + '.png'
    loss_df_name         = save_dir + '/loss_' + Exp_name + '.csv' 

    loss_stats = {'train': [],"val": []}

    best_val_loss = 100000 # initial dummy value
    early_stopping = EarlyStopping(tolerance = training_config_dict['early_stop_tolerance'], min_delta=50)
    step  = 0
    #==============================================================================================================#
    #==============================================================================================================#
    epochs = training_config_dict['epochs']
    
    for epoch in range(1, epochs+1):
        
        training_start_time = time.time()
        # TRAINING
        train_epoch_loss = 0
        model.train()

        for batch_idx, sample in enumerate(data_loader_training):

            input_train        = sample['input'].to(device) #.type(torch.LongTensor)
            train_label_true   = sample['label_class'].to(device)


            train_attention_scores, train_out = model(input_train)
            optimizer.zero_grad()
            
            train_loss  = loss_func(train_out, train_label_true) #
            
            train_loss.backward()
            optimizer.step()
            step += 1

            train_epoch_loss += train_loss.item()

        with torch.no_grad():
            
            val_epoch_loss = 0
            #val_epoch_acc = 0
            model.eval()
            for batch, sample in enumerate(data_loader_validate):
                
                input_val      = sample['input'].to(device) #.type(torch.LongTensor)
                label_true_val = sample['label_class'].to(device)

                valid_attention_scores, pred_val = model(input_val)
            
                val_loss       = loss_func(pred_val, label_true_val)
                val_epoch_loss += val_loss.item()
        
                
        training_duration_time = (time.time() - training_start_time)

        loss_stats['train'].append(train_epoch_loss/len(data_loader_training))
        loss_stats['val'].append(val_epoch_loss/len(data_loader_validate))
        print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss/len(data_loader_training):.4f} | Val Loss: {val_epoch_loss/len(data_loader_validate):.4f} | Time(s): {training_duration_time:.3f}') 
        #   | Train HSS: {train_epoch_acc/len(data_loader_training):.4f} | Val HSS: {val_epoch_acc/len(data_loader_validate):.4f}
        if (val_epoch_loss/len(data_loader_validate)) < best_val_loss or epoch==0:
                    
            best_val_loss=(val_epoch_loss/len(data_loader_validate))
            torch.save(model.state_dict(), best_model_name)
            
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
    _ = save_loss_df(loss_stats, loss_df_name, loss_fig_name)

    return model, loss_stats

def extract_selfattention_maps(transformer_encoder,x, mask, src_key_padding_mask):
    attention_maps = []
    num_layers = transformer_encoder.num_layers
    num_heads = transformer_encoder.layers[0].self_attn.num_heads
    norm_first = transformer_encoder.layers[0].norm_first
    with torch.no_grad():
        for i in range(num_layers):
            # compute attention of layer i
            h = x.clone()
            if norm_first:
                h = transformer_encoder.layers[i].norm1(h)
            attn = transformer_encoder.layers[i].self_attn(h, h, h,attn_mask=mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
            attention_maps.append(attn)
            # forward of layer i
            x = transformer_encoder.layers[i](x,src_mask=mask,src_key_padding_mask=src_key_padding_mask)
    return attention_maps

class attention_map_vis(): 
    def __init__(self, att_maps):
        self.att_maps = att_maps

    def return_atten_guass(self, maps):
        heads_layers_output_scores= None

        for head in range(maps.shape[0]):
            layers_output_scores = None
            for layer in range(maps.shape[3]): 
                scores = maps[head, :, 5, layer]

                
                max_idx    = np.argmax(scores)
                max_scores = np.max(scores) 
                max_50th   = np.percentile(scores, 50)
                max_90th   = np.percentile(scores, 90)

                output_map = np.array((max_idx, max_scores, max_50th, max_90th))
                output_map = np.expand_dims(output_map, axis = -1)

                if layers_output_scores is None: 
                    layers_output_scores = output_map
                else: 
                    layers_output_scores = np.concatenate((layers_output_scores, output_map), axis = -1) 

            layers_output_scores = np.expand_dims(layers_output_scores, axis = 0)


            if heads_layers_output_scores is None: 
                heads_layers_output_scores = layers_output_scores
            else: 
                heads_layers_output_scores = np.concatenate((heads_layers_output_scores, layers_output_scores), axis = 0) 

        return heads_layers_output_scores
    
    def return_atten_map(self, maps):
        heads_layers_output_maps = None


        for head in range(maps.shape[0]):
            layers_output_maps = None
            for layer in range(maps.shape[3]): 
                scores = maps[head, :, :, layer]

                scores = np.sum(scores, axis = 1)

                if len(scores) == 16: 
                    output_map = self.convert_scores_to_maps(scores)
                    output_map = np.expand_dims(output_map, axis = -1)

                    if layers_output_maps is None: 
                        layers_output_maps = output_map
                    else: 
                        layers_output_maps = np.concatenate((output_map, layers_output_maps), axis = -1) 

                else: 
                    #layers_output = [i[0] for i in sorted(enumerate(scores), key=lambda k: k[1], reverse=True)]
                    layers_output = np.expand_dims(scores, axis = -1)
                    if layers_output_maps is None: 
                        layers_output_maps = layers_output
                    else: 
                        layers_output_maps = np.concatenate((layers_output_maps, layers_output), axis = -1) 


            layers_output_maps = np.expand_dims(layers_output_maps, axis = 0)

            if heads_layers_output_maps is None: 
                heads_layers_output_maps = layers_output_maps
            else: 
                heads_layers_output_maps = np.concatenate((heads_layers_output_maps, layers_output_maps), axis = 0) 

        return heads_layers_output_maps


    def convert_scores_to_maps(self, scores):
        out     = np.ones((32, 32),dtype = float)
        scores  = scores.reshape(4,4) 

        for i in range(4):
            for j in range(4): 
                value = scores[i, j]
                scor_map = np.full(64, value)
                scor_map = scor_map.reshape(8, 8)
                out[i*8:(i+1)*8, j*8:(j+1)*8] = scor_map
        return out
    

    def return_all_attention_maps(self): 
        output_maps, output_scores  = None, None
        for m in range(self.att_maps.shape[0]):

            heads_layers_output_maps = self.return_atten_map(self.att_maps[m, :, :, :, :])
            heads_layers_output_maps = np.expand_dims(heads_layers_output_maps, axis = 0)
            if output_maps is None: 
                output_maps = heads_layers_output_maps
            else: 
                output_maps = np.concatenate((output_maps, heads_layers_output_maps), axis = 0)

            heads_layers_output_scores = self.return_atten_guass(self.att_maps[m, :, :, :, :])
            heads_layers_output_scores = np.expand_dims(heads_layers_output_scores, axis = 0)
            if output_scores is None: 
                output_scores = heads_layers_output_scores
            else: 
                output_scores = np.concatenate((output_scores, heads_layers_output_scores), axis = 0)


        return output_maps, output_scores

def attention_map_visualize(df, inputs, attention_maps, variable:int, date:str): 

    idx = df[df['date_cycletime'] == date].index[0]
    num_heads  = attention_maps.shape[1]
    num_layers = attention_maps.shape[4]

    fig, axs = plt.subplots(4, 8, figsize = (24, 12))

    img1 = inputs[idx, variable, ::-1, :]
    min_ = np.min(attention_maps[idx, :, :, :, :])

    for i in range(num_layers):
        for j in range(num_heads):
            img2 = attention_maps[idx, j, ::-1, :, i]
            vis1 = axs[i, j].imshow(img1, cmap = 'gray')
            vis2 = axs[i, j].imshow(img2, alpha=0.4)
            
            vis2.set_clim(0.0589, 0.062)

            if j == num_heads -1: 
                divider = make_axes_locatable(axs[i, j])
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar1 = fig.colorbar(vis2, cax=cax)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    for i in range(num_layers): 
        axs[i, 0].set_ylabel(r'Layer {0}'.format(i+1))
    for j in range(num_heads):
        axs[0, j].set_title(r'Head {0}'.format(j+1))

    
    plt.show()

def attention_map_guassian_visualize(df, inputs, attention_scores, variable:int, date:str): 

    idx = df[df['date_cycletime'] == date].index[0]
    num_heads  = attention_scores.shape[1]
    num_layers = attention_scores.shape[3]

    fig, axs = plt.subplots(1, num_layers, figsize = (24, 6))

    img1 = inputs[idx, variable, ::-1, :]
    # Create a Rectangle patch
    
    colors = ['c', 'm', 'y', 'k', 'r', 'b', 'g', 'brown']
    for i in range(num_layers):
        vis1 = axs[i].imshow(img1, cmap = 'gray',alpha=0.4)
        rect = patches.Rectangle((7, 7), 2, 2, linewidth=1, edgecolor='r', facecolor = 'r')
        # Add the patch to the Axes
        axs[i].add_patch(rect)
        for v in range(num_heads): 
            list_ = attention_scores[idx, v, :, i]

            pixel_idx = list_[0]
            #print(f"Layer: {i}| Head: {v} = {pixel_idx}")
 
            if pixel_idx == 0: 
                coordinates = (4, 4)
            elif pixel_idx == 1: 
                coordinates = (14, 6)
            elif pixel_idx == 2: 
                coordinates = (22, 6)
            elif pixel_idx == 3: 
                coordinates = (30, 6)
            elif pixel_idx == 4: 
                coordinates = (6, 14)
            elif pixel_idx == 5: 
                coordinates = (14, 14)
            elif pixel_idx == 6: 
                coordinates = (22, 14)
            elif pixel_idx == 7: 
                coordinates = (30, 14)
            elif pixel_idx == 8: 
                coordinates = (6, 22)
            elif pixel_idx == 9: 
                coordinates = (14, 22)
            elif pixel_idx == 10: 
                coordinates = (22, 22)
            elif pixel_idx == 11: 
                coordinates = (30, 22)
            elif pixel_idx == 12: 
                coordinates = (6, 30)
            elif pixel_idx == 13: 
                coordinates = (14, 30)
            elif pixel_idx == 14: 
                coordinates = (22, 30)
            elif pixel_idx == 15: 
                coordinates = (30, 30)

            Perc50 = list_[2]
            Perc90 = list_[3]

            HeadName = r'Head {0}'.format(v)
            circ50 = Circle(coordinates, radius= Perc50*10, color = colors[v], fill = False, linewidth = 2, label = HeadName)
            axs[i].add_patch(circ50)
            circ90 = Circle(coordinates, radius= Perc90*20, color = colors[v],fill = False, linestyle = '--', linewidth = 2)
            axs[i].add_patch(circ90)

        axs[i].set_xticks([])
        axs[i].set_yticks([])
    axs[3].legend()

    for l in range(num_layers): 
        axs[l].set_title(r'Layer {0}'.format(l+1))

    
    plt.show()