import numpy as np
import pandas as pd
import torch 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time 
#from torch.cuda.amp import autocast, GradScaler

device ="cuda" if torch.cuda.is_available() else "cpu"

def predict(model, data_loader_training, data_loader_validate, data_loader_testing, SaveDir = None, Exp_name = None,):

    #'/data1/fog/SparkMET/EXPs/'
    save_dir =  SaveDir + Exp_name
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

            train_attention_scores, train_out = model(input_train)
            #============================================================================================
            # Output
            train_out          = torch.exp(train_out)
            train_out          = train_out.detach().cpu().numpy()
            train_fog_preds.append(train_out[:, 1])
            train_nonfog_preds.append(train_out[:, 0])

            # Attention scores: 
        #     layer_attention_outputs = None
        #     for layer_output in train_attention_scores[1]:
        #         layer_output = np.expand_dims(layer_output.detach().cpu().numpy(), axis  = -1)
        #         layer_output = layer_output[:, :, 1:, 1:, :]
        #         if layer_attention_outputs is None: 
        #             layer_attention_outputs = layer_output
        #         else: 
        #             layer_attention_outputs = np.concatenate((layer_attention_outputs, layer_output), axis = -1)
        #     train_attention_outputs.append(layer_attention_outputs) 
            
        # train_attention_outputs  = np.concatenate(train_attention_outputs, axis = 0)

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

            pred_val = torch.exp(pred_val)

            pred_val = pred_val.detach().cpu().numpy()
            valid_fog_preds.append(pred_val[:, 1])
            valid_nonfog_preds.append(pred_val[:, 0])

            # Attention scores: 
        #     layer_attention_outputs_v = None
        #     for layer_output_v in valid_attention_scores[1]:
        #         layer_output_v = np.expand_dims(layer_output_v.detach().cpu().numpy(), axis  = -1)
        #         layer_output_v = layer_output_v[:, :, 1:, 1:, :]
        #         if layer_attention_outputs_v is None: 
        #             layer_attention_outputs_v = layer_output_v
        #         else: 
        #             layer_attention_outputs_v = np.concatenate((layer_attention_outputs_v, layer_output_v), axis = -1)

        #     valid_attention_outputs.append(layer_attention_outputs_v)   

        # valid_attention_outputs  = np.concatenate(valid_attention_outputs, axis = 0)

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


            pred_test = torch.exp(pred_test)
            pred_test = pred_test.detach().cpu().numpy()
            test_fog_preds.append(pred_test[:, 1])
            test_nonfog_preds.append(pred_test[:, 0])

            # Attention scores: 
            # layer_attention_outputs_test = None
            # for layer_output_test in test_attention_scores[1]:
            #     layer_output_test = np.expand_dims(layer_output_test.detach().cpu().numpy(), axis  = -1)
            #     layer_output_test = layer_output_test[:, :, 1:, 1:, :]
            #     if layer_attention_outputs_test is None: 
            #         layer_attention_outputs_test = layer_output_test
            #     else: 
            #         layer_attention_outputs_test = np.concatenate((layer_attention_outputs_test, layer_output_test), axis = -1)

            # test_attention_outputs.append(layer_attention_outputs_test)    

        #test_attention_outputs = np.concatenate(test_attention_outputs, axis = 0)


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
    #raw_attention_scores = [train_attention_outputs, valid_attention_outputs, test_attention_outputs]

    return predictions, #raw_attention_scores

def train(model, optimizer, loss_func, data_loader_training, data_loader_validate, 
          epochs=100, early_stop_tolerance= 50, SaveDir = None, Exp_name = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = SaveDir + Exp_name
 
    best_model_name      = save_dir + '/best_model_' + Exp_name + '.pth'
    loss_fig_name        = save_dir + '/loss_' + Exp_name + '.png'
    loss_df_name         = save_dir + '/loss_' + Exp_name + '.csv' 

    loss_stats = {'train': [],"val": []}

    best_val_loss  = 100000 # initial dummy value
    early_stopping = EarlyStopping(tolerance = early_stop_tolerance, min_delta=50)
    step           = 0
    #==============================================================================================================#
    #==============================================================================================================#

    #scaler = GradScaler()
    for epoch in range(1, epochs+1):
        
        training_start_time = time.time()
        # TRAINING
        train_epoch_loss = 0
        model.train()

        for batch_idx, sample in enumerate(data_loader_training):

            input_train        = sample['input'].to(device) #.type(torch.LongTensor)
            train_label_true   = sample['label_class'].to(device)
            optimizer.zero_grad()

            #with torch.autocast(device_type = device, dtype = torch.float16):
            _ , train_out = model(input_train)
            
            train_loss  = loss_func(train_out, train_label_true) #


            #scaler.scale(train_loss).backward()
            train_loss.backward()
            #scaler.step(optimizer) #
            optimizer.step()
            #scaler.update()
            step += 1
            train_epoch_loss += train_loss.item()

        with torch.no_grad():
            
            val_epoch_loss = 0
            #val_epoch_acc = 0
            model.eval()
            for batch, sample in enumerate(data_loader_validate):
                
                input_val      = sample['input'].to(device) #.type(torch.LongTensor)
                label_true_val = sample['label_class'].to(device)
                #with torch.autocast(device_type = device, dtype = torch.float16):
                _ , pred_val = model(input_val)
            
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



