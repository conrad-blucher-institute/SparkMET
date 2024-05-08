import torch
import numpy as np
import random
import argparse
from fog_dataloader import FogDataloader as fog
from utils import SparkMET as sm
import optuna


def objective(trial):
    seed = 1987
    torch.manual_seed(seed) # important 
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Check if there is GPU(s): {torch.cuda.is_available()}")

    Exp_name    = 'tune_svt'
    epochs      = 50
    emb_type    = 'SVT'
    opt         = 'adamw'
    loss        = 'BCE'
    early_stop  = 50
    emb_size    = 1024
    num_heads   = 8  
    num_layers  = 6


    # Hyperparameters to be tuned
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3, 0.4, 0.5])
    mlp_size = trial.suggest_categorical("mlp_size", [512, 1028, 2048])


    #print(f"lr = {lr} | wd = {weight_decay} | dropout = {dropout_rate} | emb_size = {emb_size} | heads = {num_heads} | layers: {num_layers} | mlp_size = {mlp_size}")

    SaveDir = '/data1/fog/Hamid/SparkMET/EXPs/'

    data_loader_training, data_loader_validate, data_loader_testing = fog.Fog_DataLoader_npz(
        batch_size   = batch_size,
        WeightR      = False,
        SaveDir      = SaveDir,
        Exp_name     = Exp_name
    )

    SparkMET_Config = sm.Get_model_config(
        img_size     = 32,
        in_channel   = 388,
        in_time      = 4,
        embd_size    = emb_size,
        mlp_size     = mlp_size,
        num_heads    = num_heads,
        dropout      = dropout_rate,
        num_layers   = num_layers,
        embd_type    = emb_type,
        conv_type    = '2d',
    ).return_config()


    SparkMET_Obj = sm.SparkMET(
        SparkMET_Config,
        SaveDir=SaveDir,
        Exp_Name=Exp_name
    )

    model, optimizer, loss_func = SparkMET_Obj.compile(
        optmizer = opt,
        loss     = loss, #'BCE',  # NLLLoss',
        lr       = lr,
        wd       = weight_decay
    )

    model, loss_stat = SparkMET_Obj.train(
        model,
        optimizer,
        loss_func,
        data_loader_training,
        data_loader_validate,
        epochs=epochs,
        early_stop_tolerance=early_stop
    )

    list_outputs = SparkMET_Obj.predict(
        model,
        data_loader_training,
        data_loader_testing
    )

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials = 100)

# Print the result
best_params = study.best_params
best_loss = study.best_value
print(f"Best parameters: {best_params}")
print(f"Best loss: {best_loss}")