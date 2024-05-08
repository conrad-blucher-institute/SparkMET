import torch
import numpy as np
import random 
import argparse
from fog_dataloader import FogDataloader as fog
from utils import SparkMET as sm


def main(args):
    seed = 1987
    torch.manual_seed(seed) # important 
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0")
    print(f"Check if there is GPU(s): {torch.cuda.is_available()}")

    # for i in range(9):
    Exp_name    = args.exp_name #+ '_' + str(i)
    batch_size  = args.batch_size
    embd_size   = args.embd_size
    num_heads   = args.num_heads
    num_layers  = args.num_layers
    mlp_size    = args.mlp_size
    dropout     = args.dropout
    lr          = args.lr
    wd          = args.wd
    epochs      = args.epochs
    embd_type   = args.embd_type
    opt         = args.opt
    loss        = args.loss
    early_stop  = args.early_stop
    conv_type   = args.conv_type

    print(f"EmbType: {embd_type} |lr = {lr} | wd = {wd} | dropout = {dropout}")

    SaveDir = '/data1/fog/Hamid/SparkMET/EXPs/' 
 
    # data_loader_training, data_loader_validate, data_loader_testing = fog.Fog_DataLoader_npz(
    #     batch_size   = batch_size,
    #     WeightR      = False,
    #     SaveDir      = SaveDir,
    #     Exp_name     = Exp_name
    # )

    data_loader_training, data_loader_validate, data_loader_testing = fog.Fog_DataLoader_npz_cv(batch_size = batch_size, 
                                                                                                kfold_id  = 8, 
                                                                                                SaveDir = SaveDir, 
                                                                                                Exp_name = Exp_name, 
                                                                                                WeightR = False)


    configs = sm.ModelConfig(
        img_size     = 32,
        in_channel   = 388,
        in_time      = 4,
        embd_size    = embd_size,
        mlp_size     = mlp_size,
        num_heads    = num_heads,
        dropout      = dropout,
        num_layers   = num_layers,
        embd_type    = embd_type,
        conv_type    = conv_type,
    ).return_config()

    Spark = sm.SparkMET(
        configs,
        SaveDir=SaveDir,
        Exp_Name=Exp_name
    )

    model, optimizer, loss_func = Spark.compile(
        optmizer = opt,
        loss     = loss, 
        lr       = lr,
        wd       = wd
    )


    model, loss_stat = Spark.train(
        model,
        optimizer,
        loss_func,
        data_loader_training,
        data_loader_validate,
        epochs=epochs,
        early_stop_tolerance=early_stop
    )

    list_outputs = Spark.predict(
        model,
        data_loader_training,
        data_loader_testing,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SparkMET Experiment")
    parser.add_argument("--exp_name",   type = str,   default= "test", help = "Experiment name")
    parser.add_argument("--batch_size", type = int,   default= 32,    help = "Batch size")
    parser.add_argument("--embd_size",  type = int,   default= 1024,   help = "Embedding size")
    parser.add_argument("--mlp_size",   type = int,   default= 2048,    help = "MLP size")
    parser.add_argument("--num_heads",  type = int,   default= 8,      help = "Number of attention heads")
    parser.add_argument("--num_layers", type = int,   default= 6,      help = "Number of transformer layers")
    parser.add_argument("--lr",         type = float, default= 0.0001,  help = "Learning rate")
    parser.add_argument("--wd",         type = float, default= 0.01,   help = "Weight decay")
    parser.add_argument("--dropout",    type = float, default= 0.1,    help = "dropout rate")
    parser.add_argument("--epochs",     type = int,   default= 200,    help = "Number of training epochs")
    parser.add_argument("--early_stop", type = int,   default= 50,     help = "The number of early stop epoches")
    parser.add_argument("--embd_type",  type = str,   default= 'VVT',  help = "The embedding type")
    parser.add_argument("--conv_type",  type = str,   default= '2d',   help = "The convolution type")
    parser.add_argument("--opt",        type = str,   default= 'adamw',help = "The optimizer type")
    parser.add_argument("--loss",       type = str,   default= 'BCE',  help = "The loss function")
    args = parser.parse_args()
    main(args)

# python run.py --exp_name test --batch_size 128 --embd_size 1024 --num_heads 8 --num_layers 8 --lr 0.001 --wd 0.01 --epochs 2