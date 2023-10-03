# # import itertools
# # a = [[64, 256],[1024, 2048],[8, 16], [6, 12], [0.001, 0.0001]]
# # full_list = list(itertools.product(*a))


# # for i in range(len(full_list)):

# # batch_size = full_list[i][0]
# # embd_size  = full_list[i][1]
# # num_heads  = full_list[i][2]
# # num_layers = full_list[i][3]
# # wd         = full_list[i][4]



import torch
import numpy
import argparse
from fog_dataloader import FogDataloader as fog
from utils import SparkMET as sm

def main(args):
    seed = 1987
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Check if there is GPU(s): {torch.cuda.is_available()}")

    Exp_name    = args.exp_name
    batch_size  = args.batch_size
    embd_size   = args.embd_size
    num_heads   = args.num_heads
    num_layers  = args.num_layers
    lr          = args.lr
    wd          = args.wd
    epochs      = args.epochs

    SaveDir = '/data1/fog/SparkMET/EXPs/'

    data_loader_training, data_loader_validate, data_loader_testing = fog.Fog_DataLoader_npz(
        batch_size   = batch_size,
        WeightR      = False,
        SaveDir      = SaveDir,
        Exp_name     = Exp_name
    )

    SparkMET_Config = sm.SparkMET_Configs(
        img_size     = 32,
        in_channel   = 388,
        in_time      = 4,
        embd_size    = embd_size,
        num_heads    = num_heads,
        num_layers   = num_layers,
        FactType     = 'Emb_2D_Patch'
    ).return_config()

    SparkMET_Obj = sm.SparkMET(
        SparkMET_Config,
        SaveDir=SaveDir,
        Exp_Name=Exp_name
    )

    model, optimizer, loss_func = SparkMET_Obj.compile(
        optmizer = 'adam',
        loss     = 'BCE',  # NLLLoss',
        lr       = lr,
        wd       = wd
    )

    model, loss_stat = SparkMET_Obj.train(
        model,
        optimizer,
        loss_func,
        data_loader_training,
        data_loader_validate,
        epochs=epochs,
        early_stop_tolerance=200
    )

    list_outputs = SparkMET_Obj.predict(
        model,
        data_loader_training,
        data_loader_validate,
        data_loader_testing
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SparkMET Experiment")
    parser.add_argument("--exp_name", type=str, default="test", help = "Experiment name")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--embd_size", type=int, default=1024, help="Embedding size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    args = parser.parse_args()
    main(args)


# python run.py --exp_name test --batch_size 128 --embd_size 1024 --num_heads 8 --num_layers 8 --lr 0.001 --wd 0.01 --epochs 2