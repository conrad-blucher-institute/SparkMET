import warnings
warnings.simplefilter(action='ignore')
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json

from models import configs
from src import dataloader
from models import transformers, engine

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Check if there is GPU(s): {torch.cuda.is_available()}")




def main(data_config_dict: dict,
        model_config_dict : dict, 
        training_config_dict: dict,
        Exp_name: str,
):

    data_loader_training, data_loader_validate, data_loader_testing = dataloader.return_data_loaders (data_config_dict, training_config_dict, Exp_name)

     
    if data_config_dict.data_straucture == '1D': 
        model = transformers.Transformer1d(model_config_dict)

    elif data_config_dict.data_straucture == '2D':
        model_type = 'ViT-L_32'
        config = transformers.CONFIGS[model_type]
        model = transformers.VisionTransformer(config, img_size=32, num_classes=2,)


    parallel_net = nn.DataParallel(model, device_ids = [0,1,2, 3])
    parallel_net = parallel_net.to(0)


    parallel_net, loss_stat = engine.train(parallel_net, data_loader_training, data_loader_validate, data_loader_testing, Exp_name)

    list_output = engine.predict(parallel_net, 
                                                            data_loader_training, 
                                                            data_loader_validate, 
                                                            data_loader_testing, 
                                                            Exp_name = Exp_name,)

if __name__ == "__main__":

    main(data_config_dict     = config.data_config_dict,
        model_config_dict     = model_config_dict, 
        training_config_dict  = training_config_dict,
        Exp_name              = 'test')

