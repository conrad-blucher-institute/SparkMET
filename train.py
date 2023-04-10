import warnings
warnings.simplefilter(action='ignore')
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json

from src import dataloader
from models import transformers, engine, configs

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Check if there is GPU(s): {torch.cuda.is_available()}")


Exp_name  = 'test'
data_loader_training, data_loader_validate, data_loader_testing = dataloader.return_data_loaders(configs.data_config_dict, 
                                                                                                 configs.get_train_hyperparameter_config, 
                                                                                                 Exp_name)

#model = transformers.Transformer1d(configs.SparkMET_1D_config()).to(device)
model = transformers.VisionTransformer(configs.SparkMET_3D_config(), img_size=32, num_classes=2,).to(device)


loss_func = torch.nn.NLLLoss() 
optimizer = optim.Adam(model.parameters(), 
                        lr = configs.get_train_hyperparameter_config['lr'], 
                        weight_decay = configs.get_train_hyperparameter_config['wd'])

model, loss_stat = engine.train(model, optimizer, loss_func,
                                        configs.get_train_hyperparameter_config,
                                        data_loader_training, 
                                        data_loader_validate, 
                                        Exp_name)
                                  

list_output = engine.predict(model, 
                            data_loader_training, 
                                data_loader_validate, 
                                data_loader_testing, 
                                Exp_name = Exp_name,)


#def main(data_config_dict: dict,
#        model_config_dict : dict, 
#       training_config_dict: dict,
#        Exp_name: str,
#):

#parallel_net = nn.DataParallel(model, device_ids = [0, 1, 2, 3])
#parallel_net = parallel_net.to(0)

#if __name__ == "__main__":
#    main(data_config_dict     = configs.data_config_dict,
#        model_config_dict     = configs.SparkMET_1D_config(), 
#       training_config_dict  = configs.get_train_hyperparameter_config,
#       Exp_name              = 'test')





