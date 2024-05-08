import torch
import torch.optim as optim
import ml_collections
import numpy as np
import random
import sys
sys.path.append('../')
from models import multiview_vit as vit
from models import ViT_LRP as lrp
from utils import engine, losses

seed = 1987
torch.manual_seed(seed) # important 
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda:0")


class ModelConfig():
    def __init__(self, img_size: int, 
                 in_channel: int, 
                 in_time: int, 
                 embd_size: int, 
                 mlp_size: int,
                 num_heads: int, 
                 num_layers: int, 
                 dropout:float,
                 conv_type:str, 
                 embd_type: str,):
        

        self.embd_type = embd_type
        self.img_size = img_size
        self.in_channel = in_channel
        self.in_time = in_time
        self.embd_size = embd_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type

    def return_config(self):
        if self.embd_type   == 'VVT':
            return self.SparkMET_VVT()
            
        elif self.embd_type == 'UVT':
            return self.SparkMET_UVT()
        
        elif self.embd_type == 'STT':
            return self.SparkMET_STT()
        
        elif self.embd_type == 'FactSTT':
            return self.SparkMET_FactSTT()
        
        elif self.embd_type == 'FactVSTT':
            return self.SparkMET_FactVSTT()
        
        elif self.embd_type == 'PIT':
            return self.SparkMET_PASTT()
        
        elif self.embd_type == 'SVT':
            return self.SparkMET_SVT()
    
    def SparkMET_VVT(self):
        """Returns the Spatio-temporal configuration."""
        config = ml_collections.ConfigDict()
        config.img_size = self.img_size
        config.patches = int(self.img_size/4) #ml_collections.ConfigDict({'size': (int(self.img_size/4), int(self.img_size/4))}) # 8x8
        config.embd_size = self.embd_size 
        config.in_channels = self.in_channel
        config.in_times = self.in_time
        config.mlp_dim = self.mlp_size
        config.num_heads = self.num_heads
        config.num_layers = self.num_layers
        config.attention_dropout_rate = self.dropout 
        config.dropout_rate = self.dropout 
        config.embd_type = 'VVT'
        config.conv_type = self.conv_type
        config.classifier = 'token'
        config.representation_size = None

        return config
   
    def SparkMET_UVT(self):

        """Returns the channel-wise configuration."""
        config = ml_collections.ConfigDict()
        config.img_size = self.img_size
        config.patches = self.img_size 
        config.embd_size = self.embd_size
        config.in_channels = 1
        config.in_times = self.in_time
        config.mlp_dim = self.mlp_size
        config.num_heads = self.num_heads
        config.num_layers = self.num_layers
        config.attention_dropout_rate = self.dropout 
        config.dropout_rate = self.dropout 
        config.embd_type = 'UVT'
        config.conv_type = self.conv_type
        config.classifier = 'token'
        config.representation_size = None
        return config
    
    def SparkMET_STT(self):
        """Returns the patch wise configuration."""
        config = ml_collections.ConfigDict()
        config.img_size = self.img_size
        config.patches = int(self.img_size/4)
        config.embd_size = self.embd_size
        config.in_channels = (int(self.in_channel/self.in_time)) 
        config.in_times = self.in_time
        config.mlp_dim = self.mlp_size
        config.num_heads = self.num_heads
        config.num_layers = self.num_layers
        config.attention_dropout_rate = self.dropout 
        config.dropout_rate = self.dropout 
        config.embd_type = 'STT'
        config.conv_type = self.conv_type
        config.classifier = 'token'
        config.representation_size = None
        return config
    
    def SparkMET_FactSTT(self):
        """Returns the patch wise configuration."""
        config = ml_collections.ConfigDict()
        config.img_size = self.img_size
        config.patches = int(self.img_size/4) 
        config.embd_size = self.embd_size
        config.in_channels = (int(self.in_channel/self.in_time)) 
        config.in_times = self.in_time
        config.mlp_dim = self.mlp_size
        config.num_heads = self.num_heads
        config.num_layers = self.num_layers
        config.attention_dropout_rate = self.dropout 
        config.dropout_rate = self.dropout 
        config.embd_type = 'FactSTT'
        config.conv_type = self.conv_type
        config.classifier = 'token'
        config.representation_size = None
        return config
    
    def SparkMET_FactVSTT(self):
        """Returns the patch wise configuration."""
        config = ml_collections.ConfigDict()
        config.img_size = self.img_size
        config.patches = int(self.img_size/4)
        config.embd_size = self.embd_size
        config.in_channels = (int(self.in_channel/self.in_time)) 
        config.in_times = self.in_time
        config.mlp_dim = self.mlp_size
        config.num_heads  = self.num_heads
        config.num_layers = self.num_layers
        config.attention_dropout_rate = self.dropout 
        config.dropout_rate = self.dropout 
        config.embd_type = 'FactVSTT'
        config.conv_type = self.conv_type
        config.classifier = 'token'
        config.representation_size = None
        return config


    def SparkMET_PASTT(self):
        """Returns the ViT configuration."""
        config = ml_collections.ConfigDict()
        config.img_size = self.img_size
        config.patches = int(self.img_size/4)
        config.embd_size = self.embd_size
        config.in_channels = (int(self.in_channel/self.in_time))
        config.in_times = self.in_time
        config.mlp_dim = self.mlp_size
        config.num_heads = self.num_heads
        config.num_layers = self.num_layers
        config.attention_dropout_rate = self.dropout 
        config.dropout_rate = self.dropout 
        config.embd_type = 'PIT'
        config.conv_type = self.conv_type
        config.classifier = 'token'
        config.representation_size = None
        return config
    
    def SparkMET_SVT(self):
        """Returns the ViT configuration."""
        config = ml_collections.ConfigDict()
        config.img_size = self.img_size
        config.patches = int(self.img_size/4)
        config.embd_size = self.embd_size
        config.in_channels = (int(self.in_channel/self.in_time))
        config.in_times = self.in_time
        config.mlp_dim = self.mlp_size
        config.num_heads = self.num_heads
        config.num_layers = self.num_layers
        config.attention_dropout_rate = self.dropout 
        config.dropout_rate = self.dropout 
        config.embd_type = 'SVT'
        config.conv_type = self.conv_type
        config.classifier = 'token'
        config.representation_size = None
        return config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SparkMET():
    def __init__(self, config, SaveDir : str, Exp_Name:str):
        self.config   = config
        self.SaveDir  = SaveDir
        self.Exp_Name = Exp_Name

    def compile(self, optmizer: str, loss: str, lr: float, wd:float, ):

        self.model = vit.multi_view_models(self.config, cond = False).to(device)
        # self.model = lrp.VisionTransformer(self.config).to(device)

        num_parameters = count_parameters(self.model)
        print(f"The number of model's parameters = {num_parameters}")


        if optmizer == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=lr, 
                                    weight_decay=wd)
        elif optmizer == 'adam': 
            self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr = lr, 
                                    weight_decay = wd, 
                                    betas=(0.9, 0.999), eps=1e-8)
        elif optmizer is None: 
            self.optimizer = optim.AdamW(self.model.parameters(), 
                                         lr=lr, 
                                         weight_decay=wd)
        
        if loss == 'NLLLoss': 
            self.loss_func  = torch.nn.NLLLoss() 
        elif loss == 'BCE': 
            self.loss_func  = torch.nn.BCEWithLogitsLoss() 
        elif loss == 'MyBCE': 
            self.loss_func  = losses.MyFocalLoss(gamma=100, alpha = 5) 
        elif loss =='focal':
            self.loss_func = losses.MyFocalLoss(gamma=5, alpha = 5)

        return self.model, self.optimizer, self.loss_func
    
    def train(self, model, optimizer, loss_func, data_loader_training, data_loader_validate, epochs: int, early_stop_tolerance: int):

        trained_model, loss_stat = engine.train(model, 
                                                optimizer, 
                                                loss_func, 
                                                data_loader_training, 
                                                data_loader_validate, 
                                                epochs, 
                                                early_stop_tolerance, 
                                                self.SaveDir, 
                                                self.Exp_Name)
        
        return trained_model, loss_stat
    
    def predict(self, model, data_loader_training, data_loader_testing):
        list_output = engine.predict(model, 
                                data_loader_training, 
                                data_loader_testing, 
                                self.config,
                                SaveDir = self.SaveDir, 
                                Exp_name = self.Exp_Name, 
                                by = 'HSS')
        return list_output
    
    def inference(self, model, data_loader):
        list_output = engine.inference(model, 
                                       data_loader, 
                                       emb_type = self.config.embd_type, 
                                       SaveDir = self.SaveDir, 
                                       Exp_name =  self.Exp_Name, )

        return list_output
