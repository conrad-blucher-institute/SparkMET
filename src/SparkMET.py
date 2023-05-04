import torch
import torch.optim as optim
from src import models, engine
import ml_collections


device = "cuda" if torch.cuda.is_available() else "cpu"


class SparkMET_Configs():
    def __init__(self, img_size: int, in_channel: int, in_time: int, embd_size: int, num_heads: int, num_layers: int, FactType: str,):
        self.FactType   = FactType
        self.img_size   = img_size
        self.in_channel = in_channel
        self.in_time    = in_time
        self.embd_size  = embd_size
        self.num_heads  = num_heads
        self.num_layers = num_layers

    def return_config(self):
        if self.FactType   == 'Emb_2D_SP_Patch':
            return self.SparkMET_4D_Emb_2D_SP_Patch()
        elif self.FactType == 'Emb_2D_Channel':
            return self.SparkMET_4D_Emb_2D_Channel()
        elif self.FactType == 'Emb_2D_Patch':
            return self.SparkMET_4D_Emb_2D_Patch()
    
    def SparkMET_4D_Emb_2D_SP_Patch(self):
        """Returns the ViT configuration."""
        config = ml_collections.ConfigDict()
        config.patches     = ml_collections.ConfigDict({'size': (int(self.img_size/4), int(self.img_size/4))})
        config.embd_size   = self.embd_size
        config.in_channels = (int(self.in_channel/self.in_time))
        config.in_times    = self.in_time
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = 512
        config.transformer.num_heads = self.num_heads
        config.transformer.num_layers = self.num_layers
        config.transformer.attention_dropout_rate = 0.0
        config.transformer.dropout_rate = 0.3
        config.transformer.Emb_M = 'Emb_2D_SP_Patch'
        config.classifier = 'token'
        config.representation_size = None

        return config
   
    def SparkMET_4D_Emb_2D_Channel(self):

        """Returns the ViT configuration."""
        config = ml_collections.ConfigDict()
        config.patches     = ml_collections.ConfigDict({'size': (self.img_size, self.img_size)})
        config.embd_size   = self.embd_size
        config.in_channels = 1
        config.in_times    = self.in_time
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = 512
        config.transformer.num_heads = self.num_heads
        config.transformer.num_layers = self.num_layers
        config.transformer.attention_dropout_rate = 0.0
        config.transformer.dropout_rate = 0.3
        config.transformer.Emb_M = 'Emb_2D_Channel'
        config.classifier = 'token'
        config.representation_size = None
        return config
    
    def SparkMET_4D_Emb_2D_Patch(self):
        """Returns the ViT configuration."""
        config = ml_collections.ConfigDict()
        config.patches     = ml_collections.ConfigDict({'size':(int(self.img_size/4), int(self.img_size/4))})
        config.embd_size   = self.embd_size
        config.in_channels = self.in_channel
        config.in_times    = self.in_time
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = 512
        config.transformer.num_heads  = self.num_heads
        config.transformer.num_layers = self.num_layers
        config.transformer.attention_dropout_rate = 0.0
        config.transformer.dropout_rate = 0.3
        config.transformer.Emb_M = 'Emb_2D_Patch'
        config.classifier = 'token'
        config.representation_size = None
        return config



class SparkMET():
    def __init__(self, SparkMET_Configs, SaveDir : str, Exp_Name:str):
        self.SparkMET_Configs   = SparkMET_Configs
        self.SaveDir = SaveDir
        self.Exp_Name = Exp_Name


    def compile(self, optmizer: str, loss: str, lr: float, wd:float, ):

        self.model = models.VisionTransformer(self.SparkMET_Configs, img_size = 32, num_classes=2,).to(device)

        if optmizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr = lr, 
                                    weight_decay = wd)
        if loss == 'NLLLoss': 
            self.loss_func  = torch.nn.NLLLoss() 

        return self.model, self.optimizer, self.loss_func
    
    def train(self, model, optimizer, loss_func, data_loader_training, data_loader_validate, epochs: int, early_stop_tolerance: int):

        trained_model, loss_stat = engine.train(model, optimizer, loss_func,
                                        data_loader_training, data_loader_validate, 
                                        epochs, early_stop_tolerance, self.SaveDir, 
                                        self.Exp_Name)
        
        return trained_model, loss_stat
    
    def predict(self, model, data_loader_training, data_loader_validate, data_loader_testing):
        list_output = engine.predict(model, 
                            data_loader_training, 
                                data_loader_validate, 
                                data_loader_testing, self.SaveDir, 
                                Exp_name = self.Exp_Name,)
        return list_output
