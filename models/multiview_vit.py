import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from torch import Tensor
#from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.jit import Final

seed = 1987
torch.manual_seed(seed) # important 
torch.cuda.manual_seed(seed)
np.random.seed(seed)



class Embeddings_2D_Patch(nn.Module):
    """Construct the embeddings from patch, position embeddings.
        Input: x (size: TxWxHxC)==> Fag Case: 32x32x388
         
        number of patches = (image_size//patch_size)*(image_size//patch_size) = (32/8)*(32/8) = 16
        position encoding = 1 
        embedding size    = 4*8*8*97 (defult = 1024) ==> it can be any number dividable by number of heads!

        Output: embeding token ((16 + 1 )x embeding size (1024)) = 1x17x1024
    """
    def __init__(self, config, img_size):
        super(Embeddings_2D_Patch, self).__init__()

        img_size   = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        n_patches  = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) # in_channels # 

        assert n_patches is not None, ('Number of Patches Can NOT be None!')

        self.patch_embeddings    = Conv2d(in_channels = config.in_channels,
                                       out_channels   = config.embd_size,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))
        self.dropout             = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):

        B = x.shape[0]
        x = torch.reshape(x, [B,  x.shape[1]*x.shape[4], x.shape[2], x.shape[3]])
    
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = self.patch_embeddings(x)

        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class Embeddings_2D_Channel(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size):
        super(Embeddings_2D_Channel, self).__init__()

        img_size   = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        n_patches  = 388

        self.patch_embeddings    = Conv2d(in_channels  = 1,
                                       out_channels    = config.embd_size,
                                       kernel_size     = patch_size,
                                       stride          = patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))
        self.dropout             = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.reshape(x, [B,  x.shape[1]*x.shape[4], x.shape[2], x.shape[3]])

        x_stack = None

        for i in range(x.shape[1]):
            this_variable = x[:, i, :, :]
            this_variable = torch.unsqueeze(this_variable, dim=1) 
            this_variable = self.patch_embeddings(this_variable)

            this_variable = this_variable.flatten(2)
            this_variable = this_variable.transpose(-1, -2)
            if x_stack is None: 
                x_stack = this_variable
            else:
                x_stack = torch.cat((x_stack, this_variable), dim=1)

        x_stack = torch.cat((cls_tokens, x_stack), dim=1)
        embeddings = x_stack + self.position_embeddings

        return embeddings
    
class Embeddings_2D_SpatioTemporal_Patch(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    Input: x (size: TxWxHxC)==> Fag Case: 4x32x32x97
        
    number of patches = (image_size//patch_size)*(image_size//patch_size) = (32/8)*(32/8) = 16
    position encoding = 1 
    embedding size    = (8*8*97)*4 (defult = 1024) ==> it can be any number dividable by number of heads!

    Output: embeding token ((4*16 + 1 )x embeding size (1024)) = 1x65x1024

    """
    def __init__(self, config, img_size):
        super(Embeddings_2D_SpatioTemporal_Patch, self).__init__()

        img_size   = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        temporal_res = 4
        n_patches  = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * temporal_res # in_channels # 

        assert n_patches is not None, ('Number of Patches Can NOT be None!')

        self.patch_embeddings = Conv2d(in_channels = config.in_channels,
                                       out_channels   = config.embd_size,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))
        self.dropout             = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = x.permute(0, 4, 2, 3, 1)
    
        B = x.shape[0]
        T = x.shape[-1]
        C = x.shape[1] 
        cls_tokens = self.cls_token.expand(B, -1, -1)


        x_stack = None
        for t in range(T): 
            this_time = x[:, :, :, :, t]
            this_time = self.patch_embeddings(this_time)

            this_time = this_time.flatten(2)
            this_time = this_time.transpose(-1, -2)

            if x_stack is None: 
                x_stack = this_time
            else:
                x_stack = torch.cat((x_stack, this_time), dim = 1)

        x_stack = torch.cat((cls_tokens, x_stack), dim = 1)
        embeddings = x_stack + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings
    
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1     = Linear(config.embd_size, config.transformer["mlp_dim"])
        self.fc2     = Linear(config.transformer["mlp_dim"], config.embd_size)
        self.act_fn  = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):


    def __init__(
            self,
            config
    ):
        super().__init__()
        self.config = config
        qkv_bias=False,
        qk_norm=False,

        assert self.config.embd_size % self.config.transformer["num_heads"] == 0, 'dim should be divisible by num_heads'
   
        self.head_dim = self.config.embd_size // self.config.transformer["num_heads"]
        self.scale = self.head_dim ** -0.5


        self.qkv = nn.Linear(self.config.embd_size, self.config.embd_size * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(self.config.transformer["attention_dropout_rate"])
        self.proj = nn.Linear(self.config.embd_size, self.config.embd_size)
        self.proj_drop = nn.Dropout(self.config.transformer["attention_dropout_rate"])


        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.normal_(self.proj.bias, std=1e-6)
        nn.init.normal_(self.qkv.bias, std=1e-6)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.config.transformer["num_heads"], self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)


        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        weights = attn.softmax(dim=-1)
        attn = self.attn_drop(weights)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights
    
class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.embd_size
        self.attention_norm = LayerNorm(config.embd_size, eps=1e-6)
        self.ffn_norm       = LayerNorm(config.embd_size, eps=1e-6)
        self.ffn            = Mlp(config)
        self.attn           = Attention(config) 

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.embd_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)

        return encoded, attn_weights

class Transformer_Emb_Patch(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer_Emb_Patch, self).__init__()

        self.embeddings = Embeddings_2D_Patch(config, img_size=img_size)
        self.encoder    = Encoder(config)
    def forward(self, input_ids):
        embedding_output      = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

class Transformer_Emb_Channel(nn.Module): 
    def __init__(self, config, img_size):
        super(Transformer_Emb_Channel, self).__init__()

        self.embeddings = Embeddings_2D_Channel(config, img_size=img_size)
        self.encoder    = Encoder(config)

    def forward(self, input_ids):
        embedding_output      = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights
    
class Transformer_Emb_SP(nn.Module):  
    def __init__(self, config, img_size):
        super(Transformer_Emb_SP, self).__init__()

        self.embeddings = Embeddings_2D_SpatioTemporal_Patch(config, img_size=img_size)
        self.encoder    = Encoder(config)

    def forward(self, input_ids):
        embedding_output      = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights  
    
class Factorized_VisionTransformer(nn.Module):
    def __init__(self, config, img_size = 32, num_classes = 2,):
        super(Factorized_VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.classifier  = config.classifier


        Emb_M = config.transformer["Emb_M"]

        if Emb_M   == 'Emb_2D_Patch':
            self.transformer = Transformer_Emb_Patch(config, img_size)
        elif Emb_M == 'Emb_2D_Channel':
            self.transformer = Transformer_Emb_Channel(config, img_size)
        elif Emb_M == 'Emb_2D_SP_Patch':
            self.transformer = Transformer_Emb_SP(config, img_size)

        self.head        = Linear(config.embd_size, num_classes) #, bias = True

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.normal_(self.head.bias, std=1e-6)
  

    def forward(self, x):
        x, weights = self.transformer(x)

        logits = self.head(x[:, 0])

        return weights, logits 





        

