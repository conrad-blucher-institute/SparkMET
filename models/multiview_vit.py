import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm, Conv3d
from torch.nn.modules.utils import _pair
from torch import Tensor
import random
from einops import rearrange
# from timm.models.layers import DropPath
from utils import engine
device = torch.device("cuda:0")





seed = 1987
torch.manual_seed(seed) #important 
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class VisEmbdPatch(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.proj = nn.Embedding(64, config.embd_size)

    def forward(self, vis):

        aux = self.proj(vis) 

        return aux
    
class VVTEmbdPatch(nn.Module):
    """ VVT: Vanilla Vision Tokenizer
        Construct the embeddings from patch, position embeddings.
        Input: x (size: TxWxHxC)==> Fag Case: 32x32x388
         
        number of patches = (image_size//patch_size)*(image_size//patch_size) = (32/8)*(32/8) = 16
        position encoding = 1 
        embedding size    = 4*8*8*97 (defult = 1024) ==> it can be any number dividable by number of heads!

        Output: embeding token ((16 + 1 )x embeding size (1024)) = 1x17x1024
    """
    def __init__(self, config):
        super(VVTEmbdPatch, self).__init__()

        img_size   = _pair(config.img_size)
        patch_size = _pair(config.patches)
        n_patches  = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) # in_channels # 

        assert n_patches is not None, ('Number of Patches Can NOT be None!')

        if config.conv_type == '2d':
            self.patch_embeddings    = Conv2d(in_channels = config.in_channels,
                                            out_channels = config.embd_size,
                                            kernel_size  = patch_size,
                                            stride       = patch_size)
            
        elif config.conv_type == '3d':
            self.patch_embeddings    = Conv3d(in_channels = config.in_channels,
                                        out_channels = config.embd_size,
                                        kernel_size  = patch_size,
                                        stride       = patch_size)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))
        self.dropout             = Dropout(config.dropout_rate)

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

class UVTEmbd2DPatchV1(nn.Module):
    """ Uniform Variable Tokenizer
        Construct the embeddings from patch, position embeddings.
        Input: x (size: TxWxHxC)==> Fag Case: 32x32x388
         
        number of patches = C = 388
        position encoding = 1 
        embedding size    = (defult = 1024) ==> it can be any number dividable by number of heads!

        Output: embeding token ((388 + 1 )x embeding size (1024)) = 1x389x1024
    """
    def __init__(self, config):
        super(UVTEmbd2DPatchV1, self).__init__()

        patch_size = _pair(config.img_size)
        n_patches  = 388

        self.patch_embeddings    = Conv2d(in_channels  = 1,
                                       out_channels    = config.embd_size,
                                       kernel_size     = patch_size,
                                       stride          = patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))
        self.dropout             = Dropout(config.dropout_rate)


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
        embeddings = self.dropout(embeddings)

        return embeddings

class STTEmbdPatch(nn.Module):
    """ Spatio-Temporal Tokenizer
    Construct the embeddings from patch, position embeddings.
    Input: x (size: TxWxHxC)==> Fag Case: 4x32x32x97
        
    number of patches = (image_size//patch_size)*(image_size//patch_size) = (32/8)*(32/8) = 16
    position encoding = 1 
    embedding size    = (8*8*97)*4 (defult = 1024) ==> it can be any number dividable by number of heads!

    Output: embeding token ((4*16 + 1 )x embeding size (1024)) = 1x65x1024

    """
    def __init__(self, config):
        super(STTEmbdPatch, self).__init__()

        img_size     = _pair(config.img_size)
        patch_size   = _pair(config.patches)
        temporal_res = 4
        n_patches    = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * temporal_res # in_channels # 

        assert n_patches is not None, ('Number of Patches Can NOT be None!')

        if config.conv_type == '2d':
            self.patch_embeddings = Conv2d(in_channels = config.in_channels,
                                        out_channels   = config.embd_size,
                                        kernel_size    = patch_size,
                                        stride         = patch_size)
        elif config.conv_type == '3d':
            self.patch_embeddings = Conv3d(in_channels = config.in_channels,
                                    out_channels   = config.embd_size,
                                    kernel_size    = patch_size,
                                    stride         = patch_size)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))
        self.dropout             = Dropout(config.dropout_rate)

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

class PITEmbdPatch(nn.Module):

    def __init__(self, config):
        super(PITEmbdPatch, self).__init__()

        img_size     = _pair(config.img_size)
        patch_size   = _pair(config.patches)
        group_res    = 5
        n_patches    = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * group_res

        assert n_patches is not None, ('Number of Patches Can NOT be None!')

        
        self.indices, self.tensors = engine.return_variable_index(category = 'v1')

        self.patch_embeddings_G1 = Conv2d(in_channels = len(self.indices[0]),
                                       out_channels   = config.embd_size,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)

        self.patch_embeddings_G2 = Conv2d(in_channels = len(self.indices[1]),
                                       out_channels   = config.embd_size,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)

        
        self.patch_embeddings_G3 = Conv2d(in_channels = len(self.indices[2]),
                                       out_channels   = config.embd_size,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)


        
        self.patch_embeddings_G4 = Conv2d(in_channels = len(self.indices[3]),
                                       out_channels   = config.embd_size,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)

        self.patch_embeddings_G5 = Conv2d(in_channels = len(self.indices[4]),
                                       out_channels   = config.embd_size,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))
        self.dropout             = Dropout(config.dropout_rate)

    def forward(self, x):
        x = x.permute(0, 4, 2, 3, 1)
        B = x.shape[0]
        x = torch.reshape(x, [B,  x.shape[1]*x.shape[4], x.shape[2], x.shape[3]])
        cls_tokens = self.cls_token.expand(B, -1, -1)

        G1 = x.index_select(1, self.tensors[0])
        G1_this_time = self.patch_embeddings_G1(G1)
        G1_this_time = G1_this_time.flatten(2)
        G1_this_time = G1_this_time.transpose(-1, -2)

        G2 = x.index_select(1, self.tensors[1])
        G2_this_time = self.patch_embeddings_G2(G2)
        G2_this_time = G2_this_time.flatten(2)
        G2_this_time = G2_this_time.transpose(-1, -2)

        G3 = x.index_select(1, self.tensors[2])
        G3_this_time = self.patch_embeddings_G3(G3)
        G3_this_time = G3_this_time.flatten(2)
        G3_this_time = G3_this_time.transpose(-1, -2)

        G4 = x.index_select(1, self.tensors[3])
        G4_this_time = self.patch_embeddings_G4(G4)
        G4_this_time = G4_this_time.flatten(2)
        G4_this_time = G4_this_time.transpose(-1, -2)

        G5 = x.index_select(1, self.tensors[4])
        G5_this_time = self.patch_embeddings_G5(G5)
        G5_this_time = G5_this_time.flatten(2)
        G5_this_time = G5_this_time.transpose(-1, -2)

        x_stack_this_time = torch.cat((G1_this_time, G2_this_time, G3_this_time, G4_this_time, G5_this_time), dim = 1)

        x_stack_this_time = torch.cat((cls_tokens, x_stack_this_time), dim = 1)
        embeddings = x_stack_this_time + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class SVTEmbdPatch(nn.Module):

    def __init__(self, config):
        super(SVTEmbdPatch, self).__init__()

        img_size     = _pair(config.img_size)
        patch_size   = _pair(config.patches)
        variable_cat = 9
        n_patches    = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * variable_cat

        assert n_patches is not None, ('Number of Patches Can NOT be None!')
        self.indices, self.tensors = engine.return_variable_index(category = 'var')

        self.patch_embeddings_temp = Conv2d(in_channels = len(self.indices[0]),
                                       out_channels   = config.embd_size,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)

        self.patch_embeddings_u10 = Conv2d(in_channels = len(self.indices[1]),
                                       out_channels   = config.embd_size,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)

        
        self.patch_embeddings_v10 = Conv2d(in_channels = len(self.indices[2]),
                                       out_channels   = config.embd_size,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)


        self.patch_embeddings_TKE = Conv2d(in_channels = len(self.indices[3]),
                                       out_channels   = config.embd_size,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)

        self.patch_embeddings_Q = Conv2d(in_channels = len(self.indices[4]),
                                       out_channels   = config.embd_size,
                                       kernel_size    = patch_size,
                                       stride         = patch_size)
        
        self.patch_embeddings_RH = Conv2d(in_channels = len(self.indices[5]),
                                out_channels   = config.embd_size,
                                kernel_size    = patch_size,
                                stride         = patch_size)

        self.patch_embeddings_VVEL = Conv2d(in_channels = len(self.indices[6]),
                                out_channels   = config.embd_size,
                                kernel_size    = patch_size,
                                stride         = patch_size)

        self.patch_embeddings_VIS = Conv2d(in_channels = len(self.indices[7]),
                                out_channels   = config.embd_size,
                                kernel_size    = patch_size,
                                stride         = patch_size)

        self.patch_embeddings_SST = Conv2d(in_channels = len(self.indices[8]),
                                out_channels   = config.embd_size,
                                kernel_size    = patch_size,
                                stride         = patch_size)
        

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))
        self.dropout             = Dropout(config.dropout_rate)

    def forward(self, x):
        x = x.permute(0, 4, 2, 3, 1)
        B = x.shape[0]
        x = torch.reshape(x, [B,  x.shape[1]*x.shape[4], x.shape[2], x.shape[3]])
        cls_tokens = self.cls_token.expand(B, -1, -1)

        temp_idx = x.index_select(1, self.tensors[0])
        temp = self.patch_embeddings_temp(temp_idx)
        temp = temp.flatten(2)
        temp = temp.transpose(-1, -2)

        u10_idx = x.index_select(1, self.tensors[1])
        U10 = self.patch_embeddings_u10(u10_idx)
        U10 = U10.flatten(2)
        U10 = U10.transpose(-1, -2)

        v10_idx = x.index_select(1, self.tensors[2])
        V10 = self.patch_embeddings_v10(v10_idx)
        V10 = V10.flatten(2)
        V10 = V10.transpose(-1, -2)

        TKE_idx = x.index_select(1, self.tensors[3])
        TKE = self.patch_embeddings_TKE(TKE_idx)
        TKE = TKE.flatten(2)
        TKE = TKE.transpose(-1, -2)

  
        Q_idx = x.index_select(1, self.tensors[4])
        Q = self.patch_embeddings_Q(Q_idx)
        Q = Q.flatten(2)
        Q = Q.transpose(-1, -2)

        RH_idx = x.index_select(1, self.tensors[5])
        RH = self.patch_embeddings_RH(RH_idx)
        RH = RH.flatten(2)
        RH = RH.transpose(-1, -2)

        VVEL_idx = x.index_select(1, self.tensors[6])
        VVEL = self.patch_embeddings_VVEL(VVEL_idx)
        VVEL = VVEL.flatten(2)
        VVEL = VVEL.transpose(-1, -2)

        VIS_idx = x.index_select(1, self.tensors[7])
        VIS = self.patch_embeddings_VIS(VIS_idx)
        VIS = VIS.flatten(2)
        VIS = VIS.transpose(-1, -2)

        SST_idx = x.index_select(1, self.tensors[8])
        SST = self.patch_embeddings_SST(SST_idx)
        SST = SST.flatten(2)
        SST = SST.transpose(-1, -2)

        x_stack_this_time = torch.cat((temp, U10, V10, TKE, Q, RH, VVEL, VIS, SST), dim = 1)

        x_stack_this_time = torch.cat((cls_tokens, x_stack_this_time), dim = 1)
        embeddings = x_stack_this_time + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings
    
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1     = Linear(config.embd_size, config.mlp_dim)
        self.fc2     = Linear(config.mlp_dim, config.embd_size)
        self.act_fn  = torch.nn.functional.gelu
        self.dropout = Dropout(config.dropout_rate)

    #     self._init_weights()
    # def _init_weights(self):
    #     nn.init.xavier_uniform_(self.fc1.weight)
    #     nn.init.xavier_uniform_(self.fc2.weight)
    #     nn.init.normal_(self.fc1.bias, std=1)
    #     nn.init.normal_(self.fc2.bias, std=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):

    def __init__(
            self,
            config
    ):
        super().__init__()
        self.config = config
        qkv_bias = False,
        qk_norm  = False,

        assert self.config.embd_size % self.config.num_heads == 0, 'dim should be divisible by num_heads'
   
        self.head_dim = self.config.embd_size // self.config.num_heads
        self.scale = self.head_dim ** -0.5


        self.qkv = nn.Linear(self.config.embd_size, self.config.embd_size * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(self.config.attention_dropout_rate)
        self.proj = nn.Linear(self.config.embd_size, self.config.embd_size)
        self.proj_drop = nn.Dropout(self.config.attention_dropout_rate)


    #     self._init_weights()

    # def _init_weights(self):
    #     nn.init.xavier_uniform_(self.proj.weight)
    #     nn.init.xavier_uniform_(self.qkv.weight)
    #     nn.init.normal_(self.proj.bias, std=1)
    #     nn.init.normal_(self.qkv.bias, std=1)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.config.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
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
        for _ in range(config.num_layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)

        return encoded, attn_weights

class multi_view_models(nn.Module):
    def __init__(self, config, cond:str):
        super(multi_view_models, self).__init__()

        self.config = config
        self.cond = cond

        self.embd_type = config.embd_type

        self.vis_embeddings = VisEmbdPatch(config)
        
        if self.embd_type == 'VVT':
            self.embeddings = VVTEmbdPatch(config)

        elif self.embd_type == 'UVT':
            self.embeddings = UVTEmbd2DPatchV1(config)

        elif self.embd_type == 'STT':
            self.embeddings = STTEmbdPatch(config)

        elif self.embd_type == 'SVT':
            self.embeddings = SVTEmbdPatch(config)

        elif self.embd_type == 'PIT':
            self.embeddings = PITEmbdPatch(config)

        self.encoder = Encoder(config)

        self.pool = None
        self.to_latent = nn.Identity()
        self.head = nn.Linear(config.embd_size, 2)
        

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier uniform initialization for Linear layers
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Typically, LayerNorm bias is initialized to 0 and weight to 1
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, vis):

        if self.cond: 
            vis_viewed = self.vis_embeddings(vis).unsqueeze(1)
            x = self.embeddings(x)
            x =  torch.cat((x, vis_viewed), dim = 1)
            
        else: 
            x  = self.embeddings(x)

        x, weights = self.encoder(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        logits = self.head(x) 
        
        return logits 
























# class UVTEmbd2DPatchV2(nn.Module):
#     """ Uniform Variable Tokenizer
#         Construct the embeddings from patch, position embeddings.
#         Input: x (size: TxWxHxC)==> Fag Case: 32x32x388
         
#         number of patches = C = 388
#         position encoding = 1 
#         embedding size    = (defult = 1024) ==> it can be any number dividable by number of heads!

#         Output: embeding token ((388 + 1 )x embeding size (1024)) = 1x389x1024
#     """
#     def __init__(self, config):
#         super(UVTEmbd2DPatchV2, self).__init__()

#         patch_size = _pair(32)
#         n_patches  = 388

#         self.patch_embeddings    = Conv2d(in_channels  = 1,
#                                        out_channels    = config.embd_size,
#                                        kernel_size     = patch_size,
#                                        stride          = patch_size)

#         self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
#         self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))
#         self.dropout             = Dropout(config.dropout_rate)


#     def forward(self, x):

#         B = x.shape[0]
#         cls_tokens = self.cls_token.expand(B, -1, -1)

#         # Reshape x to bring the channel dimension as the batch dimension
#         # Original x shape: [B, C, H, W]
#         x = x.permute(1, 0, 2, 3)  # New shape: [C, B, H, W]
#         x = x.reshape(-1, 1, *x.shape[-2:])  # New shape: [C * B, 1, H, W]

#         # Apply the convolution to all patches simultaneously
#         x = self.patch_embeddings(x)

#         # Reshape and permute to get back to the original form
#         x = x.reshape(-1, B, x.shape[-3], x.shape[-2] * x.shape[-1]).transpose(0, 1)  # [B, C, embd_size, H*W]
#         x = x.flatten(2)  #[B, C, embd_size]

#         # Concatenate the class token
#         x = torch.cat((cls_tokens, x), dim=1)

#         embeddings = x + self.position_embeddings
#         embeddings = self.dropout(embeddings)

#         return embeddings

# class FactorizedSpatialEmbdPatch(nn.Module):
#     def __init__(self, config):
#         super(FactorizedSpatialEmbdPatch, self).__init__()

#         img_size   = _pair(config.img_size)
#         patch_size = _pair(config.patches)
#         n_patches  = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) 

#         assert n_patches is not None, ('Number of Patches Can NOT be None!')

#         if config.conv_type == '2d':
#             self.patch_embeddings    = Conv2d(in_channels = config.in_channels,
#                                             out_channels = config.embd_size,
#                                             kernel_size  = patch_size,
#                                             stride       = patch_size)
            
#             # self.patch_embeddings    = nn.Sequential(
#             #                             nn.Conv2d(config.in_channels, config.embd_size // 4, kernel_size=7, stride=4, padding=3, bias=False),  
#             #                             nn.BatchNorm2d(config.embd_size // 4),
#             #                             nn.PReLU(),
#             #                             nn.Conv2d(config.embd_size // 4, config.embd_size // 2, kernel_size=3, stride=2, padding=1, bias=False),  
#             #                             nn.BatchNorm2d(config.embd_size // 2),
#             #                             nn.PReLU(),
#             #                             nn.Conv2d(config.embd_size // 2, config.embd_size, kernel_size=3, stride=1, padding=1, bias=False), 
#             #                         )
            
#         elif config.conv_type == '3d':
#             self.patch_embeddings    = Conv3d(in_channels = 1,
#                                         out_channels = config.embd_size,
#                                         kernel_size = (97, 8, 8),
#                                         stride = (8, 8, 8))
        
#         self.spatial_position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
#         self.spatial_cls_token = nn.Parameter(torch.zeros(1, 1, config.embd_size))
#         self.spatial_dropout = Dropout(config.dropout_rate)


#     def forward(self, x):

#         x = x.permute(0, 4, 2, 3, 1)
    
#         B = x.shape[0]
#         T = x.shape[-1]
#         C = x.shape[1] 
#         spatial_cls_tokens = self.spatial_cls_token.expand(B, -1, -1)

#         embeddings = []
#         for t in range(T):
#             spatial_t = x[:, :, :, :, t]
#             # spatial_t1 = torch.unsqueeze(spatial_t1, axis = 1)
#             spatial_t = self.patch_embeddings(spatial_t)
#             spatial_t = spatial_t.flatten(2)
#             spatial_t = spatial_t.transpose(-1, -2)
#             spatial_t = torch.cat((spatial_cls_tokens, spatial_t), dim = 1)
#             embeddings_t = spatial_t + self.spatial_position_embeddings
#             embeddings_t = self.spatial_dropout(embeddings_t)
#             embeddings.append(embeddings_t)

#         return embeddings

# class FactorizedVariableEmbdPatch(nn.Module):
#     def __init__(self, config):
#         super(FactorizedVariableEmbdPatch, self).__init__()
#         patch_size = _pair(config.img_size)
#         n_patches  = 97

#         self.patch_embeddings    = Conv2d(in_channels  = 1,
#                                        out_channels    = config.embd_size,
#                                        kernel_size     = patch_size,
#                                        stride          = patch_size)

#         self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
#         self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))
#         self.dropout             = Dropout(config.dropout_rate)

#     def forward(self, x):
#         x = x.permute(0, 4, 2, 3, 1)
#         B = x.shape[0]
#         T = x.shape[-1]
#         cls_tokens = self.cls_token.expand(B, -1, -1)

#         embedings = []

#         for t in range(T): 
#             this_time = x[:, :, :, :, t]
#             time_tokens = None
#             for v in range(this_time.shape[1]):
#                 this_variable = this_time[:, v, :, :]
#                 this_variable = torch.unsqueeze(this_variable, dim=1) 
#                 this_variable = self.patch_embeddings(this_variable)

#                 this_variable = this_variable.flatten(2)
#                 this_variable = this_variable.transpose(-1, -2)
#                 if time_tokens is None: 
#                     time_tokens = this_variable
#                 else:
#                     time_tokens = torch.cat((time_tokens, this_variable), dim=1)

#             time_tokens = torch.cat((cls_tokens, time_tokens), dim=1)
#             time_embeddings = time_tokens + self.position_embeddings
#             time_embeddings = self.dropout(time_embeddings)

#             embedings.append(time_embeddings)

#         return embedings

# class FactorizedTemporalEmbdPatch(nn.Module):
#     def __init__(self, config):
#         super(FactorizedTemporalEmbdPatch, self).__init__()
#         self.temporal_cls_token  = nn.Parameter(torch.zeros(1, 1, config.embd_size))
    
#     def forward(self, x):
#         B = x.shape[0]
#         temporal_cls_tokens = self.temporal_cls_token.expand(B, -1, -1)
#         x = torch.cat((temporal_cls_tokens, x), dim=1)

#         return x

# class PASTT(nn.Module):

#     def __init__(self, config):
#         super(PASTT, self).__init__()

#         img_size     = _pair(config.img_size)
#         patch_size   = _pair(config.patches)
#         temporal_res = 4
#         group_res    = 5
#         n_patches    = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * group_res

#         assert n_patches is not None, ('Number of Patches Can NOT be None!')

#         self.patch_embeddings_G1 = Conv2d(in_channels = len(indices_tensor_G1),
#                                        out_channels   = config.embd_size,
#                                        kernel_size    = patch_size,
#                                        stride         = patch_size)
        
#         self.patch_embeddings_G2 = Conv2d(in_channels = len(indices_tensor_G2),
#                                        out_channels   = config.embd_size,
#                                        kernel_size    = patch_size,
#                                        stride         = patch_size)
        
#         self.patch_embeddings_G3 = Conv2d(in_channels = len(indices_tensor_G3),
#                                        out_channels   = config.embd_size,
#                                        kernel_size    = patch_size,
#                                        stride         = patch_size)
        
#         self.patch_embeddings_G4 = Conv2d(in_channels = len(indices_tensor_G4),
#                                        out_channels   = config.embd_size,
#                                        kernel_size    = patch_size,
#                                        stride         = patch_size)
        
#         self.patch_embeddings_G5 = Conv2d(in_channels = 4,
#                                        out_channels   = config.embd_size,
#                                        kernel_size    = patch_size,
#                                        stride         = patch_size)
        
#         self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
#         self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))
#         self.dropout             = Dropout(config.dropout_rate)

#     def forward(self, x):
#         x = x.permute(0, 4, 2, 3, 1)
    
#         B = x.shape[0]
#         T = x.shape[-1]
#         C = x.shape[1] 
#         cls_tokens = self.cls_token.expand(B, -1, -1)

#         out = []
#         for t in range(T): 
#             this_time = x[:, :, :, :, t]

#             G1 = this_time.index_select(1, indices_tensor_G1)
#             G1_this_time = self.patch_embeddings_G1(G1)
#             G1_this_time = G1_this_time.flatten(2)
#             G1_this_time = G1_this_time.transpose(-1, -2)

#             G2 = this_time.index_select(1, indices_tensor_G2)
#             G2_this_time = self.patch_embeddings_G2(G2)
#             G2_this_time = G2_this_time.flatten(2)
#             G2_this_time = G2_this_time.transpose(-1, -2)

#             G3 = this_time.index_select(1, indices_tensor_G3)
#             G3_this_time = self.patch_embeddings_G3(G3)
#             G3_this_time = G3_this_time.flatten(2)
#             G3_this_time = G3_this_time.transpose(-1, -2)

#             G4 = this_time.index_select(1, indices_tensor_G4)
#             G4_this_time = self.patch_embeddings_G4(G4)
#             G4_this_time = G4_this_time.flatten(2)
#             G4_this_time = G4_this_time.transpose(-1, -2)

#             indices_tensor_G5 = torch.tensor([93, 94, 95, 96]).to(device)
#             G5 = this_time.index_select(1, indices_tensor_G5)
#             G5_this_time = self.patch_embeddings_G5(G5)
#             G5_this_time = G5_this_time.flatten(2)
#             G5_this_time = G5_this_time.transpose(-1, -2)

#             x_stack_this_time = torch.cat((G1_this_time, G2_this_time, G3_this_time, G4_this_time, G5_this_time), dim = 1)

#             x_stack_this_time = torch.cat((cls_tokens, x_stack_this_time), dim = 1)
#             embeddings = x_stack_this_time + self.position_embeddings
#             embeddings = self.dropout(embeddings)

#             out.append(embeddings)

#         return out



# class CrossAttnEncoder(nn.Module):
#     def __init__(self, config):
#         super(Encoder, self).__init__()

#         self.layer = nn.ModuleList()
#         self.encoder_norm = LayerNorm(config.embd_size, eps=1e-6)
#         for _ in range(config.num_layers):
#             layer = CrossAttnBlock(config)
#             self.layer.append(copy.deepcopy(layer))

#     def forward(self, hidden_states):
#         attn_weights = []
#         for layer_block in self.layer:
#             hidden_states, weights = layer_block(hidden_states)
#             attn_weights.append(weights)
#         encoded = self.encoder_norm(hidden_states)

#         return encoded, attn_weights


# class CrossAttnBlock(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         self.ln_1 = nn.LayerNorm(config.embed_dim)
#         self.attn = CrossAttention(config)
#         # self.drop_path = DropPath(config.drop_path) if config.drop_path > 0. else nn.Identity()
#         self.ln_2 = nn.LayerNorm(config.embed_dim)
#         self.mlp = Mlp(in_features = config.embed_dim, 
#                        hidden_features = config.embed_dim*4, 
#                        act_layer = nn.GELU, 
#                        drop = config.Proj_drop)


#     def forward(self, x, x2):
#         h = x
#         x, weights = self.attn(x, x2)
#         x = x + h

#         x = x + self.mlp(self.ln_2(x))
#         return x, weights


# def exists(val):
#     return val is not None

# def default(val, d):
#     return val if exists(val) else d

# class SwiGLU(nn.Module):
#     def forward(self, x):
#         x, gate = x.chunk(2, dim=-1)
#         return F.silu(gate) * x

# class CrossAttention(nn.Module):
#     def __init__(
#         self,
#         config,
#         parallel_ff=False,

#     ):
#         super().__init__()
#         self.heads = config.num_heads
#         dim_head = int(config.embed_dim / config.num_heads)
#         self.scale = dim_head ** -0.5
#         inner_dim = config.num_heads * dim_head
#         x2_dim = default(config.embed_dim, config.embed_dim)

#         self.attn_dropout = nn.Dropout(config.Attn_drop)
#         self.resid_dropout = nn.Dropout(config.Proj_drop)


#         self.norm = nn.LayerNorm(config.embed_dim)
#         self.x2_norm = nn.LayerNorm(x2_dim) #if norm_text else nn.Identity()

#         self.to_q = nn.Linear(config.embed_dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(x2_dim, dim_head * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, config.embed_dim, bias=False)

#         # whether to have parallel feedforward

#         ff_inner_dim = 4 * config.embed_dim

#         self.ff = nn.Sequential(
#             nn.Linear(config.embed_dim, ff_inner_dim * 2, bias=False),
#             SwiGLU(),
#             nn.Linear(ff_inner_dim, config.embed_dim, bias=False)
#         ) if parallel_ff else None

#     def forward(self, x1, x2):
#         """
#         einstein notation
#         b - batch
#         h - heads
#         n, i, j - sequence length (base sequence length, source, target)
#         d - feature dimension
#         """

#         # pre-layernorm, for queries and context
#         x1 = self.norm(x1)
#         x2 = self.x2_norm(x2)
#         # get queries
#         q = self.to_q(x1)
#         q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
#         # scale
#         q = q * self.scale

#         # get key / values
#         k, v = self.to_kv(x2).chunk(2, dim=-1)
#         # query / key similarity
#         sim = torch.einsum('b h i d, b j d -> b h i j', q, k)

#         # attention
#         sim = sim - sim.amax(dim=-1, keepdim=True)
#         weights = sim.softmax(dim=-1)
#         attn = self.attn_dropout(weights)
#         # aggregate
#         out = torch.einsum('b h i j, b j d -> b h i d', attn, v)
#         # merge and combine heads
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         # add parallel feedforward (for multimodal layers)
#         if exists(self.ff):
#             out = out + self.ff(x1)

#         out = self.resid_dropout(out)
#         return out, weights
