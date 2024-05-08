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
device = torch.device("cuda:1")

from models.layers_ours import *
from models.weight_init import trunc_normal_
from models.layer_helpers import to_2tuple

seed = 1987
torch.manual_seed(seed) #important 
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

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
        num_patches  = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) # in_channels # 

        self.num_patches = num_patches
        self.img_size = img_size
        self.patch_size = patch_size

        assert num_patches is not None, ('Number of Patches Can NOT be None!')

        self.proj    = Conv2d(in_channels = config.in_channels,
                                        out_channels = config.embd_size,
                                        kernel_size  = patch_size,
                                        stride       = patch_size)
            

    def forward(self, x):

        B = x.shape[0]
        x = torch.reshape(x, [B,  x.shape[1]*x.shape[4], x.shape[2], x.shape[3]])
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(-1, -2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = torch.reshape(x, [x.shape[0],  x.shape[1]*x.shape[4], x.shape[2], x.shape[3]])
        B, C, H, W = x.shape

        #look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1, 2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)
    
class PIT(nn.Module):

    def __init__(self, config):
        super(PIT, self).__init__()

        img_size     = _pair(config.img_size)
        patch_size   = _pair(config.patches)
        group_res    = 5
        num_patches    = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * group_res

        assert num_patches is not None, ('Number of Patches Can NOT be None!')
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.indices, self.tensors = engine.return_variable_index(category = 'v2')

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
    

    def forward(self, x):
        x = x.permute(0, 4, 2, 3, 1)
        B = x.shape[0]
        x = torch.reshape(x, [B,  x.shape[1]*x.shape[4], x.shape[2], x.shape[3]])

        G1 = x.index_select(1, self.tensors[0])
        G1_this_time = self.patch_embeddings_G1(G1).flatten(2).transpose(1, 2)

        G2 = x.index_select(1, self.tensors[1])
        G2_this_time = self.patch_embeddings_G2(G2).flatten(2).transpose(1, 2)

        G3 = x.index_select(1, self.tensors[2])
        G3_this_time = self.patch_embeddings_G3(G3).flatten(2).transpose(1, 2)

        G4 = x.index_select(1, self.tensors[3])
        G4_this_time = self.patch_embeddings_G4(G4).flatten(2).transpose(1, 2)

        G5 = x.index_select(1, self.tensors[4])
        G5_this_time = self.patch_embeddings_G5(G5).flatten(2).transpose(1, 2)


        x_stack_this_time = torch.cat((G1_this_time, G2_this_time, G3_this_time, G4_this_time, G5_this_time), dim = 1)

        return x_stack_this_time
    
    def relprop(self, cam, **kwargs):
        # create 5 groups: 
        cam_g1 = cam.index_select(1, self.tensors[0])
        cam_g1 = cam_g1.transpose(1, 2)
        cam_g1 = cam_g1.reshape(cam_g1.shape[0], cam_g1.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_g1_proj = self.proj.relprop(cam_g1, **kwargs)

        cam_g2 = cam.index_select(1, self.tensors[1])
        cam_g2 = cam_g2.transpose(1, 2)
        cam_g2 = cam_g2.reshape(cam_g2.shape[0], cam_g2.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_g2_proj = self.proj.relprop(cam_g2, **kwargs)

        cam_g3 = cam.index_select(1, self.tensors[2])
        cam_g3 = cam_g3.transpose(1, 2)
        cam_g3 = cam_g3.reshape(cam_g3.shape[0], cam_g3.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_g3_proj = self.proj.relprop(cam_g3, **kwargs)

        cam_g4 = cam.index_select(1, self.tensors[3])
        cam_g4 = cam_g4.transpose(1, 2)
        cam_g4 = cam_g4.reshape(cam_g4.shape[0], cam_g4.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_g4_proj = self.proj.relprop(cam_g4, **kwargs)

        cam_g5 = cam.index_select(1, self.tensors[4])
        cam_g5 = cam_g5.transpose(1, 2)
        cam_g5 = cam_g5.reshape(cam_g5.shape[0], cam_g5.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_g5_proj = self.proj.relprop(cam_g5, **kwargs)

        return cam_g1_proj, cam_g2_proj, cam_g3_proj, cam_g4_proj, cam_g5_proj

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
        num_patches    = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * temporal_res # in_channels # 
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        assert num_patches is not None, ('Number of Patches Can NOT be None!')

        self.patch_embeddings = Conv2d(in_channels = config.in_channels,
                                    out_channels   = config.embd_size,
                                    kernel_size    = patch_size,
                                    stride         = patch_size)



    def forward(self, x):
        x = x.permute(0, 4, 2, 3, 1)
    
        B = x.shape[0]
        T = x.shape[-1]
        C = x.shape[1] 

        x_stack = None
        for t in range(T): 
            this_time = x[:, :, :, :, t]
            this_time = self.patch_embeddings(this_time)

            this_time = this_time.flatten(2)
            this_time = this_time.transpose(1, 2)

            if x_stack is None: 
                x_stack = this_time
            else:
                x_stack = torch.cat((x_stack, this_time), dim = 1)

        return x_stack
    
    def relprop(self, cam, **kwargs):
        cam = cam.permute(0, 4, 2, 3, 1)
        T = cam.shape[-1]
        cam_stack = None

        for t in range(T):
            cam = cam.transpose(1, 2)
            cam = cam.reshape(cam.shape[0], cam.shape[1],
                        (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
            cam = self.proj.relprop(cam, **kwargs)

            if cam_stack is None: 
                cam_stack = cam
            else:
                cam_stack = torch.cat((cam_stack, cam), dim = 1)

        return cam_stack

class SVT(nn.Module):

    def __init__(self, config):
        super(SVT, self).__init__()

        img_size     = _pair(config.img_size)
        patch_size   = _pair(config.patches)
        variable_cat = 9
        num_patches    = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * variable_cat
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        assert num_patches is not None, ('Number of Patches Can NOT be None!')


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

    def forward(self, x):
        x = x.permute(0, 4, 2, 3, 1)
        B = x.shape[0]
        x = torch.reshape(x, [B,  x.shape[1]*x.shape[4], x.shape[2], x.shape[3]])


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

        return x_stack_this_time
   
    def relprop(self, cam, **kwargs):
        # create 5 groups: 
        cam_temp = cam.index_select(1, self.tensors[0])
        cam_temp = cam_temp.transpose(1, 2)
        cam_temp = cam_temp.reshape(cam_temp.shape[0], cam_temp.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_temp_proj = self.proj.relprop(cam_temp, **kwargs)

        cam_u10 = cam.index_select(1, self.tensors[1])
        cam_u10 = cam_u10.transpose(1, 2)
        cam_u10 = cam_u10.reshape(cam_u10.shape[0], cam_u10.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_u10_proj = self.proj.relprop(cam_u10, **kwargs)

        cam_v10 = cam.index_select(1, self.tensors[2])
        cam_v10 = cam_v10.transpose(1, 2)
        cam_v10 = cam_v10.reshape(cam_v10.shape[0], cam_v10.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_v10_proj = self.proj.relprop(cam_v10, **kwargs)

        cam_TKE = cam.index_select(1, self.tensors[3])
        cam_TKE = cam_TKE.transpose(1, 2)
        cam_TKE = cam_TKE.reshape(cam_TKE.shape[0], cam_TKE.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_TKE_proj = self.proj.relprop(cam_TKE, **kwargs)

        cam_Q = cam.index_select(1, self.tensors[4])
        cam_Q = cam_Q.transpose(1, 2)
        cam_Q = cam_Q.reshape(cam_Q.shape[0], cam_Q.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_Q_proj = self.proj.relprop(cam_Q, **kwargs)

        cam_RH = cam.index_select(1, self.tensors[4])
        cam_RH = cam_RH.transpose(1, 2)
        cam_RH = cam_RH.reshape(cam_RH.shape[0], cam_RH.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_RH_proj = self.proj.relprop(cam_RH, **kwargs)

        cam_VVEL = cam.index_select(1, self.tensors[4])
        cam_VVEL = cam_VVEL.transpose(1, 2)
        cam_VVEL = cam_VVEL.reshape(cam_VVEL.shape[0], cam_VVEL.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_VVEL_proj = self.proj.relprop(cam_VVEL, **kwargs)

        cam_VIS = cam.index_select(1, self.tensors[4])
        cam_VIS = cam_VIS.transpose(1, 2)
        cam_VIS = cam_VIS.reshape(cam_VIS.shape[0], cam_VIS.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_VIS_proj = self.proj.relprop(cam_VIS, **kwargs)

        cam_SST = cam.index_select(1, self.tensors[4])
        cam_SST = cam_SST.transpose(1, 2)
        cam_SST = cam_SST.reshape(cam_SST.shape[0], cam_SST.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam_SST_proj = self.proj.relprop(cam_SST, **kwargs)

        return cam_temp_proj, cam_u10_proj, cam_v10_proj, cam_TKE_proj, cam_Q_proj, cam_RH_proj, cam_VVEL_proj, cam_VIS_proj, cam_SST_proj
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x, g):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        if g: 
            attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x, g):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2), g)])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam

class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, config,):
        super().__init__()
        self.num_classes = 2
        self.num_features = config.embd_size  # num_features for consistency with other models

        # VVT
        if config.embd_type == 'VVT':
            self.patch_embed = PatchEmbed(img_size = config.img_size, 
                                        patch_size = config.patches, 
                                        in_chans=config.in_channels, 
                                        embed_dim=config.embd_size)
            
        elif config.embd_type == 'PIT':
            self.patch_embed = PIT(config)

        elif config.embd_type == 'STT':
            self.patch_embed = STTEmbdPatch(config)

        elif config.embd_type == 'SVT':
            self.patch_embed = SVT(config)

        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embd_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embd_size))

        self.blocks = nn.ModuleList([
            Block(
                dim = config.embd_size, 
                num_heads = config.num_heads, 
                mlp_ratio = 4, 
                qkv_bias = False,
                drop = config.dropout_rate, 
                attn_drop = config.attention_dropout_rate)

            for i in range(config.num_layers)])

        self.norm = LayerNorm(config.embd_size)

        mlp_head=False,
        if mlp_head:
            # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
            self.head = Mlp(config.embd_size, int(config.embd_size * 4), 2)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            self.head = Linear(config.embd_size, 2)

        # FIXME not quite sure what the proper weight init is supposed to be,
        # normal / trunc normal w/ std == .02 similar to other Bert like transformers
        trunc_normal_(self.pos_embed, std=.02)  # embeddings same as weights?
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.pool = IndexSelect()
        self.add = Add()

        self.inp_grad = None

    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, g = True):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        if g: 
            x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x, g)

        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x

    def relprop(self, cam=None,method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam
        
        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam
            
        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

