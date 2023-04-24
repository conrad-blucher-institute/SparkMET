import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
import copy
import logging
import math
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import re
import seaborn as sns
from torch import Tensor
from models import configs

#======================================================================================================#
#====================================     Transformer 1D     ==========================================#
#======================================================================================================#

class Transformer1d(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples, n_classes)
        
    Pararmetes:

    """

    #def __init__(self, d_model, nhead, dim_feedforward, n_classes, dropout, activation, verbose=False):
    def __init__(self, model_config_dict):
        super(Transformer1d, self).__init__()

        self.d_model         = model_config_dict.embed_dim
        self.nhead           = model_config_dict.transformer['num_heads']
        self.dropout         = model_config_dict.transformer['dropout_rate']
        self.activation      = model_config_dict.transformer['activation']
        self.n_classes       = model_config_dict.transformer['num_class']
        self.verbose         = False
        
        encoder_layer = nn.TransformerEncoderLayer(d_model          = self.d_model, 
                                                    nhead           = self.nhead, 
                                                    dim_feedforward = 512, 
                                                    dropout = self.dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dense               = nn.Linear(self.d_model, self.n_classes)
        self.softmax             = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = x.permute(1, 0, 2)
        out = self.transformer_encoder(x)
        out = out.mean(0)
        out = self.dense(out)
        out_softmax = self.softmax(out)
        return out, out_softmax 

#======================================================================================================#
#================================ 1D Transformer from scratch =========================================#
#======================================================================================================#
## https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

class Embedding1D(nn.Module):
    def __init__(self, vector_size, embed_dim):
        """
        Args:
            vector_size: size of input vector
            embed_dim: dimension of embeddings
        """
        super(Embedding1D, self).__init__()
        self.embed = nn.Embedding(vector_size, embed_dim)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out

class PositionalEmbedding1D(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding1D, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
      
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x

'''def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)

class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        print(f"{self.q(query).shape}, {self.k(key).shape}, {self.v(value).shape}")
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))
    
class MultiHeadAttention1D(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        print(f"{query.shape} | {key.shape} | {value.shape}")
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )'''

class MultiHeadAttention1D(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention1D, self).__init__()

        #self.embed_dim       = embed_dim # 12 dim
        #self.n_heads         = n_heads   # 8
        #self.single_head_dim = int(self.embed_dim / self.n_heads)   #512/8 = 64  . each key,query, value will be of 64d
       
        #key,query and value matrixes    #64 x 64   
        self.query_matrix = nn.Linear(64, 64, bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix   = nn.Linear(64, 64, bias=False)
        self.value_matrix = nn.Linear(64, 64, bias=False)
        self.out          = nn.Linear(512, 512) 

    def forward(self, key, query, value, mask=None):    #batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        
        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        
        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        # query dimension can change in decoder during inference. 
        # so we cant take general seq_length
        seq_length_query = query.size(1)



        # 32x10x512
        key   = key.reshape(32, 744, 8, 64)   # batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.reshape(32, 744, 8, 64) # (32x10x8x64)
        value = value.reshape(32, 744, 8, 64) # (32x10x8x64)32, seq_length, self.n_heads, self.single_head_dim

        key   = self.key_matrix(key)       # (32x10x8x64)
        query = self.query_matrix(query)   
        value = self.value_matrix(value)

        print(f"{key.shape} | {query.shape} | {value.shape}")

        query = query.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        key = key.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        value = value.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
       
        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = key.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(query, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
      
        
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)
 
        #mutiply with value matrix
        scores = torch.matmul(scores, value)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64) 
        
        #concatenated output
        scores = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        
        scores = self.out(scores) #(32,10,512) -> (32,10,512)
       
        return scores

class TransformerBlock1D(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock1D, self).__init__()
        
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        #dim_ = max(embed_dim // n_heads, 1)
        #self.attention = MultiHeadAttention1D(num_heads = n_heads, dim_in = embed_dim, dim_q = 64, dim_k = 64) 
        self.attention = MultiHeadAttention1D(embed_dim=512, n_heads=8)
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self,key, query, value):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block
        
        """

        attention_out          = self.attention(key,query,value)  #32x10x512
        
        attention_residual_out = attention_out + value  #32x10x512
        norm1_out              = self.dropout1(self.norm1(attention_residual_out)) #32x10x512

        feed_fwd_out           = self.feed_forward(norm1_out) #32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out  = feed_fwd_out + norm1_out #32x10x512
        norm2_out              = self.dropout2(self.norm2(feed_fwd_residual_out)) #32x10x512

        return norm2_out

class TransformerEncoder1D(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, seq_len, variable_len, embed_dim, num_layers=4, expansion_factor=4, n_heads=8):
        super(TransformerEncoder1D, self).__init__()
        
        self.embedding_layer    = Embedding1D(variable_len, embed_dim)
        self.positional_encoder = PositionalEmbedding1D(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock1D(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
    
    def forward(self, x):
        embed_out = self.embedding_layer(x)[0, ...]
        out       = self.positional_encoder(embed_out)

        for layer in self.layers:
            out = layer(out, out, out)

        return out  #32x10x512
    
class Transformer1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples, n_classes)
        
    Pararmetes:

    """

    #def __init__(self, d_model, nhead, dim_feedforward, n_classes, dropout, activation, verbose=False):
    def __init__(self, model_config_dict):
        super(Transformer1D, self).__init__()

        self.variable_len    = model_config_dict.variable_len
        self.embed_dim       = model_config_dict.embed_dim
        self.nhead           = model_config_dict.transformer['num_heads']
        self.num_layers      = model_config_dict.transformer['num_layers']
        self.dropout         = model_config_dict.transformer['dropout_rate']
        self.activation      = model_config_dict.transformer['activation']
        self.n_classes       = model_config_dict.transformer['num_class']
        self.verbose         = False
        
        
        self.transformer_encoder = TransformerEncoder1D(self.variable_len, 
                                                         self.variable_len, 
                                                         self.embed_dim, 
                                                         num_layers = self.num_layers, 
                                                         expansion_factor=4, 
                                                         n_heads=self.nhead)
        
        self.dense               = nn.Linear(self.embed_dim, self.n_classes)
        self.softmax             = nn.LogSoftmax(dim=1)
        

    def forward(self, x):

        x   = x.permute(1, 0, 2)
        out = self.transformer_encoder(x)
        out = out.mean(0)
        out = self.dense(out)
        out_softmax = self.softmax(out)
        return out, out_softmax 
#======================================================================================================#
#==================================== Vision Transformer ==============================================#
#======================================================================================================#

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.embd_size / self.num_attention_heads)
        self.all_head_size       = self.num_attention_heads * self.attention_head_size

        self.query  = Linear(config.embd_size, self.all_head_size)
        self.key    = Linear(config.embd_size, self.all_head_size)
        self.value  = Linear(config.embd_size, self.all_head_size)

        self.out          = Linear(config.embd_size, config.embd_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer   = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer   = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs  = self.softmax(attention_scores)
        weights          = attention_probs if self.vis else None
        attention_probs  = self.attn_dropout(attention_probs)

        context_layer    = torch.matmul(attention_probs, value_layer)
        context_layer    = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer    = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.embd_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.embd_size)
        self.act_fn = ACT2FN["gelu"]
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

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size):
        super(Embeddings, self).__init__()

        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) # in_channels # 

        self.patch_embeddings    = Conv2d(in_channels  = config.in_channels,
                                       out_channels = config.embd_size,
                                       kernel_size  = patch_size,
                                       stride       = patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))

        self.dropout    = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)


        x = self.patch_embeddings(x)

        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.embd_size
        self.attention_norm = LayerNorm(config.embd_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.embd_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)


    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        
        #x = self.softmax(x)
        return x, weights



class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.embd_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)

            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)

        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder    = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output      = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=32, num_classes=2, zero_head=False, vis=True):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head   = zero_head
        self.classifier  = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head        = Linear(config.embd_size, num_classes)
        self.softmax     = nn.LogSoftmax(dim=1)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])
        pred   = self.softmax(logits)

        return attn_weights, pred 

#======================================================================================================#
#==================================== Vision Transformer V2 ===========================================#
#======================================================================================================#


class Attention_V2(nn.Module):
    def __init__(self, config, vis):
        super(Attention_V2, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.embd_size / self.num_attention_heads)
        self.all_head_size       = self.num_attention_heads * self.attention_head_size

        self.query  = Linear(config.embd_size, self.all_head_size)
        self.key    = Linear(config.embd_size, self.all_head_size)
        self.value  = Linear(config.embd_size, self.all_head_size)

        self.out          = Linear(config.embd_size, config.embd_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer   = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer   = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs  = self.softmax(attention_scores)
        weights          = attention_probs if self.vis else None
        attention_probs  = self.attn_dropout(attention_probs)

        context_layer    = torch.matmul(attention_probs, value_layer)
        context_layer    = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer    = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights

class Mlp_V2(nn.Module):
    def __init__(self, config):
        super(Mlp_V2, self).__init__()
        self.fc1 = Linear(config.embd_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.embd_size)
        self.act_fn = ACT2FN["gelu"]
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

class Embeddings_V2(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size):
        super(Embeddings_V2, self).__init__()

        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        n_patches  = 388

        self.patch_embeddings    = Conv2d(in_channels  = 1,
                                       out_channels    = config.embd_size,
                                       kernel_size     = 32,
                                       stride          = 32)
        #self.patch_embeddings    = nn.Sequential(nn.Linear(1, config.embd_size)
        #)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.embd_size))
        self.cls_token           = nn.Parameter(torch.zeros(1, 1, config.embd_size))

        self.dropout    = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

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

class Block_V2(nn.Module):
    def __init__(self, config, vis):
        super(Block_V2, self).__init__()
        self.hidden_size = config.embd_size
        self.attention_norm = LayerNorm(config.embd_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.embd_size, eps=1e-6)
        self.ffn = Mlp_V2(config)
        self.attn = Attention_V2(config, vis)


    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        
        #x = self.softmax(x)
        return x, weights


class Encoder_V2(nn.Module):
    def __init__(self, config, vis):
        super(Encoder_V2, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.embd_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block_V2(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)

            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)

        return encoded, attn_weights

class Transformer_V2(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer_V2, self).__init__()
        self.embeddings = Embeddings_V2(config, img_size=img_size)
        self.encoder    = Encoder_V2(config, vis)

    def forward(self, input_ids):
        embedding_output      = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

class VisionTransformer_V2(nn.Module):

    def __init__(self, config, img_size=32, num_classes=2, zero_head=False, vis=True):
        super(VisionTransformer_V2, self).__init__()
        self.num_classes = num_classes
        self.zero_head   = zero_head
        self.classifier  = config.classifier

        self.transformer = Transformer_V2(config, img_size, vis)
        self.head        = Linear(config.embd_size, num_classes)
        self.softmax     = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x, attn_weights = self.transformer(x)

        logits = self.head(x[:, 0])
        pred   = self.softmax(logits)

        return x, attn_weights, pred 
    





    




