
import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader




    
class Transformer1d(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples, n_classes)
        
    Pararmetes:
        
    """

    def __init__(self, d_model, nhead, dim_feedforward, n_classes, dropout, activation, verbose=False):
        super(Transformer1d, self).__init__()

        self.d_model         = d_model
        self.nhead           = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout         = dropout
        self.activation      = activation
        self.n_classes       = n_classes
        self.verbose         = verbose
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, 
                                                    nhead   = self.nhead, 
                                                    dim_feedforward = self.dim_feedforward, 
                                                    dropout = self.dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dense   = nn.Linear(self.d_model, self.n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.permute(1, 0, 2)
        out = self.transformer_encoder(x)
        out = out.mean(0)
        out = self.dense(out)
        out_softmax = self.softmax(out)
        out_sigmoid = self.sigmoid(out)
    
        return out, out_softmax, out_sigmoid  