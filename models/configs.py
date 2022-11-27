
import os
import pandas as pd
import torch 
from pathlib import Path
from src import dataloader
import ml_collections


DEFAULT_IMAGE_DIR_NAME    = '/data1/fog-data/fog-maps/'
DEFAULT_TARGET_DIR_NAME   = '/data1/fog/Dataset/TARGET'

#NAM_coords = {'P1':[23, 6],
#         'P2':[22, 8],
#         'P3':[21, 10],
#         'P4':[19, 12]} 
NAM_coords = {'P1':[23, 6],
         'P2':[22, 8]} 

year_information          = {'2009':['20090101', '20091231'],
                            '2010':['20100101', '20101231'],
                            '2011':['20110101', '20111231'],
                            '2012':['20120101', '20121231'],
                            '2013':['20130101', '20131231'],
                            '2014':['20140101', '20141231'],
                            '2015':['20150101', '20151231'],
                            '2016':['20160101', '20161231'],
                            '2017':['20170101', '20171231'],
                            '2018':['20180101', '20181231'],
                            '2019':['20190101', '20191231'],
                            '2020':['20200101', '20201231']}


data_split_dict = {'train': ['2013', '2014', '2015', '2016', '2017'], 
                    'valid': ['2009', '2010', '2011'], 
                    'test': ['2018', '2019', '2020']},



config_dictionary_1D = dict(random_state=1001,
                            num_classes=2,
                            lr= 1e-5,
                            weight_decay= 1e-1, 
                            dropout= 0.4,
                            nhead= 8, 
                            dim_feedforward= 512,
                            batch_size= 32,
                            early_stop_tolerance= 50,
                            epochs= 200, 
                            )


def get_train_hyperparameter_config():
    config = ml_collections.ConfigDict()
    config.batch_size = 32
    config.lr = 1e-4
    config.wd = 1e-2
    config.early_stop = 50
    config.epochs = 200
    return config


def get_data_config():
    """Returns input data configuration."""
    config = ml_collections.ConfigDict()
    config.input_path = DEFAULT_IMAGE_DIR_NAME
    config.target_path = DEFAULT_TARGET_DIR_NAME,
    config.start_date = year_information['2009'][0]
    config.finish_date = year_information['2020'][1]
    config.data_split_dict = data_split_dict,
    config.data_straucture = '1D',
    config.lead_time_pred = 24
    config.vis_threshold = 1
    config.points_coords = NAM_coords
    config.predictor_names = dataloader.NETCDF_PREDICTOR_NAMES['All']
    return config


def get_1D_config():
    """Returns the 1D configuration."""
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 512
    config.transformer.num_heads = 8
    config.transformer.num_layers = 6
    config.transformer.dropout_rate = 0.3
    config.transformer.activation = 'relu'
    config.transformer.num_class = 2
    return config
 

def get_ViT_config():
    """Returns the ViT configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (32, 32)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 512
    config.transformer.num_heads = 8
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.3
    config.classifier = 'token'
    config.representation_size = None
    return config




                               
                
                
                
                
