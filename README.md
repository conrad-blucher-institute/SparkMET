# SparkMET: A Deep Learning Framework for Fog Forecasting
# SparkMET: A Deep Learning Framework for Fog Forecasting

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

SparkMET is a deep learning framework tailored for fog forecasting. It provides a robust and customizable pipeline for training and evaluating deep learning models using 4D input data structures.

## Table of Contents
- [Introduction](#introduction)
- [ViT Models](#vit-models)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Compilation, Training, and Testing](#model-compilation-training-and-testing)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [To Do List](#to-do-list)

## Introduction


## ViT Models
- Vanilla Vision Tokenization (VVT)
- Uniform Variable Tokenization (UVT)
- Spatial-Temporal Tokenization (STT)
- Spatial-Variable Tokenization (SVT)
- Physic-Informed Tokenization (PIT)

## Getting Started

### Prerequisites


### Installation


## Usage

### Data Preparation
Prepare the data loader with the following structure:

```python
sample = {
    "input": timeseries_predictors_matrix,  # Shape: [B, T, H, W, C] e.g., [32, 4, 32, 32, 96]
    "onehotlabel": onehotlabel,
    "label_class": label,
    "date_time": date_time,
    "round_time": round_time,
    "date_cycletime": date_cycletime,
    "vis": visibility
}
```

### Model Compilation, Training, and Testing
Configure, compile, train, and test the model using the following example:

```from SparkMet import SparkMET, ModelConfig

configs = ModelConfig(
    img_size=32,
    in_channel=388,  # Modify based on embd_type
    in_time=4,
    embd_size=1024,
    mlp_size=512,
    num_heads=8,
    dropout=0.1,
    num_layers=12,
    embd_type='VVT',  # Options: 'UVT', 'SVT', 'PIT-V1', 'PIT-V2'
    conv_type='2d'  # Options: '2d', '3d'
).return_config()

spark = SparkMET(configs, SaveDir="path/to/save", Exp_Name="ExperimentName")

model, optimizer, loss_func = spark.compile(
    optimizer="adam",
    loss="mse",
    lr=0.001,
    wd=0.01
)

model, loss_stats = spark.train(
    model,
    optimizer,
    loss_func,
    data_loader_training,
    data_loader_validate,
    epochs=50,
    early_stop_tolerance=10
)

outputs = spark.predict(
    model,
    data_loader_training,
    data_loader_testing
)
```

## To Do List
- [ ] Update fog-dataloader for bigger area and different data format
- [ ] expand EDA for more visualization 


```bash
python run.py --exp_name test --embd_type VVT --batch_size 128 --embd_size 1024 --num_heads 8 --num_layers 8 --lr 0.001 --wd 0.01 --epochs 2








