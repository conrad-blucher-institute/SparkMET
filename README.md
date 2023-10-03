# SparkMET: A Deep Learning Framework for Fog Forecasting

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

SparkMET is a deep learning framework designed for fog forecasting. It provides a flexible and customizable pipeline for training and evaluating deep learning models for 4D input data strucrture.

**Table of Contents**
- [Introduction](#introduction)
- [ViT Models](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training a Model](#training-a-model)
  - [Inference](#inference)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [To Do List](#todo-list)

## Introduction

[Provide a brief introduction to your project and explain its purpose. Describe what problem it solves or what task it performs.]

## ViT Models

- [Vanilla Vision Tokenization (VVT)]
- [Uniform Variable Tokenization (UVT)]
- [Spatio-Temporal Factorized Self-Attention (STFSA)]
- [SparkMET]

## Getting Started


### Prerequisites


### Installation


## Usage


### Training a Model

### Inference

## To Do List
- [x] Update fog-dataloader for bigger area and different data format
- [ ] merge our model to the multiview_vit module
- [ ] expand EDA for more visualization 


```bash
python run.py --exp_name test --batch_size 128 --embd_size 1024 --num_heads 8 --num_layers 8 --lr 0.001 --wd 0.01 --epochs 2








