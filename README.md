# Distillation Enhanced Social Graph Network (DESIGN).

The supplementary files of paper: Revisiting Graph Neural Network based Social Recommendation.

## Overview

- design is the implementation of our Distillation Enhanced SocIal Graph Network (DESIGN).
- design/config contains the default configuration of our models.
- design/dataset contains four social recommendation datasets used for our experiment.
- design/datautils contains codes that prepare data for the following training/testing.
- design/model contains codes for the GNN-based social recommendation models.

## Requirements
- Python 3.8.8
- pytorch 1.9.0
- Numpy 1.20.1
- dgl 0.6.1

## Usage
Execute the following scripts to train and test DESIGN on the flickr dataset with default hyper-parameters:

```
python main.py --conf_name=design --data_name=yelp --train_model --test_model
```

There are some key options of these scrips:

--conf_name: Choose the corresponding framework for training/testing. By default we use design. If one wants to train each model separately, one can set this value to design_no. Also, One can modify the configuration file in design/configs.

--data_name: Choose the dataset for training/testing. 

--train_model: Perform training process.

--test_model: Perform testing process.