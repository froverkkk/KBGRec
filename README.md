# KAN-Infused Social Recommendation: A Contrastive Graph LearningApproach with Bidirectional Feature Fusion(KBGRec)

The supplementary files of paper: Revisiting Graph Neural Network based Social Recommendation.

## Requirements
- Python 3.8.8
- pytorch 1.9.0
- Numpy 1.20.1
- dgl 0.6.1

## Usage
Execute the following scripts to train and test KBGRec on the Yelp dataset with default hyper-parameters:

```
python main.py --conf_name=design --data_name=yelp --train_model --test_model
```

There are some key options of these scrips:

--conf_name: Choose the corresponding framework for training/testing. By default we use KBGRec. One can modify the configuration file in KBGRec/configs.

--data_name: Choose the dataset for training/testing. 

--train_model: Perform training process.

--test_model: Perform testing process.