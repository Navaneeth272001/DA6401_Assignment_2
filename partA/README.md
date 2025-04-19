## Training a CNN

### This folder contains the solutions to part A of the assignment.

The assignment2A.ipynb file contains the code for building a custion CNN network. 

Note: There are actually two networks in the code, just to differentiate the answers to the question. For training the network, first cell and the 4th cell are necessary.

The training will be logged to wandb project. 

Use wandb login before running the notebook. 

The config parameters used for finetuning are as follows:

```python

config_params = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'epochs': {'values': [10, 15, 20]},
        'loss': {'values': ['cross_entropy']},
        'm': {'values': [32, 64, 128]},
        'k': {'values': [[3, 3, 3, 3, 3], [5, 5, 5, 5, 5], [2, 2, 2, 2, 2]]},
        'filter_org': {'values': ['same', 'double', 'half']},
        'optim_algo': {'values': ['sgd','momentum', 'adam']},
        'batch_size': {'values': [32, 64, 128]},
        'lr': {'values': [1e-2, 1e-3]},
        'data_aug': {'values': ['yes', 'no']},
        'batch_norm': {'values': ['yes', 'no']},
        'weight_init': {'values': ['random', 'xavier']},
        'dropout': {'values': [0.2, 0.3]},
        'activation': {'values': ['relu', 'gelu', 'silu', 'mish']},
    }
}

# you may change the project as required
sweep_id = wandb.sweep(config_params, project='Assignment2')
wandb.agent(sweep_id, function=train_model, count=30)

```
In the config files, m represents the number of filters, k represents the filter sizes.
### Dataset setup:

The dataset used is iNaturalist12K. The structure of the dataset should be organised as follows:

dataset/
└── inaturalist_12K/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    └── val/
        ├── class_1/
        ├── class_2/
        └── ...

- Make sure that the dataset is not kept inside the partA folder, but in the main folder as the dataset is needed for part B as well.
- ConvNNconfig auto-calculates flattened dimensions from convolutional blocks.
- The training data is split internally: 80% for training, 20% for validation.

### Dependencies

Python 3.10.10 (anything above 3.8.)
PyTorch
torchvision
wandb
matplotlib
numpy


