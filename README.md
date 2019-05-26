# SENet-PyTorch
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
Squeeze and Excitation network implementation with PyTorch.  
[Paper](https://arxiv.org/abs/1709.01507)  

## Requirements
* Python 3.x
* PyTorch 1.1.0
* torchvision 0.3.0

## Additional Feature
In original paper, SE block just scale. But, I added bias to SE block. It works better than original architecture.

## Quick Start
First, clone the code:  
```
git clone https://github.com/JYPark09/SENet-PyTorch.git
cd SENet-PyTorch
```
  
### Training model
You can train models with already written training codes(cifar10.py or mnist.py).
  
Before training, you can adjust hyper parameters. Just change values that exist in cifar10.py or mnist.py.
```
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 1e-1
WEIGHT_DECAY = 1e-4
```

### Using pretrained models
You can download pretrained weight in [here](https://github.com/JYPark09/SENet-PyTorch/releases). And if you want to use pretrained model, just write the code below.  

* CIFAR-10
```
net = Network(3, 128, 10, 10).cuda()
net.load_state_dict(torch.load(<CHECKPOINT_FILE_PATH>))
```
* MNIST
```
net = Network(1, 128, 10, 10).cuda()
net.load_state_dict(torch.load(<CHECKPOINT_FILE_PATH>))
```

## Issue
### I have no gpus, but I want to use pretrained model.  
* You can load pretrained model by adding map_location at torch.load.  

(example for CIFAR-10)
```
net = Network(3, 128, 10, 10)
net.load_state_dict(torch.load(<CHECKPOINT_FILE_PATH>, map_location='cpu'))
```

### CUDA Memory Allocation Fail  
* reduce BATCH_SIZE
* make smaller network architecture.

## Contact
You can contact me via e-mail (jyp10987 at gmail.com) or github issue.
