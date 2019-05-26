# SENet
Squeeze and Excitation network implementation with PyTorch.  
[Paper](https://arxiv.org/abs/1709.01507)  

# Requirements
* PyTorch 1.1.0
* torchvision 0.3.0

# MNIST
* Network architecture
	* Blocks : 10
	* Filters : 128
* Hyper-parameters
	* Optimizer : NAG
	* Momentum : 0.9
	* Weight decay : 1e-4
	* Learning rate : 1e-1 (1 to 2 epochs)
* Result (After 2 epochs)
	* Train Accuracy : 98.54%
	* Test Accuracy : 98.90%
  
[Download pretrained-model](https://github.com/JYPark09/SENet-PyTorch/releases/tag/MNIST)

# CIFAR-10  
* Network architecture
	* Blocks : 10
	* Filters : 128
* Hyper-parameters
	* Optimizer : NAG
	* Momentum : 0.9
	* Weight decay : 1e-4
	* Learning rate : 1e-1 (1 to 31 epochs)
* Result (After 31 epochs)
	* Train Accuracy : 93.1040%
	* Test Accuracy : 86.9700%
  
[Download pretrained-model](https://github.com/JYPark09/SENet-PyTorch/releases/tag/CIFAR-10)
