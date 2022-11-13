# edNet
A Simple Parallelized Neural Network Framework in C++ and Python! 

This is a Neural Network framework that covers core mechanics of traditional neural network techniques and it's approach from a high performance computing perspective. The C++ is a header only implementation, which allows for easy incorporation into existing projects. 

### Features:
- Layers:
  - Dense
- Activations:
  - SoftMax
  - ReLU
- Losses:
  - Categorical Cross Entropy
  - Sparse Categorical Crossentropy
- Optimizers:
  - SGD
  
## Usage
```cpp
// Setup some training data and labels
Matrix<double> x_train(train_size, 2);
Matrix<double> y_train(train_size, 2);

// Define the network
Dense<double> layer1(2, 16);
ReLU<double> layer2;
Dense<double> layer3(16, 2);
SoftmaxCrossEntropy<double> layer4;
optimizer::SGD<double> sgd(1.0, 0.001);

// Complete forward pass
Matrix<double> out1 = layer1.forward(x_train);
Matrix<double> out2 = layer2.forward(out1);
Matrix<double> out3 = layer3.forward(out2);
Matrix<double> out4 = layer4.forward(out3, y_train);

// Calculate loss and metric
double loss = layer4.get_loss();
double acc = metric::accuracy(y_train, out4);
```

## Installation
### C++
Requires CUDA version 10.0, C++11 or greater and appropriate compiler
1) Clone repo
2) Cd to C++ dir
3) Run "make" in terminal
4) Run "./driver 1000" for demonstration of the serial implementation
5) Run "./driver 1000" for demonstration of the parallel implementation

### Python
1) Clone repo
2) Install dependacies (numpy)
3) Run driver.py for demonstration of the serial implementation

## HPC Analysis
This codebase is paired with a report on the comparison of the serial and parrallel implementation. If you wish to read it, see this [link](https://1drv.ms/b/s!AvEmHRWzO1jBj4Y_j-OCCbseaACtQw?e=EznURm7) to download a copy. 

