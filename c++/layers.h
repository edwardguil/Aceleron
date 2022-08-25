#ifndef LAYERS_H
#define LAYERS_H
#include "matrix.h"
#include "losses.h"

class Layer {
public:
    virtual matrix::Matrix<float> forward(matrix::Matrix<float>) = 0;
    virtual matrix::Matrix<float> backward(matrix::Matrix<float>) = 0;
};


class Dense : public Layer { 
    matrix::Matrix<float> weights; 
    matrix::Matrix<float> biases;
    matrix::Matrix<float> dweights;
    matrix::Matrix<float> dbiases;
    matrix::Matrix<float> inputs;
	
public:
    Dense(int n_inputs, int n_neurons);

    void randomize_weights();

    matrix::Matrix<float> forward(matrix::Matrix<float> input);
    matrix::Matrix<float> backward(matrix::Matrix<float> dinput);
};

class ReLU : public Layer {    
    matrix::Matrix<float> inputs;
public:
    ReLU(void);
    matrix::Matrix<float> forward(matrix::Matrix<float> input);
    matrix::Matrix<float> backward(matrix::Matrix<float> dinput);

};

class Softmax : public Layer {
    matrix::Matrix<float> inputs;
public:
    Softmax(void);
    matrix::Matrix<float> forward(matrix::Matrix<float> input);
    matrix::Matrix<float> backward(matrix::Matrix<float> dinput);

};

class SoftmaxCrossEntropy : public Layer {
    float loss;
    Softmax softmax;
    CategoricalCrossentropy crossEntropy;
public:
    matrix::Matrix<float> forward(matrix::Matrix<float> input, 
	    matrix::Matrix<float> y_true);
    matrix::Matrix<float> backward(matrix::Matrix<float> dinput, 
	    matrix::Matrix<float> y_true);

};

#endif
