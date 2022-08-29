#ifndef LAYERS_H
#define LAYERS_H
#include "matrix.h"
#include "losses.h"

class Layer {
public:
    virtual matrix::Matrix<float> forward(matrix::Matrix<float>) = 0;
    virtual matrix::Matrix<float> backward(matrix::Matrix<float>) = 0;
    virtual matrix::Matrix<float> backward(matrix::Matrix<float>, 
		matrix::Matrix<float>) = 0;
};


class Dense { 
    matrix::Matrix<float> weights; 
    matrix::Matrix<float> biases;
    matrix::Matrix<float> dweights;
    matrix::Matrix<float> dbiases;

public:
    
    Dense(int n_inputs, int n_neurons);

    void randomize_weights();

    matrix::Matrix<float> forward(matrix::Matrix<float> input);
    matrix::Matrix<float> backward(matrix::Matrix<float> inputs, 
		matrix::Matrix<float> dinput);
    matrix::Matrix<float> get_dbiases();
    matrix::Matrix<float> get_dweights();
    matrix::Matrix<float> get_biases();
    matrix::Matrix<float> get_weights();
    void set_biases(matrix::Matrix<float> new_biases);
    void set_weights(matrix::Matrix<float> new_weights);

};

class ReLU {    
    matrix::Matrix<float> inputs;
public:
    ReLU(void);
    matrix::Matrix<float> forward(matrix::Matrix<float> input);
    matrix::Matrix<float> backward(matrix::Matrix<float> inputs, 
		matrix::Matrix<float> dinput);

};

class Softmax {
    matrix::Matrix<float> inputs;
public:
    Softmax(void);
    matrix::Matrix<float> forward(matrix::Matrix<float> input);
    matrix::Matrix<float> backward(matrix::Matrix<float> dinput);

};

class SoftmaxCrossEntropy {
    float loss;
    Softmax softmax;
    CategoricalCrossentropy crossEntropy;
public:
    SoftmaxCrossEntropy(void);
    matrix::Matrix<float> forward(matrix::Matrix<float> input, 
	    matrix::Matrix<float> y_true);
    matrix::Matrix<float> backward(matrix::Matrix<float> dinput, 
	    matrix::Matrix<float> y_true);
    float get_loss();
};

#endif
