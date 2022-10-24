#ifndef LAYERS_H
#define LAYERS_H
#include "matrix.h"
#include "losses.h"

class Layer {
public:
    virtual matrix::Matrix<double> forward(matrix::Matrix<double>&) = 0;
    virtual matrix::Matrix<double> backward(matrix::Matrix<double>&) = 0;
    virtual matrix::Matrix<double> backward(matrix::Matrix<double>&, 
		matrix::Matrix<double>&) = 0;
};


class Dense { 
    // 1 x n_neurons matrix storing biases
    matrix::Matrix<double> biases;
    // n_inputs x n_neurons matrix storing partial derivative w.r.t. loss
    matrix::Matrix<double> dweights;
    // 1 x n_neurons matrix storing partial derivative w.t.r loss
    matrix::Matrix<double> dbiases;

public:
    // n_inputs x n_neurons matrix storing weights
    matrix::Matrix<double> weights; 
    
    Dense(int n_inputs, int n_neurons);
    matrix::Matrix<double> forward(matrix::Matrix<double>& input);
    matrix::Matrix<double> backward(matrix::Matrix<double>& inputs, 
		matrix::Matrix<double>& dinput);
    matrix::Matrix<double> get_dbiases();
    matrix::Matrix<double> get_dweights();
    matrix::Matrix<double> get_biases();
    matrix::Matrix<double> get_weights();
    void randomize_weights();
    void set_biases(matrix::Matrix<double> new_biases);
    void set_weights(matrix::Matrix<double> new_weights);

};

class ReLU {    
public:
    ReLU(void);
    matrix::Matrix<double> forward(matrix::Matrix<double>& input);
    matrix::Matrix<double> backward(matrix::Matrix<double>& inputs, 
		matrix::Matrix<double>& dinput);

};

class Softmax {
public:
    Softmax(void);
    matrix::Matrix<double> forward(matrix::Matrix<double>& input);
    matrix::Matrix<double> backward(matrix::Matrix<double>& dinput);

};

class SoftmaxCrossEntropy {
    double loss;
    Softmax softmax;
    CategoricalCrossentropy crossEntropy;
public:
    SoftmaxCrossEntropy(void);
    matrix::Matrix<double> forward(matrix::Matrix<double>& input, 
	    matrix::Matrix<double>& y_true);
    matrix::Matrix<double> backward(matrix::Matrix<double>& dinput, 
	    matrix::Matrix<double>& y_true);
    double get_loss();
};

#endif
