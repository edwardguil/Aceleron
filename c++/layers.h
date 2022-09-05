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
    matrix::Matrix<double> weights; 
    matrix::Matrix<double> biases;
    matrix::Matrix<double> dweights;
    matrix::Matrix<double> dbiases;

public:
    
    Dense(int n_inputs, int n_neurons);

    void randomize_weights();

    matrix::Matrix<double> forward(matrix::Matrix<double>& input);
    matrix::Matrix<double> backward(matrix::Matrix<double>& inputs, 
		matrix::Matrix<double>& dinput);
    matrix::Matrix<double> get_dbiases();
    matrix::Matrix<double> get_dweights();
    matrix::Matrix<double> get_biases();
    matrix::Matrix<double> get_weights();
    void set_biases(matrix::Matrix<double> new_biases);
    void set_weights(matrix::Matrix<double> new_weights);

};

class ReLU {    
    matrix::Matrix<double> inputs;
public:
    ReLU(void);
    matrix::Matrix<double> forward(matrix::Matrix<double>& input);
    matrix::Matrix<double> backward(matrix::Matrix<double>& inputs, 
		matrix::Matrix<double>& dinput);

};

class Softmax {
    matrix::Matrix<double> inputs;
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
