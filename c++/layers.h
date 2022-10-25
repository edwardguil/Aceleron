#ifndef LAYERS_H
#define LAYERS_H
#include "matrix.h"
#include "losses.h"

template<typename dtype = double, typename vtype = std::vector<double>>
class Layer {
public:
    virtual matrix::Matrix<dtype, vtype> forward(matrix::Matrix<dtype, vtype>&) = 0;
    matrix::Matrix<dtype, vtype> forward(matrix::Matrix<dtype, vtype>&, matrix::Matrix<dtype, vtype>&) = 0;
    virtual matrix::Matrix<dtype, vtype> backward(matrix::Matrix<dtype, vtype>&, 
		matrix::Matrix<dtype, vtype>&) = 0;
};

template<typename dtype = double, typename vtype = std::vector<double>>
class Dense { 
    // n_inputs x n_neurons matrix storing weights
    matrix::Matrix<dtype, vtype> weights; 
    // 1 x n_neurons matrix storing biases
    matrix::Matrix<dtype, vtype> biases;
    // n_inputs x n_neurons matrix storing partial derivative w.r.t. loss
    matrix::Matrix<dtype, vtype> dweights;
    // 1 x n_neurons matrix storing partial derivative w.t.r loss
    matrix::Matrix<dtype, vtype> dbiases;

public:

    
    Dense(int n_inputs, int n_neurons);
    matrix::Matrix<dtype, vtype> forward(matrix::Matrix<dtype, vtype>& input);
    matrix::Matrix<dtype, vtype> backward(matrix::Matrix<dtype, vtype>& inputs, 
		matrix::Matrix<dtype, vtype>& dinput);
    matrix::Matrix<dtype, vtype> get_dbiases();
    matrix::Matrix<dtype, vtype> get_dweights();
    matrix::Matrix<dtype, vtype> get_biases();
    matrix::Matrix<dtype, vtype> get_weights();
    void randomize_weights();
    void set_biases(matrix::Matrix<dtype, vtype> new_biases);
    void set_weights(matrix::Matrix<dtype, vtype> new_weights);

};

template<typename dtype = double, typename vtype = std::vector<double>>
class ReLU {    
public:
    ReLU(void);
    matrix::Matrix<dtype, vtype> forward(matrix::Matrix<dtype, vtype>& input);
    matrix::Matrix<dtype, vtype> backward(matrix::Matrix<dtype, vtype>& inputs, 
		matrix::Matrix<dtype, vtype>& dinput);

};

template<typename dtype = double, typename vtype = std::vector<double>>
class Softmax {
public:
    Softmax(void);
    matrix::Matrix<dtype, vtype> forward(matrix::Matrix<dtype, vtype>& input);
    matrix::Matrix<dtype, vtype> backward(matrix::Matrix<dtype, vtype>& dinput);

};

template<typename dtype = double, typename vtype = std::vector<double>>
class SoftmaxCrossEntropy {
    double loss;
    Softmax<dtype, vtype> softmax;
    CategoricalCrossentropy<dtype, vtype> crossEntropy;
public:
    SoftmaxCrossEntropy(void);
    matrix::Matrix<dtype, vtype> forward(matrix::Matrix<dtype, vtype>& input, 
	    matrix::Matrix<dtype, vtype>& y_true);
    matrix::Matrix<dtype, vtype> backward(matrix::Matrix<dtype, vtype>& dinput, 
	    matrix::Matrix<dtype, vtype>& y_true);
    double get_loss();
};

#include "layers.cu"

#endif
