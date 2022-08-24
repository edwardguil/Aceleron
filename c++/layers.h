#ifndef LAYERS_H
#define LAYERS_H
#include "matrix.h"

class Layer {
public:
    virtual matrix::Matrix<float> forward(matrix::Matrix<float>) = 0;
};


class Dense : public Layer { 
    matrix::Matrix<float> weights; 
    matrix::Matrix<float> biases;
	
public:
    Dense(int n_inputs, int n_neurons);

    void randomize_weights();

    matrix::Matrix<float> forward(matrix::Matrix<float> input);
};

class ReLU : public Layer {
    
public:
    matrix::Matrix<float> forward(matrix::Matrix<float> input);

};

class Softmax: public Layer {

public:
    matrix::Matrix<float> forward(matrix::Matrix<float> input);

};

#endif
