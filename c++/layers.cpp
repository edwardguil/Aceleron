#include <cstdlib>
#include <random>
#include <cstdio>
#include "matrix.h"
#include "layers.h"


// ------------- DENSE  -------------- //


// Think of n_inputs actually as n_features of the input data
Dense::Dense(int n_inputs, int n_neurons): 
	    weights(n_inputs, n_neurons), biases(1, n_neurons, 1), 
	    dweights(n_inputs, n_neurons), dbiases(1, n_neurons, 1) 
	    { 
       randomize_weights();
}

void Dense::randomize_weights() {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0,1.0);
    for (int i = 0; i < weights.rows; i++) {
	for (int j = 0; j < weights.cols; j++) {
	    weights[i][j] = (float) 1.0;//distribution(generator);
	}
    }
}

matrix::Matrix<float> Dense::forward(matrix::Matrix<float> input) {
    // Calculate the dot product between each neuron and input data
    return matrix::add(matrix::dot(input, weights), biases);
}

matrix::Matrix<float> Dense::backward(matrix::Matrix<float> inputs, 
	    matrix::Matrix<float> dinput) {


    matrix::Matrix<float> temp = matrix::dot(matrix::transpose(inputs), dinput);
    dweights = temp;

    dbiases = matrix::sum(dinput, 0, true);

    return matrix::dot(dinput, matrix::transpose(weights));
}

matrix::Matrix<float> Dense::get_dweights() {
    return dweights;
}

matrix::Matrix<float> Dense::get_dbiases() {
    return dbiases;
}

matrix::Matrix<float> Dense::get_biases() {
    return biases;
}

matrix::Matrix<float> Dense::get_weights() {
    return weights;
}
void Dense::set_biases(matrix::Matrix<float> new_biases) {
    biases = new_biases;
}
void Dense::set_weights(matrix::Matrix<float> new_weights) {
    weights = new_weights;
}

// ------------- RELU  -------------- //

ReLU::ReLU(): inputs(1, 1) {}

matrix::Matrix<float> ReLU::forward(matrix::Matrix<float> input) {
    for (int i = 0; i < input.rows; i++) {
	for (int j = 0; j < input.cols; j++) {
	    if (input[i][j] < 0) {
		input[i][j] = 0;
	    }
	}
    }
    return input;
}

matrix::Matrix<float> ReLU::backward(matrix::Matrix<float> inputs, 
	    matrix::Matrix<float> dinput) {
    for (int i = 0; i < dinput.rows; i++) {
	for (int j = 0; j < dinput.cols; j++) {
	    if (inputs[i][j] <= 0) {
		dinput[i][j] = 0;
	    }
	}
    }
    return dinput;
}

// ------------- SOFTMAX -------------- //

Softmax::Softmax(): inputs(1, 1) {}

matrix::Matrix<float> Softmax::forward(matrix::Matrix<float> input) {
    matrix::Matrix<float> temp = matrix::exp(matrix::subtract(input, 
		matrix::max(input)));
    return matrix::division(temp, matrix::sum(temp));
}

matrix::Matrix<float> Softmax::backward(matrix::Matrix<float> dinput) {
    return dinput;
}

// ------------- SOFTMAX CROSSENTROPY  -------------- //

SoftmaxCrossEntropy::SoftmaxCrossEntropy(void): softmax(), crossEntropy() {}

matrix::Matrix<float> SoftmaxCrossEntropy::forward(matrix::Matrix<float> input, 
	    matrix::Matrix<float> y_true) {
    matrix::Matrix<float> out = softmax.forward(input);
    loss = crossEntropy.calculateLoss(out, y_true);
    return out;
}

matrix::Matrix<float> SoftmaxCrossEntropy::backward(matrix::Matrix<float> dinput, 
	    matrix::Matrix<float> y_true) {
    // Expects y_true to be one hot encoded
    matrix::Matrix<int> converted = matrix::argmax(y_true);
    for (int i = 0; i < dinput.rows; i++) {
	dinput[i][converted[i][0]] -= 1;
    }
    matrix::Matrix<float> temp(1, 1, dinput.rows);
    return matrix::division(dinput, temp);
}

float SoftmaxCrossEntropy::get_loss() {
    return loss;
}
