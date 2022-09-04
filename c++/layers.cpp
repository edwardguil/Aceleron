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
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for (int i = 0; i < weights.rows; i++) {
	for (int j = 0; j < weights.cols; j++) {
	    weights[i][j] = (double) 1.0;//distribution(generator);
	}
    }
}

matrix::Matrix<double> Dense::forward(matrix::Matrix<double> input) {
    // Calculate the dot product between each neuron and input data
    return matrix::add(matrix::dot(input, weights), biases);
}

matrix::Matrix<double> Dense::backward(matrix::Matrix<double> inputs, 
	    matrix::Matrix<double> dinput) {


    matrix::Matrix<double> temp = matrix::dot(matrix::transpose(inputs), dinput);
    dweights = temp;

    dbiases = matrix::sum(dinput, 0, true);

    return matrix::dot(dinput, matrix::transpose(weights));
}

matrix::Matrix<double> Dense::get_dweights() {
    return dweights;
}

matrix::Matrix<double> Dense::get_dbiases() {
    return dbiases;
}

matrix::Matrix<double> Dense::get_biases() {
    return biases;
}

matrix::Matrix<double> Dense::get_weights() {
    return weights;
}
void Dense::set_biases(matrix::Matrix<double> new_biases) {
    biases = new_biases;
}
void Dense::set_weights(matrix::Matrix<double> new_weights) {
    weights = new_weights;
}

// ------------- RELU  -------------- //

ReLU::ReLU(): inputs(1, 1) {}

matrix::Matrix<double> ReLU::forward(matrix::Matrix<double> input) {
    for (int i = 0; i < input.rows; i++) {
	for (int j = 0; j < input.cols; j++) {
	    if (input[i][j] < 0) {
		input[i][j] = 0;
	    }
	}
    }
    return input;
}

matrix::Matrix<double> ReLU::backward(matrix::Matrix<double> inputs, 
	    matrix::Matrix<double> dinput) {
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

matrix::Matrix<double> Softmax::forward(matrix::Matrix<double> input) {
    matrix::Matrix<double> temp = matrix::exp(matrix::subtract(input, 
		matrix::max(input)));
    return matrix::division(temp, matrix::sum(temp));
}

matrix::Matrix<double> Softmax::backward(matrix::Matrix<double> dinput) {
    return dinput;
}

// ------------- SOFTMAX CROSSENTROPY  -------------- //

SoftmaxCrossEntropy::SoftmaxCrossEntropy(void): softmax(), crossEntropy() {}

matrix::Matrix<double> SoftmaxCrossEntropy::forward(matrix::Matrix<double> input, 
	    matrix::Matrix<double> y_true) {
    matrix::Matrix<double> out = softmax.forward(input);
    loss = crossEntropy.calculateLoss(out, y_true);
    return out;
}

matrix::Matrix<double> SoftmaxCrossEntropy::backward(matrix::Matrix<double> dinput, 
	    matrix::Matrix<double> y_true) {
    // Expects y_true to be one hot encoded
    matrix::Matrix<int> converted = matrix::argmax(y_true);
    for (int i = 0; i < dinput.rows; i++) {
	dinput[i][converted[i][0]] -= 1;
    }
    matrix::Matrix<double> temp(1, 1, dinput.rows);
    return matrix::division(dinput, temp);
}

double SoftmaxCrossEntropy::get_loss() {
    return loss;
}
