#include <cstdlib>
#include <random>
#include "matrix.h"
#include "layers.h"

    // Think of n_inputs actually as n_features of the input data
Dense::Dense(int n_inputs, int n_neurons): weights(n_inputs, n_neurons), 
	    biases(1, n_neurons, 1) {
       randomize_weights();
};

void Dense::randomize_weights() {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0,1.0);
    for (int i = 0; i < weights.rows; i++) {
	for (int j = 0; j < weights.cols; j++) {
	    weights[i][j] = (float) 1.0;//distribution(generator);
	}
    }
};

matrix::Matrix<float> Dense::forward(matrix::Matrix<float> input) {
    // Calculate the dot product between each neuron and input data
    return matrix::add(matrix::dot(input, weights), biases);
}

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


matrix::Matrix<float> Softmax::forward(matrix::Matrix<float> input) {
    matrix::print(matrix::max(input));
    matrix::print(matrix::subtract(input, matrix::max(input)));
    matrix::Matrix<float> temp = matrix::exp(matrix::subtract(input, 
		matrix::max(input)));
    matrix::print(temp);
    matrix::print(matrix::sum(temp));
    return matrix::division(temp, matrix::sum(temp));
}
