#include <iostream>
#include <vector>
#include <cstdlib>
#include <random>
#include "matrix.h"
#include "losses.h"

class Layer {
    public:
	virtual Matrix<float> forward(Matrix<float>) = 0;
};


class Dense : public Layer { 
    // Weights:
    //  Each col corrosponds to all the weights associated with a neuron.
    //    Example:
    //    3 inputs, 1 neuron in layer
    // 	      [[0.xx],   
    // 	      [0.xx],  X [[x.xx, x.xx, x.xx]]  
    //        [0.xx]] :
    //    3 inputs, 2 neuron
    //        [[-0.xx, -0.xx],
    //         [ 0.xx,  0.xx],
    //         [ 0.xx,  0.xx]]
    // Biases:
    //   Each col corrosponds to the biases for the neuron
    //   3 input, 1 neuron in layer.
    //        [[0]]
    //   3 input, 2 neuron in layer.
    //        [[0, 0]]
    Matrix<float> weights; 
    Matrix<float> biases;
	

public:
    // Think of n_inputs actually as n_features of the input data
    Dense(int n_inputs, int n_neurons): weights(n_inputs, n_neurons), 
	    biases(1, n_neurons, 1) 
       {
	   randomize_weights();
       }

    void randomize_weights() {
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0.0,1.0);
	for (int i = 0; i < weights.rows; i++) {
	    for (int j = 0; j < weights.cols; j++) {
		weights[i][j] = (float) 1.0;//distribution(generator);
	    }
	}
    }

    Matrix<float> forward(Matrix<float> input) {
    	// Calculate the dot product between each neuron and input data
	return matrix_add(matrix_dot(input, weights), biases);
    }
};

class ReLU : public Layer {
    
    public:
    Matrix<float> forward(Matrix<float> input) {
	for (int i = 0; i < input.rows; i++) {
	    for (int j = 0; j < input.cols; j++) {
		if (input[i][j] < 0) {
		    input[i][j] = 0;
		}
	    }
	}
	return input;
    }
};


class Softmax: public Layer {

    public:
    Matrix<float> forward(Matrix<float> input) {
	Matrix<float> temp = matrix_exp(matrix_subtract(input, 
		    matrix_max(input)));
	return matrix_division(temp, matrix_sum(temp));
    }
};

int main() {
    // Lets setup our data
    std::vector<std::vector<float>> in { 
	                       {1.0, 2.0, 3.0},
			       {-4.0, -5.0, -6.0},  
			       {7.0, 8.0, 9.0} };
    
    std::vector<std::vector<float>> true_in { {0.0, 1.0},
					   {1.0, 0.0},
					   {0.0, 1.0} };
    Matrix<float> X(3, 3);
    X.set_matrix(in);
    Matrix<float> y_true(3, 2);
    y_true.set_matrix(true_in);

    Dense layer1(3, 3);
    ReLU layer2;
    Dense layer3(3, 2);
    Softmax layer4;
    CategoricalCrossentropy loss;
    matrix_print(X);
    Matrix<float> out1 = layer1.forward(X);
    matrix_print(out1);
    Matrix<float> out2 = layer2.forward(out1);
    matrix_print(out2);
    Matrix<float> out3 = layer3.forward(out2);
    matrix_print(out3);
    Matrix<float> out4 = layer4.forward(out3);
    matrix_print(out4);
    matrix_print(loss.loss(y_true, out4));

    return 1;
}
