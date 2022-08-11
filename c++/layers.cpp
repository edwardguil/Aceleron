#include <iostream>
#include <vector>
#include <cstdlib>
#include <random>
#include "matrix.h"

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
	    biases(1, n_neurons) 
       {
	    randomize_weights();
       }

    void randomize_weights() {
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0.0,1.0);
	for (int i = 0; i < weights.rows; i++) {
	    for (int j = 0; j < weights.cols; j++) {
		weights[i][j] = distribution(generator);
		//weights[i][j] = (float) rand()/RAND_MAX;
	    }
	}
    }

    Matrix<float> forward(Matrix<float> input) {
    	// Calculate the dot product between each neuron and input data
	return add(dot(input, weights), biases);
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


int main() {
    // Lets setup our data
    std::vector<std::vector<float>> in { {1, 1, 1},
			       {-2, -2, -2}, 
    
			       {3, 3, 3} };
    Matrix<float> X(3, 3);
    X.set_matrix(in);
    // Lets define our layers
    ReLU relu;
    Dense dense(3, 3);
    // Lets do a forward pass
    Matrix<float> out = dense.forward(X);
    Matrix<float> out2 = relu.forward(out);
    print_matrix(out2);
    return 1;
}
