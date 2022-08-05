#include <iostream>
#include <vector>
#include <cstdlib>
#include "matrix.h"
/*
Author: Edward Guilfoyle
Comments: It would be wise to extend the vector class, and create a matrix
class to implement functionality like Matrix.dot(). Similar to numpy...
*/


class Dense { 
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
	for (int i = 0; i < weights.rows; i++) {
	    for (int j = 0; j < weights.cols; j++) {
		weights[i][j] = (float) rand()/RAND_MAX;
	    }
	}
    }

    Matrix<float> forward(Matrix<float> input) {
    	// Calculate the dot product between each neuron and input data
	return add(dot(input, weights), biases);
    }
};



int main() {
    std::vector<std::vector<float>> in { {1, 1, 1},
			       {2, 2, 2}, 
    			       {3, 3, 3} };
    Matrix<float> X(3, 3);
    X.set_matrix(in);
    Dense dense(3, 8);
    Matrix<float> out = dense.forward(X);
    print_matrix(out);
    //Dense dense(4, 2); 
    //dense.forward(X);
    return 1;
}
