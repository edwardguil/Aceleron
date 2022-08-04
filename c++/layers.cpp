#include <iostream>
#include <vector>
#include <cstdlib>
#include <tuple>
#include <array>
/*
Author: Edward Guilfoyle
Comments: It would be wise to extend the vector class, and create a matrix
class to implement functionality like Matrix.dot(). Similar to numpy...
*/
using namespace std;


template <typename dtype>
class Matrix {

    vector<vector<dtype>> matrix;

public:
    int rows;
    int cols;

    Matrix(int rows, int cols): matrix(rows, vector<dtype>(cols)), 
	rows(rows), cols(cols) {
    }

    vector<dtype>& operator[](int i) {
        return matrix[i];
    }

    unsigned long size() {
    	return matrix.size();
    }

    void set_matrix(vector<vector<dtype>> update) {
	// Dangerous, only use this to initialize
	matrix = update;
    }
};

template <typename dtype>
void print_matrix(Matrix<dtype> matrix) {
    char* temp;
    cout << "[";
    for(int i = 0; i < matrix.rows; i++) {
	cout << "[";
	for (int j = 0; j < matrix.cols; j++) {
	    temp = j + 1 == matrix.cols ? (char*) "" : (char*) ", ";
	    cout << matrix[i][j] << temp;
	}
	temp = i + 1 == matrix.rows ? (char*) "" : (char*) ",\n ";
	cout << "]" << temp;
    }
    cout << "]\n";
}

template <typename dtype>
Matrix<dtype> dot(Matrix<dtype> a, Matrix<dtype> b) {
    Matrix<dtype> out(a.rows, b.cols);
    // a should be inputs
    // b should be weights
    for (int i = 0; i < a.rows; i++) {
	// For each sample
	for (int j = 0; j < b.cols; j++) {
	    // For each neuron
	    dtype product = 0; 
	    for (int k = 0; k < a.cols; k++) {
		// For each feature of that sample
		product +=  a[i][k] * b[k][j];
	    }
	    //out[i][j] = product + biases[0][j];
	    out[i][j] = product;
	}
    }
    return out;
}

template <typename dtype>
Matrix<dtype> add(Matrix<dtype> a, Matrix<dtype> b) {
    // Let a be the dot(input, weights).
    // Let b be the biases
    Matrix<dtype> out(a.rows, a.cols);
    if (a.cols == b.cols && b.rows == 1) {
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; j < a.cols; j++) {
		out[i][j] = a[i][j] + b[0][j];
	    }
	}
    }
    return out;
}


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
    vector<vector<float>> in { {1, 1, 1},
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
