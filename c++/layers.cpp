#include <iostream>
#include <vector>
#include <cstdlib>
#include <tuple>
/*
Author: Edward Guilfoyle
Comments: It would be wise to extend the vector class, and create a matrix
class to implement functionality like Matrix.dot(). Similar to numpy...
*/
using namespace std;

void print_matrix(vector<vector<float>> matrix) {
    char* temp;
    cout << "[";
    for(int i = 0; i < (int) matrix.size(); i++) {
	cout << "[";
	for (int j = 0; j < (int) matrix[0].size(); j++) {
	    temp = j + 1 == (int) matrix[0].size() ? (char*) "" : (char*) ", ";
	    cout << matrix[i][j] << temp;
	}
	temp = i + 1 == (int) matrix.size() ? (char*) "" : (char*) ",\n ";
	cout << "]" << temp;
    }
    cout << "]\n";
}


template <class T>
class Matrix {

    vector<vector<T>> matrix;
    unsigned long rows;
    unsigned long cols;
    unsigned long* shape;

public:
    Matrix(int rows, int cols): matrix(rows, vector<T>(cols)) {
	cols = (unsigned long) cols;
	rows = (unsigned long) rows;
	shape[0] = rows;
	shape[1] = cols;
    }

    vector<T>& operator[](int i) {
        return matrix[i];
    }

    unsigned long size() {
    	return matrix.size();
    }

    void print() {
	char* temp;
	cout << "[";
	for(int i = 0; i < (int) matrix.size(); i++) {
	    cout << "[";
	    for (int j = 0; j < (int) matrix[0].size(); j++) {
		temp = j + 1 == (int) matrix[0].size() ? (char*) "" : (char*) ", ";
		cout << matrix[i][j] << temp;
	    }
	    temp = i + 1 == (int) matrix.size() ? (char*) "" : (char*) ",\n ";
	    cout << "]" << temp;
	}
	cout << "]\n";
    }
};

template <typename dtype>
Matrix<dtype> dot(Matrix<dtype> a, Matrix<dtype> b) {
    Matrix<dtype> out(a.rows, b.cols);
    // a should be inputs
    // b should be weights
    for (int i = 0; (int) a.rows; i++) {
	// For each sample
	for (int j = 0; j < (int) b.cols; j++) {
	    // For each neuron
	    dtype product = 0; 
	    for (int k = 0; k < (int) a.cols; k++) {
		// For each feature of that sample
		product +=  a[i][k] * b[k][j];
	    }
	    //out[i][j] = product + biases[0][j];
	    out[i][j] = product;
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
    vector<vector<float>> weights; 
    vector<vector<float>> biases;
	

public:
    // Think of n_inputs actually as n_features of the input data
    Dense(int n_inputs, int n_neurons): weights(n_inputs, 
	    vector<float>(n_neurons)), biases(1, vector<float>(n_neurons)) {
	// Fill weights vector with random values between 0-1
	// This has O**2 time complexity, there must be a better way.
	for (int i = 0; i < n_neurons; i++) {
	    for (int j = 0; j < n_inputs; j++) {
		weights[j][i] = (float) rand()/RAND_MAX;
	    }
	}
    }

    vector<vector<float>> forward(vector<vector<float>> input) {
    	// Calculate the dot product between each neuron and input data
	vector<vector<float>> out(input.size(), vector<float>(weights[0].size())); 
	for (int i = 0; i < (int) input.size(); i++) {
	    // For each sample
	    for (int j = 0; j < (int) weights[0].size(); j++) {
		// For each neuron
		int product = 0; biases[0][j]; 
		for (int k = 0; k < (int) input[0].size(); k++) {
		    // For each feature of that sample
		    product += weights[k][j] * input[i][k];
		}
		out[i][j] = product + biases[0][j];
	    }
	}
	return out;
    }
};



int main() {
    vector<vector<float>> X{ {1, 2, 3},
			     {4, 5, 6}, 
    			     {7, 8, 9} };
    Dense dense(3, 4);
    vector<vector<float>> test = dense.forward(X);
    print_matrix(test);
    //Dense dense(4, 2); 
    //dense.forward(X);
    return 1;
}
