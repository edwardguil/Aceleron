#include <iostream>
#include <cstdlib>
#include <vector>
#include "matrix.h"

template <typename dtype>
Matrix<dtype>::Matrix(int rows, int cols): matrix(rows, 
	std::vector<dtype>(cols)), rows(rows), cols(cols) {
}

template <typename dtype>
std::vector<dtype>& Matrix<dtype>::operator[](int i) {
    return matrix[i];
}

template <typename dtype>
unsigned long Matrix<dtype>::size() {
    return matrix.size();
}

template <typename dtype>
void Matrix<dtype>::set_matrix(std::vector<std::vector<dtype>> update) {
    // Dangerous, only use this to initialize
    matrix = update;
}

template <typename dtype>
void print_matrix(Matrix<dtype> matrix) {
    char* temp;
    std::cout << "[";
    for(int i = 0; i < matrix.rows; i++) {
	std::cout << "[";
	for (int j = 0; j < matrix.cols; j++) {
	    temp = j + 1 == matrix.cols ? (char*) "" : (char*) ", ";
	    std::cout << matrix[i][j] << temp;
	}
	temp = i + 1 == matrix.rows ? (char*) "" : (char*) ",\n ";
	std::cout << "]" << temp;
    }
    std::cout << "]\n";
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

/* vim: set ft=cpp: */
