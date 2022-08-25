#include <iostream>
#include <cstdlib>
#include <vector>
#include <climits>
#include <cmath>
#include <type_traits>
#include "matrix.h"
namespace matrix 
{

    template <typename dtype>
Matrix<dtype>::Matrix(int rows, int cols, dtype value): matrix(rows, 
	std::vector<dtype>(cols, value)), rows(rows), cols(cols) {
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
Matrix<dtype> Matrix<dtype>::copy() {
    Matrix<dtype> out(rows, cols);
    out.set_matrix(matrix);
    return out;
}

template <typename dtype>
void print(Matrix<dtype> matrix) {
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
    // a should be inputs
    // b should be weights
    Matrix<dtype> out(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++) {
	// For each sample
	for (int j = 0; j < a.cols; j++) {
	    // For each feature in that row
	    for (int k = 0; k < b.rows; k++) {
		// For each neuron
		out[i][j] +=  a[i][k] * b[k][j];
	    }
	}
    }
    return out;
}

template <typename dtype>
Matrix<dtype> max(Matrix<dtype> input) {
    // finds max value over axis 1 (finds max value in each row)
    // keepdims = true
    Matrix<dtype> maxValues(input.rows, 1, (dtype) -INT_MAX); 
    for (int i = 0; i < input.rows; i++) {
	for (int j = 0; j < input.cols; j++) {
	    if (input[i][j] > maxValues[i][0]) {
		maxValues[i][0] = input[i][j];
	    }
	}
    }
    return maxValues;
}


template <typename dtype>
Matrix<dtype> sum(Matrix<dtype> input, int axis, bool keepdims) {
    // Default axis=1 and keepdims=true
    if (keepdims) {
	if (axis) {
	    Matrix<dtype> out(input.rows, 1); 
	    for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
		    out[i][0] += input[i][j];
		}
	    }
	    return out;
	} else {
	    Matrix<dtype> out(1, input.cols);
	    for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
		    out[0][j] += input[i][j];
		}
	    }
	    return out;
	}
    } else {
	Matrix<dtype> out(1, 1);
	for (int i = 0; i < input.rows; i++) {
	    for (int j = 0; j < input.cols; j++) {
		out[0][0] += input[i][j];
	    }
	}
	return out;
    }
    return input;
}

template <typename dtype, typename Operator>
Matrix<dtype> matrix_general(Matrix<dtype> a, Matrix<dtype> b, Operator op) {
    Matrix<dtype> out(a.rows, a.cols);
    if (a.rows == b.rows && a.cols == b.cols) {
	// Element wise; 
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; j < a.rows; j++) {
		out[i][j] = op(a[i][j], b[i][j]);
	    }
	}
    }
    else if (a.cols == b.cols && b.rows == 1) {
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; j < a.cols; j++) {
		out[i][j] = op(a[i][j], b[0][j]);
	    }
	}
    } else if (a.rows == b.rows && b.cols == 1) {
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; j < a.cols; j++) {
		out[i][j] = op(a[i][j], b[i][0]);
	    }
	}
    } else {
	return a;
    }
    return out;
}

template <typename dtype>
Matrix<dtype> add(Matrix<dtype> a, Matrix<dtype> b) {
    return matrix_general(a, b, [](dtype x, dtype y){
	return x + y;
    });
}

template <typename dtype> 
Matrix<dtype> subtract(Matrix<dtype> a, Matrix<dtype> b) {
    return matrix_general(a, b, [](dtype x, dtype y){
	return x - y;
    });
}

template <typename dtype>
Matrix<dtype> mul(Matrix<dtype> a, Matrix<dtype> b) {
    return matrix_general(a, b, [](dtype x, dtype y){
	return x * y;
    });
}

template <typename dtype> 
Matrix<dtype> division(Matrix<dtype> a, Matrix<dtype> b) {
    return matrix_general(a, b, [](dtype x, dtype y){
	return x / y;
    });
}

template <typename dtype> 
Matrix<dtype> equals(Matrix<dtype> a, Matrix<dtype> b) {
    // This is only defined for when both inputs are Matrix<int>
    // Can also utlizie |x - y| < EPISLON. 
    return matrix_general(a, b, [](dtype x, dtype y){
	return x == y;
    });
}



template <typename dtype>
Matrix<dtype> exp(Matrix<dtype> a) {
    Matrix<dtype> out(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
	for (int j = 0; j < a.cols; j++) {
	    out[i][j] = std::exp(a[i][j]);
	}
    }
    return out;
}

template <typename dtype>
Matrix<dtype> log(Matrix<dtype> a) {
    Matrix<dtype> out(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
	for (int j = 0; j < a.cols; j++) {
	    out[i][j] = std::log(a[i][j]);
	}
    }
    return out;
}

template <typename dtype>
Matrix<dtype> mul_const(Matrix<dtype> a, dtype b) {
    Matrix<dtype> out(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
	for (int j = 0; j < a.cols; j++) {
	    out[i][j] = a[i][j] * b;
	}
    }
    return out;
}

template <typename dtype>
Matrix<int> argmax(Matrix<dtype> a) {
    // Finds idx of max value across axis=1 (rows)
    Matrix<int> out(a.rows, 1);
    dtype max;
    for (int i = 0; i < a.rows; i++) {
	max = a[i][0];
	for (int  j = 1; j < a.cols; j++) {
	    if (max < a[i][j]) {
		out[i][0] = j;
	    }
	}
    }
    return out;
}

}
