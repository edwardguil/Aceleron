#include <iostream>
#include <cstdlib>
#include <vector>
#include <climits>
#include <cmath>
#include "matrix.h"

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
void matrix_print(Matrix<dtype> matrix) {
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
Matrix<dtype> matrix_dot(Matrix<dtype> a, Matrix<dtype> b) {
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
Matrix<dtype> matrix_add(Matrix<dtype> a, Matrix<dtype> b) {
    Matrix<dtype> out(a.rows, a.cols);
    if (a.cols == b.cols && b.rows == 1) {
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; j < a.cols; j++) {
		out[i][j] = a[i][j] + b[0][j];
	    }
	}
    } else if (a.rows == b.rows && b.cols == 1) {
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; j < a.cols; j++) {
		out[i][j] = a[i][j] + b[i][0];
	    }
	}
    } else if (a.cols == b.cols && a.rows == b.cols) {
	// Element wise addittion; 
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; i < a.rows; i++) {
		out[i][j] = a[i][j] + b[i][j];
	    }
	}
    }
    return out;
}

template <typename dtype>
Matrix<dtype> matrix_subtract(Matrix<dtype> a, Matrix<dtype> b) {
    Matrix<dtype> out(a.rows, a.cols);
    if (a.cols == b.cols && b.rows == 1) {
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; j < a.cols; j++) {
		out[i][j] = a[i][j] - b[0][j];
	    }
	}
    } else if (a.rows == b.rows && b.cols == 1) {
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; j < a.cols; j++) {
		out[i][j] = a[i][j] - b[i][0];
	    }
	}
    } else if (a.cols == b.cols && a.rows == b.cols) {
	// Element wise subtraction; 
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; i < a.rows; i++) {
		out[i][j] = a[i][j] - b[i][j];
	    }
	}
    }
    return out;
}

template <typename dtype>
Matrix<dtype> matrix_mul(Matrix<dtype> a, Matrix<dtype> b) {
    Matrix<dtype> out(a.rows, a.cols);
    if (a.cols == b.cols && b.rows == 1) {
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; j < a.cols; j++) {
		out[i][j] = a[i][j] * b[0][j];
	    }
	}
    } else if (a.rows == b.rows && b.cols == 1) {
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; j < a.cols; j++) {
		out[i][j] = a[i][j] * b[i][0];
	    }
	}
    } else if (a.cols == b.cols && a.rows == b.cols) {
	// Element wise multiplication; 
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; i < a.rows; i++) {
		out[i][j] = a[i][j] * b[i][j];
	    }
	}
    }
    return out;
}

template <typename dtype>
Matrix<dtype> matrix_division(Matrix<dtype> a, Matrix<dtype> b) {
    Matrix<dtype> out(a.rows, a.cols);
    if (a.cols == b.cols && b.rows == 1) {
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; j < a.cols; j++) {
		out[i][j] = a[i][j] / b[0][j];
	    }
	}
    } else if (a.rows == b.rows && b.cols == 1) {
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; j < a.cols; j++) {
		out[i][j] = a[i][j] / b[i][0];
	    }
	}
    } else if (a.cols == b.cols && a.rows == b.cols) {
	// Element wise division
	for (int i = 0; i < a.rows; i++) {
	    for (int j = 0; i < a.rows; i++) {
		out[i][j] = a[i][j] / b[i][j];
	    }
	}
    }
    return out;
}


template <typename dtype>
Matrix<dtype> matrix_max(Matrix<dtype> input) {
    // finds max value over axis 1 (finds max value in each row)
    // keepdims = true
    Matrix<dtype> maxValues = Matrix<dtype>(input.rows, 1, (dtype) -INT_MAX); 
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
Matrix<dtype> matrix_sum(Matrix<dtype> input) {
    // Sums over axis 1 (sums each row)
    // keepdims = true
    Matrix<dtype> out = Matrix<dtype>(input.rows, 1); 
    for (int i = 0; i < input.rows; i++) {
	for (int j = 0; j < input.cols; j++) {
	    out[i][0] += input[i][j];
	}
    }
    return out;
}

template <typename dtype>
Matrix<dtype> matrix_log(Matrix<dtype> input) {
    // Element wise log
    Matrix<dtype> out(input.rows, input.cols);
    for (int i = 0; i < input.rows; i++) {
	for (int j = 0; j < input.cols; j++) {
	    out[i][j] = log(input[i][j]);
	}
    }
    return out;
}

template <typename dtype>
Matrix<dtype> matrix_exp(Matrix<dtype> input) {
    // Element wise exponential
    Matrix<dtype> out(input.rows, input.cols);
    for (int i = 0; i < input.rows; i++) {
	for (int j = 0; j < input.cols; j++) {
	    out[i][j] = exp(input[i][j]);
	}
    }
    return out;
}
