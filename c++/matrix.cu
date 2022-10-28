#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>
#include <climits>
#include <cmath>
#include "cuda.h"
//#include <type_traits>

namespace matrix {

std::vector<void*> _FREE;

auto DotTime = std::chrono::microseconds::zero();
auto MaxTime = std::chrono::microseconds::zero();
auto SumTime = std::chrono::microseconds::zero();
auto TransposeTime = std::chrono::microseconds::zero();
auto AddTime = std::chrono::microseconds::zero();
auto SubtractTime = std::chrono::microseconds::zero();
auto MulTime = std::chrono::microseconds::zero();
auto DivisionTime = std::chrono::microseconds::zero();
auto MulConstTime = std::chrono::microseconds::zero();
auto ExpTime = std::chrono::microseconds::zero();
auto LogTime = std::chrono::microseconds::zero();
auto EqualsTime = std::chrono::microseconds::zero();
auto MallocTime = std::chrono::microseconds::zero();
auto MemCpyTime = std::chrono::microseconds::zero();
auto ArgmaxTime = std::chrono::microseconds::zero();
auto ReluFwdTime = std::chrono::microseconds::zero();
auto ReluBwdTime = std::chrono::microseconds::zero();
auto SoftMaxBwdTime = std::chrono::microseconds::zero();


/* Matrix<dtype>::Matrix()
* -----
* Initiliases a Matrix object with the respective shape. 
* If value is passed, also fills the matrix with the passed value.
* 
* Matrix is simply a wrapper for a datatype, which is generally
* std::vector<dtype>.
* It was added for readability, and convience to limit the passing 
* of rows, cols through every function that required it.
*
* @rows: the number of rows the matrix should have
* @cols: the number of cols the matrix should have
* @value: the value to fill the matrix, defaults to 0.
*/
template <typename dtype, typename vtype>
Matrix<dtype, vtype>::Matrix(int rows, int cols, dtype value, bool memset): matrix(rows*cols, value), 
		rows(rows), cols(cols) {
}

template <typename dtype, typename vtype>
Matrix<dtype, vtype>::~Matrix() {}

/* Matrix::operator[]()
* -----
* Returns a inner vector of the matrix, at the specified index
*
* @i: index of the value
*
* Returns: address of the value in the vector<dtype> at index i.
*/
template <typename dtype, typename vtype>
dtype& Matrix<dtype, vtype>::operator[](int i) {
	return matrix[i];
}

/* Matrix<dtype>::set_matrix()
* -----
* Overides the underlying std::vector<dtype> with the 
* passed std::vector<dtype>. Function should only
* be used for loading data in.
* 
* @update: the std::vector<dtype> to overide the underlying one.
*/
template <typename dtype, typename vtype>
void Matrix<dtype, vtype>::set_matrix(vtype update) {
    matrix = update;
}

/* Matrix<dtype>::get_matrix()
* -----
* Getter for the underlying std::vector<dtype>
*/
template <typename dtype, typename vtype>
vtype Matrix<dtype, vtype>::get_matrix() {
    return matrix;
}

/* Matrix<dtype>::get_idx()
* -----
* Getter for the a single element in vytpe
*/
template <typename dtype, typename vtype>
dtype Matrix<dtype, vtype>::get_idx(int i) {
    return matrix[i];
}

/* Matrix<dtype>::size()
* -----
* Returns the size of the underlying data structure (rows * cols)
*/
template <typename dtype, typename vtype>
int Matrix<dtype, vtype>::size() {
    return cols * rows;
}

/* Matrix<dtype>::copy()
* -----
* Deep copies this matrix object. 
* 
* Returns: a deep copy of this matrix object.
*/
template <typename dtype, typename vtype>
Matrix<dtype, vtype> Matrix<dtype, vtype>::copy() {
    Matrix<dtype> out(rows, cols);
    out.set_matrix(matrix);
    return out;
}

/* Matrix<double, double*>::Matrix()
* -----
* Initiliases a Matrix object with the respective shape. 
* 
* Matrix<double, double*> is a wrapper for a double* stored
* on a CUDA device. This is for convience to limit the passing 
* of rows, cols through every function that required it.
*
* @rows: the number of rows the matrix should have
* @cols: the number of cols the matrix should have
* @value: the value to fill the matrix, defaults to 0.
*/
template<>
inline Matrix<double, double*>::Matrix(int rows, int cols, double value, bool memset): 
		rows(rows), cols(cols) {
	auto StartTime = std::chrono::high_resolution_clock::now();
	cuda::checkError(cudaMalloc(&matrix, sizeof(double) * (rows * cols)));
	MallocTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	if (memset) {
		StartTime = std::chrono::high_resolution_clock::now();
		// For some readon cudaMemset doesn't work... have to do the slower method of setting...
		std::vector<double> in(rows*cols, value);
		cuda::checkError(cudaMemcpy(matrix, &(in[0]), sizeof(double) * (rows * cols), cudaMemcpyHostToDevice));
		MemCpyTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	}
	//cuda::checkError(cudaMemset(matrix, value, sizeof(double) * (rows * cols)));
}

template<>
inline Matrix<double, double*>::~Matrix() {}


template<>
inline Matrix<int, int*>::~Matrix() {}

/* Matrix<dtype>::set_matrix()
* -----
* Copies the update array to the device via cudaMemCpy. The
* num of elements copied is rows * cols of this matrix.
* 
* @update: the array to copy to the device.
*/
template <>
inline void Matrix<double, double*>::set_matrix(double* update) {
	auto StartTime = std::chrono::high_resolution_clock::now();
	cuda::checkError(cudaMemcpy(matrix, update, sizeof(double) * (rows * cols), cudaMemcpyHostToDevice));
	MemCpyTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
}


/* Matrix<dtype>::get_matrix()
* -----
* Returns the array from device to host. Copies array from
* underlying matrix, to passed array dest. 
* Caller is responsible for memory management.
*
* @dest: the array where the underlying array from device
* 	will be copied too.
*/
template <>
inline void Matrix<double, double*>::get_matrix(Matrix<double>& dest) {
	auto StartTime = std::chrono::high_resolution_clock::now();
    cuda::checkError(cudaMemcpy(&(dest[0]), matrix, sizeof(double) * (rows * cols), cudaMemcpyDeviceToHost));
}


/* Matrix<dtype>::get_idx()
* -----
* Getter for the a single element in vytpe
*/
template <>
inline double Matrix<double, double*>::get_idx(int i) {
	auto StartTime = std::chrono::high_resolution_clock::now();
	double value = 0;
	cuda::checkError(cudaMemcpy(&value, matrix, sizeof(double), cudaMemcpyDeviceToHost));
    MemCpyTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return value;
}

/* Matrix<dtype>::copy()
// 
* Deep copies this matrix object. 
* 
* Returns: a deep copy of this matrix object.
*/
template<>
inline Matrix<double, double*> Matrix<double, double*>::copy() {
	auto StartTime = std::chrono::high_resolution_clock::now();
    Matrix<double, double*> out(rows, cols);
    cuda::checkError(cudaMemcpy(out.get_matrix(), matrix, sizeof(double) * out.size(), cudaMemcpyDeviceToDevice));
    MemCpyTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

/* Matrix<double, double*>::Matrix()
* -----
* Initiliases a Matrix object with the respective shape. 
* 
* Matrix<double, double*> is a wrapper for a double* stored
* on a CUDA device. This is for convience to limit the passing 
* of rows, cols through every function that required it.
*
* @rows: the number of rows the matrix should have
* @cols: the number of cols the matrix should have
* @value: the value to fill the matrix, defaults to 0.
*/
template<>
inline Matrix<int, int*>::Matrix(int rows, int cols, int value, bool memset): 
		rows(rows), cols(cols) {
	auto StartTime = std::chrono::high_resolution_clock::now();
	cuda::checkError(cudaMalloc(&matrix, sizeof(int) * (rows * cols)));
	MallocTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	if (memset) {
		StartTime = std::chrono::high_resolution_clock::now();
		// For some readon cudaMemset doesn't work... have to do the slower method of setting...
		std::vector<int> in(rows*cols, value);
		cuda::checkError(cudaMemcpy(matrix, &(in[0]), sizeof(int) * (rows * cols), cudaMemcpyHostToDevice));
		MemCpyTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	}
	//cuda::checkError(cudaMemset(matrix, value, sizeof(double) * (rows * cols)));
}

/* Matrix<dtype>::set_matrix()
* -----
* Copies the update array to the device via cudaMemCpy. The
* num of elements copied is rows * cols of this matrix.
* 
* @update: the array to copy to the device.
*/
template <>
inline void Matrix<int, int*>::set_matrix(int* update) {
	auto StartTime = std::chrono::high_resolution_clock::now();
	cuda::checkError(cudaMemcpy(matrix, update, sizeof(int) * (rows * cols), cudaMemcpyHostToDevice));
	MemCpyTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
}


/* Matrix<dtype>::get_matrix()
* -----
* Returns the array from device to host. Copies array from
* underlying matrix, to passed array dest. 
* Caller is responsible for memory management.
*
* @dest: the array where the underlying array from device
* 	will be copied too.
*/
template <>
inline void Matrix<int, int*>::get_matrix(Matrix<int>& dest) {
	auto StartTime = std::chrono::high_resolution_clock::now();
    cuda::checkError(cudaMemcpy(&(dest[0]), matrix, sizeof(int) * (rows * cols), cudaMemcpyDeviceToHost));
	MemCpyTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
}


/* Matrix<dtype>::get_idx()
* -----
* Getter for the a single element in vytpe
*/
template <>
inline int Matrix<int, int*>::get_idx(int i) {
	auto StartTime = std::chrono::high_resolution_clock::now();
	int value = 0;
	cuda::checkError(cudaMemcpy(&value, matrix, sizeof(int), cudaMemcpyDeviceToHost));
    MemCpyTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return value;
}


/* Matrix<dtype>::copy()
// 
* Deep copies this matrix object. 
* 
* Returns: a deep copy of this matrix object.
*/
template<>
inline Matrix<int, int*> Matrix<int, int*>::copy() {
	auto StartTime = std::chrono::high_resolution_clock::now();
    Matrix<int, int*> out(rows, cols);
    cuda::checkError(cudaMemcpy(out.get_matrix(), matrix, sizeof(int) * out.size(), cudaMemcpyDeviceToDevice));
MemCpyTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}




/* print()
* -----
* Prints out the parsed matrix to stdout.
*
* @matrix: the matrix to be printed
*/
template <typename dtype>
void print(Matrix<dtype> matrix) {
    char* temp;
    std::cout << "[";
    for(int i = 0; i < matrix.rows; i++) {
	std::cout << "[";
		for (int j = 0; j < matrix.cols; j++) {
			temp = j + 1 == matrix.cols ? (char*) "" : (char*) ", ";
			std::cout << matrix[i*matrix.cols+j] << temp;
		}
		temp = i + 1 == matrix.rows ? (char*) "" : (char*) ",\n ";
		std::cout << "]" << temp;
    }
    std::cout << "]\n";
}

/* print()
* -----
* Prints out the parsed matrix to stdout.
*
* @matrix: the matrix to be printed
*/
template <typename dtype>
inline void print(Matrix<dtype, dtype*> matrix) {
	dtype* data = (dtype*) malloc(sizeof(dtype) * matrix.size());
	cuda::checkError(cudaMemcpy(data, matrix.get_matrix(), sizeof(dtype) * matrix.size(), cudaMemcpyDeviceToHost));
    char* temp;
    std::cout << "[";
    for(int i = 0; i < matrix.rows; i++) {
	std::cout << "[";
		for (int j = 0; j < matrix.cols; j++) {
			temp = j + 1 == matrix.cols ? (char*) "" : (char*) ", ";
			std::cout << data[i*matrix.cols+j] << temp;
		}
		temp = i + 1 == matrix.rows ? (char*) "" : (char*) ",\n ";
		std::cout << "]" << temp;
    }
    std::cout << "]\n";
	free(data);
}

/* dot()
* -----
* Computes the dot product between two matrices. That is a ⋅ b. 
* a.cols must equal b.rows. 
*
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
*
* Returns: the matrix containing the dot product of a & b.	
* 		Will be of shape (a.rows, b.cols)
*/
template <typename dtype>
Matrix<dtype> dot(Matrix<dtype> a, Matrix<dtype> b) {
	auto StartTime = std::chrono::high_resolution_clock::now();
    // a should be inputs
    // b should be weights
    Matrix<dtype> out(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++) {
	// For each sample
		for (int j = 0; j < b.cols; j++) {
				// For each feature in that row
				for (int k = 0; k < b.rows; k++) {
					// For each neuron
					out[i*out.cols + j] +=  a[i*a.cols + k] * b[k*b.cols + j];
				}
		}
    }
	DotTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
    return out;
}

/* max()
* -----
* Finds the max value over axis 1 of a matrix. That is
* the max value in each row. Equilivent to 
* np.max(a, axis=1, keepdims=true). 
*
* @a: the matrix to find the max of
*
* Returns: a new matrix containing the max value
* 		of each row in the parsed matrix.
*		will be of shape (a.rows, 1)
*/
template <typename dtype>
Matrix<dtype> max(Matrix<dtype> a) {
	auto StartTime = std::chrono::high_resolution_clock::now();
    // keepdims = true
    Matrix<dtype> maxValues(a.rows, 1, (dtype) -INT_MAX); 
    for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			if (a[i*a.cols + j] > maxValues[i]) {
				maxValues[i] = a[i*a.cols + j];
			}
		}
    }
	MaxTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
    return maxValues;
}

/* sum()
* -----
* Sums a matrix over the respective axis, keeps the 
* same dimensions as the original matrix if keepdims 
* is true.
*
* @a: the matrix to sum
* @axis: the axis to sum over
* @keepdims: weather the matrix should keep a dimensions
* 		or lose a dimension over the opposing axis
*
* Returns: the summed matrix of shape:
*		keepdims==1 & axis==1: (a.rows, 1)
*		keepdims==1 & axis==0: (1, a.cols)
*		keepdims==0: (1, 1)
*
*/
template <typename dtype>
Matrix<dtype> sum(Matrix<dtype> a, int axis, bool keepdims) {
	auto StartTime = std::chrono::high_resolution_clock::now();
	
    // Default axis=1 and keepdims=true
    if (keepdims) {
		if (axis) {
			Matrix<dtype> out(a.rows, 1); 
			for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < a.cols; j++) {
					out[i] += a[i*a.cols + j];
				}
			}
			SumTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
			return out;
		} else {
			Matrix<dtype> out(1, a.cols);
			for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < a.cols; j++) {
					out[j] += a[i*a.cols + j];
				}
			}
			SumTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
			return out;
		}
    } else {
		Matrix<dtype> out(1, 1);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.cols; j++) {
				out[0] += a[i*a.cols + j];
			}
		}
		SumTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
		return out;
    }
    //return a;
}
/* transpose()
* -----
* Transposes the provided matrix. That is rows become
* columns and columns become rows.
*
* @a: the matrix to be transposed
*
* Returns: a transposition of the provided matrix,
*		new matrix will be of size (a.cols, a.rows)
*/
template <typename dtype>
Matrix<dtype> transpose(Matrix<dtype> a) {
	auto StartTime = std::chrono::high_resolution_clock::now();
    Matrix<dtype> out(a.cols, a.rows);
    for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			out[j*out.cols + i] = a[i*a.cols + j];
		}
    }
	TransposeTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
    return out;

}

/* matrix_general()
* -----
* Loops over two matrices, applying the respective function to 
* two elements. Matrices can be of various shapes. 
* Computes the following: 
* 	if (a.rows == b.rows && a.cols == b.cols) -> element wise.
* 	if (a.cols == b.cols && b.rows == 1) -> a[i*a.cols + j] x a[j]
*   if (a.rows == b.rows && b.cols == 1) -> a[i*a.cols + j] x b[i]
*   if (b.rows == 1 && b.cols == 1) -> a[i*a.cols + j] x b[0]
*	for i in a.rows and j in a.cols
*
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
* @op: a function the takes two values, and computes a calculation.
*		e.g. int func(a, b) { return a + b ;}
*
* Returns: A new matrix containing the result of the applied
*		function. Will be of shape (a.rows, b.cols)
*/
template <typename dtype, typename Operator>
Matrix<dtype> matrix_general(Matrix<dtype> a, Matrix<dtype> b, Operator op) {
    Matrix<dtype> out(a.rows, a.cols);
    if (a.rows == b.rows && a.cols == b.cols) {
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.cols; j++) {
				out[i*out.cols + j] = op(a[i*a.cols + j], b[i*a.cols + j]);
			}
		}
    }
    else if (a.cols == b.cols && b.rows == 1) {
		for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < a.cols; j++) {
					out[i*out.cols + j] = op(a[i*a.cols + j], b[j]);
				}
		}
    } else if (a.rows == b.rows && b.cols == 1) {
		for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < a.cols; j++) {
					out[i*out.cols + j] = op(a[i*a.cols + j], b[i]);
				}
		}
    } else if (b.rows == 1 && b.cols == 1) {
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.cols; j++) {
				out[i*out.cols + j] = op(a[i*a.cols + j], b[0]);
			}
		}
    } else {
		return a;
    }
    return out;
}

/* add()
* -----
* Computes matrix addition: a + b. See matrix_general(),
* for more information on behaviour.
*
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
*
* Returns: a new matrix of a + b with (a.rows, a.cols).
*/
template <typename dtype>
Matrix<dtype> add(Matrix<dtype> a, Matrix<dtype> b) {
	auto StartTime = std::chrono::high_resolution_clock::now();
    Matrix<dtype> out = matrix_general(a, b, [](dtype x, dtype y){
		return x + y;
    });
	AddTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;

}

/* add()
* -----
* Computes matrix subtraction: a - b. See matrix_general(),
* for more information on behaviour.
*
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
*
* Returns: a new matrix of a - b with (a.rows, a.cols).
*/
template <typename dtype> 
Matrix<dtype> subtract(Matrix<dtype> a, Matrix<dtype> b) {
	auto StartTime = std::chrono::high_resolution_clock::now();
    Matrix<dtype> out = matrix_general(a, b, [](dtype x, dtype y){
		return x - y;
    });
	SubtractTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

/* mul()
* -----
* Computes matrix multiplication: a * b. See matrix_general(),
* for more information on behaviour.
*
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
*
* Returns: a new matrix of a * b with (a.rows, a.cols).
*/
template <typename dtype>
Matrix<dtype> mul(Matrix<dtype> a, Matrix<dtype> b) {
	auto StartTime = std::chrono::high_resolution_clock::now();
	MulTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	Matrix<dtype> out = matrix_general(a, b, [](dtype x, dtype y){
		return x * y;
    });
	return out;

}

/* division()
* -----
* Computes matrix division: a / b. See matrix_general(),
* for more information on behaviour.
*
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
*
* Returns: a new matrix of a / b with (a.rows, a.cols).
*/
template <typename dtype> 
Matrix<dtype> division(Matrix<dtype> a, Matrix<dtype> b) {
	auto StartTime = std::chrono::high_resolution_clock::now();
	Matrix<dtype> out = matrix_general(a, b, [](dtype x, dtype y){
		return x / y;
    });
	DivisionTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;

}

/* add()
* -----
* Computes matrix equals: a == b. See matrix_general(),
* for more information on behaviour.
*
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
*
* Returns: a new matrix of a == b with (a.rows, a.cols).
*/
Matrix<int> equals(Matrix<int> a, Matrix<int> b) {
    // This is only defined for when both inputs are Matrix<int>
    // Can also utlizie |x - y| < EPISLON. 
	auto StartTime = std::chrono::high_resolution_clock::now();
	Matrix<int> out = matrix_general(a, b, [](int x, int y){
		return x == y;
    });
	EqualsTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
    return out;
}


/* exp()
* -----
* Computes element wise exponential of a matrix.
* for more information on behaviour.
*
* @a: the matrix to be exponentiated
*
* Returns: an exponentiated copy of a, with shape (a.rows, a.cols).
*/
template <typename dtype>
Matrix<dtype> exp(Matrix<dtype> a) {
	auto StartTime = std::chrono::high_resolution_clock::now();
    Matrix<dtype> out(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			out[i*out.cols + j] = std::exp(a[i*a.cols + j]);
		}
    }
	ExpTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
    return out;

}

/* exp()
* -----
* Computes element wise log of a matrix.
* for more information on behaviour.
*
* @a: the matrix to be log'd
*
* Returns: an log'd copy of a, with shape (a.rows, a.cols).
*/
template <typename dtype>
Matrix<dtype> log(Matrix<dtype> a) {
	auto StartTime = std::chrono::high_resolution_clock::now();
    Matrix<dtype> out(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			out[i*out.cols + j] = std::log(a[i*a.cols + j]);
		}
    }
	LogTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
    return out;
}

/* mul_const()
* -----
* Computes element wise a * b.
*
* @a: the matrix to be used in the calculation
* @b: the value 
*
* Returns: a new matrix of a * b with (a.rows, a.cols).
*/
template <typename dtype>
Matrix<dtype> mul_const(Matrix<dtype> a, dtype b) {
	auto StartTime = std::chrono::high_resolution_clock::now();
    Matrix<dtype> out(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			out[i*out.cols + j] = a[i*a.cols + j] * (float) b;
		}
    }
	MulConstTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
    return out;
}

/* argmax()
* -----
* Computes element argmax of a matrix across axis 1. That is
* finds the index of the maximum value in each row.
*
* @a: the matrix to be used in the calculation
*
* Returns: a matrix full of indices with shape(a.rows, 1).
*/
template <typename dtype>
Matrix<int> argmax(Matrix<dtype> a) {
    // Finds idx of max value across axis=1 (rows)
	auto StartTime = std::chrono::high_resolution_clock::now();
    Matrix<int> out(a.rows, 1);
    dtype max;
    for (int i = 0; i < a.rows; i++) {
		max = a[i*a.cols];
		for (int  j = 1; j < a.cols; j++) {
			if (max < a[i*a.cols + j]) {
				out[i] = j;
			}
		}
    }
	ArgmaxTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
    return out;
}

int loop_case(int arows, int acols, int brows, int bcols) {
	if (arows == brows && acols == bcols) {
		return 3; // b[i*a.cols + j]
    } else if (acols == bcols && brows == 1) {
		return 2; // b[j]
    } else if (arows == brows && bcols == 1) {
		return 1; // b[i]
    } else if (brows == 1 && bcols == 1) { 
		return 0; // b[0]
	} else {
		return -1; // not valid
	}
}


/* dot()
* -----
* Computes the dot product between two matrices. That is a ⋅ b. 
* a.cols must equal b.rows. 
*
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
*
* Returns: the matrix containing the dot product of a & b.	
* 		Will be of shape (a.rows, b.cols)
*/
template <typename dtype>
Matrix<dtype, dtype*> dot(Matrix<dtype, dtype*> a, Matrix<dtype, dtype*> b) {
    // a should be inputs
    // b should be weights
    Matrix<dtype, dtype*> out(a.rows, b.cols);
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 32;
	dim3 block(threads, threads);
    dim3 grid((a.rows+threads-1)/threads, (b.cols+threads-1)/threads);
	cuda::dot<<<grid, block>>>(a.rows, b.cols, b.rows, a.get_matrix(), b.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	DotTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}


template <typename dtype>
Matrix<dtype, dtype*> max(Matrix<dtype, dtype*> a) {	
	Matrix<dtype, dtype*> out(a.rows, 1, (dtype) -INT_MAX);
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 1024;
	int blocks = (a.rows+threads-1)/threads;
	cuda::max<<<blocks, threads>>>(a.rows, a.cols, a.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	MaxTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

template <typename dtype>
Matrix<dtype, dtype*> sum(Matrix<dtype, dtype*> a, int axis, bool keepdims) {
	int threads = 1024;
	if (keepdims) {
		if (axis) {
			Matrix<dtype, dtype*> out(a.rows, 1); 
			auto StartTime = std::chrono::high_resolution_clock::now();
			int blocks = (a.rows+threads-1)/threads;
			cuda::sum_keepdims_1<<<blocks, threads>>>(a.rows, a.cols, a.get_matrix(), out.get_matrix());
			_FREE.push_back(out.get_matrix());
			SumTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
			return out;
		} else {
			Matrix<dtype, dtype*> out(1, a.cols);
			auto StartTime = std::chrono::high_resolution_clock::now();
			int blocks = (a.cols+threads-1)/threads;
			cuda::sum_keepdims_0<<<blocks, threads>>>(a.rows, a.cols, a.get_matrix(), out.get_matrix());
			_FREE.push_back(out.get_matrix());
			SumTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
			return out;
		}
    } else {
		int blocks = (a.cols*a.rows + threads - 1) / threads;
		Matrix<dtype, dtype*> out(1, blocks);
		auto StartTime = std::chrono::high_resolution_clock::now();
		cuda::sum_reduce<<<blocks, threads, sizeof(dtype) * threads>>>(a.rows*a.cols, a.get_matrix(), out.get_matrix());
		while (blocks > 1) {
			// Largest N in current dataset is 1000, so this will never get called THIS MAY NOT WORK!!!. 
			blocks = (out.cols + threads - 1) / threads;
			cuda::sum_reduce<<<blocks, threads,  sizeof(dtype) * threads>>>(out.cols, out.get_matrix(), out.get_matrix());
			out.cols = blocks;
		}
		_FREE.push_back(out.get_matrix());
		SumTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
		return out;
    }
}

template <typename dtype>
Matrix<dtype, dtype*> transpose(Matrix<dtype, dtype*> a) {
	Matrix<dtype, dtype*> out(a.cols, a.rows); 
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 32;
	dim3 block(threads, threads);
	dim3 grid((a.rows+threads-1)/threads, (a.cols+threads-1)/threads);
	cuda::transpose<<<grid, block>>>(a.rows, a.cols, a.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	TransposeTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

template <typename dtype>
Matrix<dtype, dtype*> add(Matrix<dtype, dtype*> a, Matrix<dtype, dtype*> b) {
	Matrix<dtype, dtype*> out(a.rows, a.cols); 
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 32;
	dim3 block(threads, threads);
	dim3 grid((a.rows+threads-1)/threads, (a.cols+threads-1)/threads);
	int loop = loop_case(a.rows, a.cols, b.rows, b.cols);
	cuda::add<<<grid, block>>>(a.rows, a.cols, loop, a.get_matrix(), b.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
		AddTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

template <typename dtype>
Matrix<dtype, dtype*> subtract(Matrix<dtype, dtype*> a, Matrix<dtype, dtype*> b) {
	Matrix<dtype, dtype*> out(a.rows, a.cols); 
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 32;
	dim3 block(threads, threads);
	dim3 grid((a.rows+threads-1)/threads, (a.cols+threads-1)/threads);
	int loop = loop_case(a.rows, a.cols, b.rows, b.cols);
	cuda::subtract<<<grid, block>>>(a.rows, a.cols, loop, a.get_matrix(), b.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	SubtractTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

template <typename dtype>
Matrix<dtype, dtype*> mul(Matrix<dtype, dtype*> a, Matrix<dtype, dtype*> b) {
	Matrix<dtype, dtype*> out(a.rows, a.cols); 
	auto StartTime = std::chrono::high_resolution_clock::now();	
	int threads = 32;
	dim3 block(threads, threads);
	dim3 grid((a.rows+threads-1)/threads, (a.cols+threads-1)/threads);
	int loop = loop_case(a.rows, a.cols, b.rows, b.cols);
	cuda::mul<<<grid, block>>>(a.rows, a.cols, loop, a.get_matrix(), b.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	MulTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

template <typename dtype>
Matrix<dtype, dtype*> division(Matrix<dtype, dtype*> a, Matrix<dtype, dtype*> b) {
	Matrix<dtype, dtype*> out(a.rows, a.cols); 
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 32;
	dim3 block(threads, threads);
	dim3 grid((a.rows+threads-1)/threads, (a.cols+threads-1)/threads);
	int loop = loop_case(a.rows, a.cols, b.rows, b.cols);
	cuda::division<<<grid, block>>>(a.rows, a.cols, loop, a.get_matrix(), b.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	DivisionTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

Matrix<int, int*> equals(Matrix<int, int*> a, Matrix<int, int*> b) {
	Matrix<int, int*> out(a.rows, a.cols); 
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 32;
	dim3 block(threads, threads);
	dim3 grid((a.rows+threads-1)/threads, (a.cols+threads-1)/threads);
	int loop = loop_case(a.rows, a.cols, b.rows, b.cols);
	cuda::cuda_equals<<<grid, block>>>(a.rows, a.cols, loop, a.get_matrix(), b.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	EqualsTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

template <typename dtype>
Matrix<dtype, dtype*> exp(Matrix<dtype, dtype*> a) {
	Matrix<dtype, dtype*> out(a.rows, a.cols); 
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 32;
	dim3 block(threads, threads);
	dim3 grid((a.rows+threads-1)/threads, (a.cols+threads-1)/threads);
	cuda::cuda_exp<<<grid, block>>>(a.rows, a.cols, a.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	ExpTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

template <typename dtype>
Matrix<dtype, dtype*> log(Matrix<dtype, dtype*> a) {
	Matrix<dtype, dtype*> out(a.rows, a.cols); 
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 32;
	dim3 block(threads, threads);
	dim3 grid((a.rows+threads-1)/threads, (a.cols+threads-1)/threads);
	cuda::cuda_log<<<grid, block>>>(a.rows, a.cols, a.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	LogTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

template <typename dtype>
Matrix<dtype, dtype*> mul_const(Matrix<dtype, dtype*> a, dtype b) {
	Matrix<dtype, dtype*> out(a.rows, a.cols); 
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 32;
	dim3 block(threads, threads);
	dim3 grid((a.rows+threads-1)/threads, (a.cols+threads-1)/threads);
	cuda::mul_const<<<grid, block>>>(a.rows, a.cols, b, a.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	MulConstTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

template <typename dtype>
Matrix<int, int*> argmax(Matrix<dtype, dtype*> a) {
	Matrix<int, int*> out(a.rows, 1);
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 1024;
	int blocks = (a.rows+threads-1)/threads;
	cuda::argmax<<<blocks, threads>>>(a.rows, a.cols, a.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	ArgmaxTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

template <typename dtype>
Matrix<dtype, dtype*> relu_fwd(Matrix<dtype, dtype*> a) {
	Matrix<dtype, dtype*> out(a.rows, a.cols); 
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 32;
	dim3 block(threads, threads);
	dim3 grid((a.rows+threads-1)/threads, (a.cols+threads-1)/threads);
	cuda::relu_fwd<<<grid, block>>>(a.rows, a.cols, a.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	ReluFwdTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

template <typename dtype>
Matrix<dtype, dtype*> relu_bwd(Matrix<dtype, dtype*> a, Matrix<dtype, dtype*> b) {
	Matrix<dtype, dtype*> out(a.rows, a.cols); 
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 32;
	dim3 block(threads, threads);
	dim3 grid((a.rows+threads-1)/threads, (a.cols+threads-1)/threads);
	cuda::relu_bwd<<<grid, block>>>(a.rows, a.cols, a.get_matrix(), b.get_matrix(), out.get_matrix());
	_FREE.push_back(out.get_matrix());
	ReluBwdTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return out;
}

template <typename dtype>
Matrix<dtype, dtype*> softmax_bwd(Matrix<dtype, dtype*> a, Matrix<int, int*> b) {
	Matrix<dtype, dtype*> out(a.rows, a.cols); 
	auto StartTime = std::chrono::high_resolution_clock::now();
	int threads = 1024;
	int blocks = (a.rows+threads-1)/threads;
	cuda::softmax_bwd<<<blocks, threads>>>(a.rows, a.cols, a.get_matrix(), b.get_matrix(), a.get_matrix());
	_FREE.push_back(out.get_matrix());
	SoftMaxBwdTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - StartTime );
	return a;
}

void _free() {
	while (!_FREE.empty()) {
		cuda::checkError(cudaFree(_FREE.back()));
		_FREE.pop_back();
	}
	// while (!_FREEI.empty()) {
	// 	cuda::checkError(cudaFree(_FREEI.back()));
	// 	_FREEI.pop_back();
	// }
}

}
