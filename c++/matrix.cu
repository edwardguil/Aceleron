#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <climits>
#include <cmath>
#include <type_traits>

namespace matrix {

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
Matrix<dtype, vtype>::Matrix(int rows, int cols, dtype value): matrix(rows*cols, value), 
		rows(rows), cols(cols) {
}

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
inline Matrix<double, double*>::Matrix(int rows, int cols, double value): 
		rows(rows), cols(cols) {
	cudaMalloc(&matrix, sizeof(double) * (rows * cols));
}

/* Matrix<dtype>::set_matrix()
* -----
* Copies the update array to the device via cudaMemCpy. The
* num of elements copied is rows * cols of this matrix.
* 
* @update: the array to copy to the device.
*/
template <>
inline void Matrix<double, double*>::set_matrix(double* update) {
    cudaMemcpy(matrix, update, sizeof(double) * (rows * cols), cudaMemcpyHostToDevice);
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
inline void Matrix<double, double*>::get_matrix(double* dest) {
    cudaMemcpy(dest, matrix, sizeof(double) * (rows * cols), cudaMemcpyDeviceToHost);
}

/* Matrix<dtype>::copy()
// 
* Deep copies this matrix object. 
* 
* Returns: a deep copy of this matrix object.
*/
template<>
inline Matrix<double, double*> Matrix<double, double*>::copy() {
    Matrix<double, double*> out(rows, cols);
    out.set_matrix(matrix);
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
    // keepdims = true
    Matrix<dtype> maxValues(a.rows, 1, (dtype) -INT_MAX); 
    for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			if (a[i*a.cols + j] > maxValues[i]) {
				maxValues[i] = a[i*a.cols + j];
			}
		}
    }
    return maxValues;
}

/* dot()
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
    // Default axis=1 and keepdims=true
    if (keepdims) {
		if (axis) {
			Matrix<dtype> out(a.rows, 1); 
			for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < a.cols; j++) {
					out[i] += a[i*a.cols + j];
				}
			}
			return out;
		} else {
			Matrix<dtype> out(1, a.cols);
			for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < a.cols; j++) {
					out[j] += a[i*a.cols + j];
				}
			}
			return out;
		}
    } else {
		Matrix<dtype> out(1, 1);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.cols; j++) {
				out[0] += a[i*a.cols + j];
			}
		}
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
    Matrix<dtype> out(a.cols, a.rows);
    for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			out[j*out.cols + i] = a[i*a.cols + j];
		}
    }
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
    return matrix_general(a, b, [](dtype x, dtype y){
		return x + y;
    });
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
    return matrix_general(a, b, [](dtype x, dtype y){
		return x - y;
    });
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
    return matrix_general(a, b, [](dtype x, dtype y){
		return x * y;
    });
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
    return matrix_general(a, b, [](dtype x, dtype y){
		return x / y;
    });
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
template <typename dtype> 
Matrix<dtype> equals(Matrix<dtype> a, Matrix<dtype> b) {
    // This is only defined for when both inputs are Matrix<int>
    // Can also utlizie |x - y| < EPISLON. 
    return matrix_general(a, b, [](dtype x, dtype y){
	return x == y;
    });
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
    Matrix<dtype> out(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			out[i*out.cols + j] = std::exp(a[i*a.cols + j]);
		}
    }
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
    Matrix<dtype> out(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			out[i*out.cols + j] = std::log(a[i*a.cols + j]);
		}
    }
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
    Matrix<dtype> out(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			out[i*out.cols + j] = a[i*a.cols + j] * (float) b;
		}
    }
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
    return out;
}

}