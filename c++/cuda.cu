
#include <iostream>

namespace cuda {

void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

/* dot()
* -----
* Computes the dot product between two matrices.That is a ⋅ b. 
* a.cols must equal b.rows. 
*
* @a_rows: the number of 'rows' in a
* @b_cols: the number of 'cols' in b
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
* @c: the output matrix
*
* Returns: the matrix containing the dot product of a & b.	
* 		Will be of shape (a.rows, b.cols)
*/
template <typename dtype>
__global__
void dot(int a_rows, int b_cols, int common, dtype* a, dtype* b, dtype* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= b_cols)) {
        return;
    }

    dtype sum = 0;
    for (int k = 0; k < common; ++k) {
        sum += a[i*common + k] * b[k*b_cols + j];
    }
    c[i * b_cols + j] = sum;
}

/* max()
* -----
* Finds the max value over axis 1 of a matrix. That is
* the max value in each row. Equilivent to 
* np.max(a, axis=1, keepdims=true). 
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the matrix to find the max of
* @b: the output matrix
*
* Returns: a new matrix containing the max value
* 		of each row in the parsed matrix.
*		will be of shape (a.rows, 1)
*/
template <typename dtype>
__global__
void max(int a_rows, int a_cols, dtype* a, dtype* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i

    // Ensure i not larger than respective array
    if ((i >= a_rows)) {
        return;
    }

    dtype max = a[i];
    for (int j = 0; j < a_cols; ++j) {
        if (a[i*a_cols + j] > max) {
            max = a[i*a_cols + j];
        }
    }
    b[i] = max;
}

/* sum()
* -----
* Sums a matrix over the respective axis 0, keeps the 
* same dimensions as the original matrix. 
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the matrix to find the sum
* @b: the output matrix
*
* Returns: the summed matrix of shape (1, a.cols)
*/
template <typename dtype>
__global__
void sum_keepdims_0(int a_rows, int a_cols, dtype* a, dtype* b) {
    // Provide blocks in column majour order
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i

    // Ensure i not larger than respective array
    if ((i >= a_cols)) {
        return;
    }

    dtype sum = 0;
    for (int j = 0; j < a_rows; ++j) {
        sum += a[j*a_cols + i];
    }
    b[i] = sum;
}

/* sum()
* -----
* Sums a matrix over the respective axis 1, keeps the 
* same dimensions as the original matrix. 
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the matrix to find the sum
* @b: the output matrix
*
* Returns: the summed matrix of shape (a.rows, 1)
*/
template <typename dtype>
__global__
void sum_keepdims_1(int a_rows, int a_cols, dtype* a, dtype* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i

    // Ensure i not larger than respective array
    if ((i >= a_rows)) {
        return;
    }

    dtype sum = 0;
    for (int j = 0; j < a_cols; ++j) {
        sum += a[i*a_cols + j];
    }
    b[i] = sum;
}

/* sum()
* -----
* Sums the entire matrix input matrix.
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the matrix to find the sum
* @b: the output matrix
*
* Returns: the summed matrix of shape (1, 1)
*/
__global__
void sum_reduce(int N, double* a, double* b) {
    extern __shared__ double sdata[];
    // each thread loads one element from global to shared mem
    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= N) {
        sdata[tid] = 0;
        return;
    }

    sdata[tid] = a[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    __syncthreads();
    // write result for this block to global mem
    if (tid == 0) { 
        b[blockIdx.x] = sdata[0];
    }
}

/* sum()
* -----
* Sums the entire matrix input matrix.
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the matrix to find the sum
* @b: the output matrix
*
* Returns: the summed matrix of shape (1, 1)
*/
__global__
void sum_reduce(int N, int* a, int* b) {
    extern __shared__ int idata[];
    // each thread loads one element from global to shared mem
    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= N) {
        idata[tid] = 0;
        return;
    }

    idata[tid] = a[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            idata[tid] += idata[tid + s];
        }
        __syncthreads();
    }
    __syncthreads();
    // write result for this block to global mem
    if (tid == 0) { 
        b[blockIdx.x] = idata[0];
    }
}

/* transpose()
* -----
* Transposes the provided matrix. That is rows become
* columns and columns become rows.
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the matrix to transpose
* @b: the output matrix
*
* Returns: a transposition of the provided matrix,
*		new matrix will be of size (a.cols, a.rows)
*/
template <typename dtype>
__global__
void transpose(int a_rows, int a_cols, dtype* a, dtype* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_cols)) {
        return;
    }

    b[j*a_rows + i] = a[i*a_cols + j];
}



/* add()
* -----
* Computes matrix addition: a + b according to the loop
* value. For more information on behaviour see 
* matrix::loop_case().
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @loop: the loop case, should be a value between 0 & 3
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
* @b: the output matrix
*
* Returns: a new matrix of a + b with (a.rows, a.cols).
*/
template <typename dtype>
__global__
void add(int a_rows, int a_cols, int loop, dtype* a, dtype* b, dtype* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_cols)) {
        return;
    }

    // Conditional calculations
    if (loop == 3) {
        c[i*a_cols + j] = a[i*a_cols + j] +  b[i*a_cols + j];
    } else if (loop == 2) {
		c[i*a_cols + j] = a[i*a_cols + j] + b[j];
    } else {
        c[i*a_cols + j] = a[i*a_cols + j] + b[i*loop];  
    } 
}

/* subtract()
* -----
* Computes matrix subtraction: a - b according to the loop
* value. For more information on behaviour see 
* matrix::loop_case().
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @loop: the loop case, should be a value between 0 & 3
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
* @b: the output matrix
*
* Returns: a new matrix of a - b with (a.rows, a.cols).
*/
template <typename dtype>
__global__
void subtract(int a_rows, int a_cols, int loop, dtype* a, dtype* b, dtype* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_cols)) {
        return;
    }

    // Conditional calculations
    if (loop == 3) {
        c[i*a_cols + j] = a[i*a_cols + j] -  b[i*a_cols + j];
    } else if (loop == 2) {
		c[i*a_cols + j] = a[i*a_cols + j] - b[j];
    } else {
        c[i*a_cols + j] = a[i*a_cols + j] - b[i*loop];  
    } 
}

/* mul()
* -----
* Computes matrix multiplication: a * b according to the loop
* value. For more information on behaviour see 
* matrix::loop_case().
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @loop: the loop case, should be a value between 0 & 3
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
* @b: the output matrix
*
* Returns: a new matrix of a * b with (a.rows, a.cols).
*/
template <typename dtype>
__global__
void mul(int a_rows, int a_cols, int loop, dtype* a, dtype* b, dtype* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_cols)) {
        return;
    }

    // Conditional calculations
    if (loop == 3) {
        c[i*a_cols + j] = a[i*a_cols + j] * b[i*a_cols + j];
    } else if (loop == 2) {
		c[i*a_cols + j] = a[i*a_cols + j] * b[j];
    } else {
        c[i*a_cols + j] = a[i*a_cols + j] * b[i*loop];  
    } 
}

/* division()
* -----
* Computes matrix division: a / b according to the loop
* value. For more information on behaviour see 
* matrix::loop_case().
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @loop: the loop case, should be a value between 0 & 3
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
* @b: the output matrix
*
* Returns: a new matrix of a / b with (a.rows, a.cols).
*/
template <typename dtype>
__global__
void division(int a_rows, int a_cols, int loop, dtype* a, dtype* b, dtype* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_cols)) {
        return;
    }

    // Conditional calculations
    if (loop == 3) {
        c[i*a_cols + j] = a[i*a_cols + j]  / b[i*a_cols + j];
    } else if (loop == 2) {
		c[i*a_cols + j] = a[i*a_cols + j] / b[j];
    } else {
        c[i*a_cols + j] = a[i*a_cols + j] / b[i*loop];  
    } 
}

/* cuda_equals()
* -----
* Computes matrix division: a == b according to the loop
* value. For more information on behaviour see 
* matrix::loop_case().
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @loop: the loop case, should be a value between 0 & 3
* @a: the first matrix to be used in the calculation
* @b: the second matrix to be used in the calculation
* @b: the output matrix
*
* Returns: a new matrix of a == b with (a.rows, a.cols).
*/
__global__
void cuda_equals(int a_rows, int a_cols, int loop, int* a, int* b, int* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_cols)) {
        return;
    }

    // Conditional calculations
    if (loop == 3) {
        c[i*a_cols + j] = a[i*a_cols + j] == b[i*a_cols + j];
    } else if (loop == 2) {
		c[i*a_cols + j] = a[i*a_cols + j] == b[j];
    } else {
        c[i*a_cols + j] = a[i*a_cols + j] == b[i*loop];  
    } 
}


/* mul_const()
* -----
* Computes element wise matrix mutliplication with the
* parsed value. 
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the matrix to be used in the calculation
* @b: the output matrix
*
* Returns: a new matrix of a * b with (a.rows, a.cols).
*/
template <typename dtype>
__global__
void mul_const(int a_rows, int a_cols, dtype value, dtype* a, dtype* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_cols)) {
        return;
    }
    b[i*a_cols + j] = a[i*a_cols + j] * value;
}

/* cuda_log()
* -----
* Computes element wise matrix log.
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the matrix to be used in the calculation
* @b: the output matrix
*
* Returns: a log'd copy of a, with shape (a.rows, a.cols).
*/
template <typename dtype>
__global__
void cuda_log(int a_rows, int a_cols, dtype* a, dtype* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_cols)) {
        return;
    }
    b[i*a_cols + j] = log(a[i*a_cols + j]);
}

/* cuda_log()
* -----
* Computes element wise matrix exp.
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the matrix to be used in the calculation
* @b: the output matrix
*
* Returns: a exp'd copy of a, with shape (a.rows, a.cols).
*/
template <typename dtype>
__global__
void cuda_exp(int a_rows, int a_cols, dtype* a, dtype* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_cols)) {
        return;
    }
    b[i*a_cols + j] = exp(a[i*a_cols + j]);
}


/* argmax()
* -----
* Computes element argmax of a matrix across axis 1. That is
* finds the index of the maximum value in each row.
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the matrix to be used in the calculation
* @b: the output matrix
*
* Returns: a matrix full of indices with shape(a.rows, 1).
*/
template <typename dtype>
__global__
void argmax(int a_rows, int a_cols, dtype* a, int* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i

    // Ensure i not larger than respective array
    if ((i >= a_rows)) {
        return;
    }

    int max = 0;
    for (int j = 1; j < a_cols; ++j) {
        if (a[i*a_cols + max] < a[i*a_cols + j]) {
            max = j;
        }
    }
    b[i] = max;
}


/* relu_fwd()
* -----
* Applies element wise ReLU to a matrix.
* That is any values less than zero in the matrix
* input will be set to zero. Otherwise unchanged.
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the matrix to be used in the calculation
* @b: the output matrix
*
* Returns: the relu'd matrix with shape(a.rows, a.cols).
*/
template <typename dtype>
__global__
void relu_fwd(int a_rows, int a_cols, dtype* a, dtype* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_cols)) {
        return;
    }

    if (a[i*a_cols + j] < 0) {
        b[i*a_cols + j] = 0;
    } else {
        b[i*a_cols + j] = a[i*a_cols + j];
    }
}

/* relu_bwd()
* -----
* Computes the partial derivate w.r.t the relus output, utilising 
* the matrix of derivatives. 
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the relus original output
* @b: a matrix of partial derivatives
* @c: the output matrix
*
* Returns: the partial derivative w.r.t the relus output 
            with shape(a.rows, a.cols).
*/
template <typename dtype>
__global__
void relu_bwd(int a_rows, int a_cols, dtype* a, dtype* b, dtype* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_cols)) {
        return;
    }

    if (a[i*a_cols + j] <= 0) {
        c[i*a_cols + j] = 0;
    } else {
        c[i*a_cols + j] = b[i*a_cols + j];
    }
}

/* relu_bwd()
* -----
* Computes the partial derivate w.r.t the Softmax output, utilising 
* the matrix of derivatives. 
*
* @a_rows: the number of 'rows' in a
* @a_cols: the number of 'cols' in a
* @a: the output from the forward pass of the softmax function
* @b: the class labels for the input
* @c: the output matrix
*
* Returns: the partial derivative w.r.t the layers output
            with shape(a.rows, a.cols).
*/
template <typename dtype>
__global__
void softmax_bwd(int a_rows, int a_cols, dtype* a, int* b, dtype* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    // Ensure i not larger than respective array
    if ((i >= a_rows)) {
        return;
    }
    c[i*a_cols + b[i]] = a[i*a_cols + b[i]] - 1;
}

}