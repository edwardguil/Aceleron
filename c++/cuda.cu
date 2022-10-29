
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