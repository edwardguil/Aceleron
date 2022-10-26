
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

__global__
void dot(int a_rows, int b_cols, int common, double* a, double* b, double* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= b_cols)) {
        return;
    }

    double sum = 0;
    for (int k = 0; k < common; ++k) {
        sum += a[i*common + k] * b[k*b_cols + j];
    }
    c[i * b_cols + j] = sum;
}

__global__
void max(int a_rows, int a_cols, double* a, double* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i

    // Ensure i not larger than respective array
    if ((i >= a_rows)) {
        return;
    }

    for (int j = 0; j < a_cols; ++j) {
        if (a[i*a_cols + j] > b[i]) {
            b[i] = a[i*a_cols + j];
        }
    }
}

__global__
void sum_keepdims_0(int a_rows, int a_cols, double* a, double* b) {
    // Provide blocks in column majour order
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i

    // Ensure i not larger than respective array
    if ((i >= a_cols)) {
        return;
    }

    for (int j = 0; j < a_rows; ++j) {
        b[i] += a[j*a_cols + i];
    }
}

__global__
void sum_keepdims_1(int a_rows, int a_cols, double* a, double* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i

    // Ensure i not larger than respective array
    if ((i >= a_rows)) {
        return;
    }

    for (int j = 0; j < a_cols; ++j) {
        b[i] += a[i*a_cols + j];
    }
}

__global__
void sum_reduce(double* a, double* b) {
extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = a[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) { 
        b[blockIdx.x] = sdata[0];
    }
}

__global__
void add(int a_rows, int a_cols, int loop, double* a, double* b, double* c) {
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
        c[i*a_cols + j] = a[i*a_cols + j] + b[0*loop];  
    } 
}

__global__
void subtract(int a_rows, int a_cols, int loop, double* a, double* b, double* c) {
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
        c[i*a_cols + j] = a[i*a_cols + j] - b[0*loop];  
    } 
}

__global__
void mul(int a_rows, int a_cols, int loop, double* a, double* b, double* c) {
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
        c[i*a_cols + j] = a[i*a_cols + j] * b[0*loop];  
    } 
}

__global__
void division(int a_rows, int a_cols, int loop, double* a, double* b, double* c) {
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
        c[i*a_cols + j] = a[i*a_cols + j] / b[0*loop];  
    } 
}

__global__
void cuda_log(int a_rows, int a_cols, double* a, double* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_rows)) {
        return;
    }
    b[i*a_cols + j] = log(a[i*a_cols + j]);
}

__global__
void cuda_exp(int a_rows, int a_cols, double* a, double* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_rows)) {
        return;
    }
    b[i*a_cols + j] = exp(a[i*a_cols + j]);
}

__global__
void relu_fwd(int a_rows, int a_cols, double* a, double* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global i
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Global j

    // Ensure i/j not larger than respective array
    if ((i >= a_rows) || (j >= a_rows)) {
        return;
    }

    if (a[i*a_cols + j] < 0) {
        b[i*a_cols + j] = 0;
    } else {
        b[i*a_cols + j] = a[i*a_cols + j];
    }
}

}