
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

}