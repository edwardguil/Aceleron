#ifndef CUDA_H
#define CUDA_H

namespace cuda {

void checkError(cudaError_t e);


__global__
void dot(int a_rows, int b_cols, int common, double* a, double* b, double* c) ;

}
#endif