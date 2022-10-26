#ifndef CUDA_H
#define CUDA_H

namespace cuda {

void checkError(cudaError_t e);

__global__
void dot(int a_rows, int b_cols, int common, double* a, double* b, double* c);

__global__
void max(int a_rows, int a_cols, double* a, double* b);

__global__
void sum_keepdims_0(int a_rows, int a_cols, double* a, double* b);

__global__
void sum_keepdims_1(int a_rows, int a_cols, double* a, double* b);

__global__
void sum_reduce(double* a, double* b);

__global__
void add(int a_rows, int a_cols, int loop, double* a, double* b, double* c);

__global__
void subtract(int a_rows, int a_cols, int loop, double* a, double* b, double* c);

__global__
void mul(int a_rows, int a_cols, int loop, double* a, double* b, double* c);

__global__
void division(int a_rows, int a_cols, int loop, double* a, double* b, double* c);

__global__
void mul_const(int a_rows, int a_cols, double value, double* a, double* b);

__global__
void cuda_log(int a_rows, int a_cols, double* a, double* b);

__global__
void cuda_exp(int a_rows, int a_cols, double* a, double* b);

__global__
void relu_fwd(int a_rows, int a_cols, double* a, double* b);

}
#endif