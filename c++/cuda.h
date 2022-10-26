#ifndef CUDA_H
#define CUDA_H

namespace cuda {

void checkError(cudaError_t e);

template <typename dtype>
__global__
void dot(int a_rows, int b_cols, int common, dtype* a, dtype* b, dtype* c);

template <typename dtype>
__global__
void max(int a_rows, int a_cols, dtype* a, dtype* b);

template <typename dtype>
__global__
void sum_keepdims_0(int a_rows, int a_cols, dtype* a, dtype* b);

template <typename dtype>
__global__
void sum_keepdims_1(int a_rows, int a_cols, dtype* a, dtype* b);

__global__
void sum_reduce(int N, double* a, double* b);

__global__
void sum_reduce(int N, int* a, int* b);

template <typename dtype>
__global__
void add(int a_rows, int a_cols, int loop, dtype* a, dtype* b, dtype* c);

template <typename dtype>
__global__
void subtract(int a_rows, int a_cols, int loop, dtype* a, dtype* b, dtype* c);

template <typename dtype>
__global__
void mul(int a_rows, int a_cols, int loop, dtype* a, dtype* b, dtype* c);

template <typename dtype>
__global__
void division(int a_rows, int a_cols, int loop, dtype* a, dtype* b, dtype* c);

__global__
void cuda_equals(int a_rows, int a_cols, int loop, int* a, int* b, int* c);

template <typename dtype>
__global__
void mul_const(int a_rows, int a_cols, dtype value, dtype* a, dtype* b);

template <typename dtype>
__global__
void cuda_log(int a_rows, int a_cols, dtype* a, dtype* b);

template <typename dtype>
__global__
void cuda_exp(int a_rows, int a_cols, dtype* a, dtype* b);

template <typename dtype>
__global__
void argmax(int a_rows, int a_cols, dtype* a, int* b);

template <typename dtype>
__global__
void relu_fwd(int a_rows, int a_cols, dtype* a, dtype* b);

}

#include "cuda.cu"

#endif