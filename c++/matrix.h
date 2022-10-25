#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
namespace matrix 
{
template <typename dtype, typename vtype = std::vector<dtype, std::allocator<dtype>>>
class Matrix {
    vtype matrix;
public:
    int rows;
    int cols;

    Matrix(int rows, int cols, dtype value = 0);
    Matrix();

    dtype& operator[](int i);

    void set_matrix(vtype update);

    int size();

    vtype get_matrix();
    void get_matrix(Matrix<dtype>);
    
    Matrix<dtype> copy();
};

template <typename dtype>
void print(Matrix<dtype> matrix);

template <typename dtype>
Matrix<dtype> dot(Matrix<dtype> a, Matrix<dtype> b);

template <typename dtype>
Matrix<dtype> max(Matrix<dtype> input);

template <typename dtype>
Matrix<dtype> sum(Matrix<dtype> input, int axis = 1, bool keepdims = true);

template <typename dtype>
Matrix<dtype> transpose(Matrix<dtype> input);

template <typename dtype, class Operator>
Matrix<dtype> general(Matrix<dtype> a, Matrix<dtype> b, Operator op);

template <typename dtype> 
Matrix<dtype> add(Matrix<dtype> a, Matrix<dtype> b);

template <typename dtype> 
Matrix<dtype> subtract(Matrix<dtype> a, Matrix<dtype> b);

template <typename dtype>
Matrix<dtype> mul(Matrix<dtype> a, Matrix<dtype> b);

template <typename dtype> 
Matrix<dtype> division(Matrix<dtype> a, Matrix<dtype> b);

template <typename dtype>
Matrix<dtype> mul_const(Matrix<dtype> a, dtype b);

template <typename dtype>
Matrix<dtype> exp(Matrix<dtype> a);

template <typename dtype>
Matrix<dtype> log(Matrix<dtype> a);

template <typename dtype>
Matrix<int> argmax(Matrix<dtype> a);

}

#include "matrix.cu"

#endif
