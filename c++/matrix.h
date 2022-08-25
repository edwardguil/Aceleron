#ifndef MATRIX_H
#define MATRIX_H

#include <vector>



namespace matrix 
{
template <typename dtype>
class Matrix {

    std::vector<std::vector<dtype>> matrix;

public:
    int rows;
    int cols;

    Matrix(int rows, int cols, dtype value = 0); 

    Matrix();

    std::vector<dtype>& operator[](int i);

    unsigned long size();

    void set_matrix(std::vector<std::vector<dtype>> update);

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

#include "matrix.tpp"

#endif
