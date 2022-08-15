#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

template <typename dtype>
class Matrix {

    std::vector<std::vector<dtype>> matrix;

public:
    int rows;
    int cols;

    Matrix(int rows, int cols, dtype value = 0); 

    std::vector<dtype>& operator[](int i);

    unsigned long size();

    void set_matrix(std::vector<std::vector<dtype>> update);
};

template <typename dtype>
void matrix_print(Matrix<dtype> matrix);

template <typename dtype>
Matrix<dtype> matrix_dot(Matrix<dtype> a, Matrix<dtype> b);

template <typename dtype>
Matrix<dtype> matrix_max(Matrix<dtype> input);

template <typename dtype>
Matrix<dtype> matrix_sum(Matrix<dtype> input);

template <typename dtype, class Operator>
Matrix<dtype> matrix_general(Matrix<dtype> a, Matrix<dtype> b, Operator op);

template <typename dtype> 
Matrix<dtype> matrix_add(Matrix<dtype> a, Matrix<dtype> b);

template <typename dtype> 
Matrix<dtype> matrix_subtract(Matrix<dtype> a, Matrix<dtype> b);

template <typename dtype>
Matrix<dtype> matrix_mul(Matrix<dtype> a, Matrix<dtype> b);

template <typename dtype> 
Matrix<dtype> matrix_division(Matrix<dtype> a, Matrix<dtype> b);


template <typename dtype>
Matrix<dtype> matrix_exp(Matrix<dtype> a);

template <typename dtype>
Matrix<dtype> matrix_log(Matrix<dtype> a);



#include "matrix.tpp"

#endif
