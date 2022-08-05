#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

template <typename dtype>
class Matrix {

    std::vector<std::vector<dtype>> matrix;

public:
    int rows;
    int cols;

    Matrix(int rows, int cols); 

    std::vector<dtype>& operator[](int i);

    unsigned long size();

    void set_matrix(std::vector<std::vector<dtype>> update);
};

template <typename dtype>
void print_matrix(Matrix<dtype> matrix);

template <typename dtype>
Matrix<dtype> dot(Matrix<dtype> a, Matrix<dtype> b);


template <typename dtype> 
Matrix<dtype> add(Matrix<dtype> a, Matrix<dtype> b);

#include "matrix.tpp"

#endif
