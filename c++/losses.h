#ifndef LOSSES_H
#define LOSSES_H
#include "matrix.h"
// Abstract Class
template<typename dtype = double, typename vtype = std::vector<double>>
class Loss {
public: 
    // Abstract Method
    virtual matrix::Matrix<dtype, vtype> loss(matrix::Matrix<dtype, vtype> y_true, 
		matrix::Matrix<dtype, vtype> y_pred) = 0;
    
    double calculateLoss(matrix::Matrix<dtype, vtype> y_true, 
		matrix::Matrix<dtype, vtype> y_pred);
};

template<typename dtype = double, typename vtype = std::vector<double>>
class CategoricalCrossentropy: public Loss<dtype, vtype> {
public: 
    matrix::Matrix<dtype, vtype> loss(matrix::Matrix<dtype, vtype> y_true, 
		matrix::Matrix<dtype, vtype> y_pred);
};

template<typename dtype = double, typename vtype = std::vector<double>>
class SparseCategoricalCrossentropy: public Loss<dtype, vtype> {
public: 
    matrix::Matrix<dtype, vtype> loss(matrix::Matrix<dtype, vtype> y_true, 
		matrix::Matrix<dtype, vtype> y_pred);
};

#include "losses.cu"
#endif
