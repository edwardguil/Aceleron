#ifndef LOSSES_H
#define LOSSES_H
#include "matrix.h"
// Abstract Class
template <typename dtype, typename vtype = std::vector<dtype, std::allocator<dtype>>>
class Loss {
public: 
    // Abstract Method
    virtual matrix::Matrix<dtype, vtype> loss(matrix::Matrix<dtype, vtype> y_true, 
		matrix::Matrix<dtype, vtype> y_pred) = 0;
    
    double calculateLoss(matrix::Matrix<dtype, vtype> y_true, 
		matrix::Matrix<dtype, vtype> y_pred);
};

template <typename dtype, typename vtype = std::vector<dtype, std::allocator<dtype>>>
class CategoricalCrossentropy: public Loss<dtype, vtype> {
public: 
    matrix::Matrix<dtype, vtype> loss(matrix::Matrix<dtype, vtype> y_true, 
		matrix::Matrix<dtype, vtype> y_pred);
};

template <typename dtype, typename vtype = std::vector<dtype, std::allocator<dtype>>>
class SparseCategoricalCrossentropy: public Loss<dtype, vtype> {
public: 
    matrix::Matrix<dtype, vtype> loss(matrix::Matrix<dtype, vtype> y_true, 
		matrix::Matrix<dtype, vtype> y_pred);
};

#include "losses.cu"
#endif
