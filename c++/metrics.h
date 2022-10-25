#ifndef METRICS_H
#define METRICS_H
#include "matrix.h"

namespace metric
{
    template<typename dtype = double, typename vtype = std::vector<double>>
    double accuracy(matrix::Matrix<dtype, vtype> y_true, matrix::Matrix<dtype, vtype> y_pred);
}

#include "metrics.cu"

#endif
