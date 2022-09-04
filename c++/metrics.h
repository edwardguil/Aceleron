#ifndef METRICS_H
#define METRICS_H
#include "matrix.h"

namespace metric
{
    double accuracy(matrix::Matrix<double> y_true, matrix::Matrix<double> y_pred);
}

#endif
