#ifndef METRICS_H
#define METRICS_H
#include "matrix.h"

namespace metric
{
    float accuracy(matrix::Matrix<float> y_true, matrix::Matrix<float> y_pred);
}

#endif
