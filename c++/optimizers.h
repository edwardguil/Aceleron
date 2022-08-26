#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H
#include "matrix.h"
namespace optimizer
{
class SGD {
    int iterations;
    float learning_rate;
    float decay;

public: 
    SGD(float learning_rate=1, float decay=0);
}
#endif
