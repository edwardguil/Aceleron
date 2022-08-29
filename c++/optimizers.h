#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H
#include "matrix.h"
#include "layers.h"
namespace optimizer
{
class SGD {
    int iterations;
    float learning_rate;
    float decay;
    float current_learning_rate;

public: 
    SGD(float learning_rate, float decay);
    void pre_update();
    void update(Dense* layer);
    void post_update();
};

}
#endif
