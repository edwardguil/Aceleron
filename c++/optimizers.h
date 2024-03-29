#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H
#include "matrix.h"
#include "layers.h"
namespace optimizer
{
template <typename dtype = double, typename vtype = std::vector<double>>
class SGD {
    int iterations;
    double learning_rate;
    double decay;
    double current_learning_rate;

public: 
    SGD(double learning_rate, double decay);
    void pre_update();
    void update(Dense<dtype, vtype>* layer);
    void post_update();
    double get_lr();
};

}

#include "optimizers.cu"

#endif
