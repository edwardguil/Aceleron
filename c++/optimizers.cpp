#include "optimizers.h"
#include "layers.h"
#include "matrix.h"

namespace optimizer 
{

SGD::SGD(double learning_rate, double decay): learning_rate(learning_rate), 
	decay(decay) { 
    iterations = 0;
    current_learning_rate = learning_rate;
}

void SGD::pre_update() {
    current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
}

void SGD::update(Dense* layer) {
    matrix::Matrix<double> weights_updates = matrix::mul_const((*layer).get_dweights(), -current_learning_rate);
    matrix::Matrix<double> biases_updates = matrix::mul_const((*layer).get_dbiases(), -current_learning_rate);
    
    (*layer).set_weights(matrix::add(weights_updates, ((*layer).get_weights())));
    (*layer).set_biases(matrix::add(biases_updates, (*layer).get_biases())); 
}

void SGD::post_update() {
    iterations += 1;
}

double SGD::get_lr() {
    return current_learning_rate;
}

}
