#include "optimizers.h"
#include "layers.h"
#include "matrix.h"

namespace optimizer 
{

SGD::SGD(float learning_rate, float decay): learning_rate(learning_rate),
    		decay(decay), current_learning_rate(learning_rate) {}

void SGD::update(Dense layer) {
    current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
    matrix::Matrix<float> weight_updates = matrix::mul_const(layer.get_dweights(), 
	    - current_learning_rate);
    matrix::Matrix<float> bias_updates = matrix::mul_const(layer.get_dbiases(), 
	    - current_learning_rate);
    layer.set_weights(matrix::add(weight_updates, layer.get_weights()));
    layer.set_biases(matrix::add(biases_updates, layer.get_biases())); 
}


}
