#include "optimizers.h"
#include "layers.h"
#include "matrix.h"

namespace optimizer 
{

/* SGD::SGD()
* -----
* Constructs a new SGD Object. SGD = Stochastic Gradient Descent, 
* SGD is a fairly common approach for optimizing the gradient descent
* processes. It achieves faster iterations over traditional gradient descent
* but a lower convergence rate. Momemtum is yet to be implemented.       
*
* @learing_rate: the starting learning rate
* @decay: the amount the learning rate should decay after each 
*       iteration. 0.001 is standard.
*/
SGD::SGD(double learning_rate, double decay): learning_rate(learning_rate), 
	decay(decay) { 
    iterations = 0;
    current_learning_rate = learning_rate;
}

/* SGD::pre_update()
* -----
* Reduces the current learning rate by a %. Should be called
* after every iteration and before every update. 
*/
void SGD::pre_update() {
    current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
}

/* SGD::update()
* -----
* Updates the weights and biases for the supplied layer. The layer must
* have been through the back propogation algorithm to assertian it's 
* partial derivatives. Should be called at every iteration on every 
* dense layer.
* 
* @layer: the layer to be updated
*/
void SGD::update(Dense* layer) {
    matrix::Matrix<double> weights_updates = matrix::mul_const((*layer).get_dweights(), -current_learning_rate);
    matrix::Matrix<double> biases_updates = matrix::mul_const((*layer).get_dbiases(), -current_learning_rate);
    
    (*layer).set_weights(matrix::add(weights_updates, ((*layer).get_weights())));
    (*layer).set_biases(matrix::add(biases_updates, (*layer).get_biases())); 
}

/* SGD::post_update()
* -----
* Increases the number of iterations by one. Should be
* called after every iteration. Important for decaying
* the learning rate.
*/
void SGD::post_update() {
    iterations += 1;
}

/* SGD::get_lr()
* -----
* Getter for the current_learning_rate. That is. 
* the learning rate after decay. 
*
* Returns: the private member current_learning_rate.
*/
double SGD::get_lr() {
    return current_learning_rate;
}

}
