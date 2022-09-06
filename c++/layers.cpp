#include <cstdlib>
#include <random>
#include <cstdio>
#include "matrix.h"
#include "layers.h"


// ------------- DENSE  -------------- //

/* Dense::Dense()
* -----
* Constructs a new Dense layer. Intializses the weights, 
* biases, dweights and dbiases. Biases are initialized to one,
* weights are randomized. 
*
* @n_inputs: the number of features/dimensions of the input data (after flattening)
* @n_neurons: the number of neurons to be contained in the layer
*/
Dense::Dense(int n_inputs, int n_neurons): 
            weights(n_inputs, n_neurons), biases(1, n_neurons, 1), 
            dweights(n_inputs, n_neurons), dbiases(1, n_neurons, 1) 
            { 
       randomize_weights();
}

/* Dense::randomize_weights()
* -----
* Randomizes the layers weights using a uniform distribution.
*/
void Dense::randomize_weights() {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for (int i = 0; i < weights.rows; i++) {
        for (int j = 0; j < weights.cols; j++) {
            weights[i][j] = (double) distribution(generator);
        }
    }
}

/* Dense::forward()
* -----
* Passes the input through the layer. Computes the dot product
* between input and transposed weights, then add biases. 
*
* @input: the input to be passed through the layer 

* Returns: the resulting matrix after performing operations.
*/
matrix::Matrix<double> Dense::forward(matrix::Matrix<double>& input) {
    // Calculate the dot product between each neuron and input data
    return matrix::add(matrix::dot(input, weights), biases);
}

/* Dense::backward()
* -----
* Computes the partial derivate w.r.t the weights and biases utilising 
* the matrix of derivatives. Stores the results in member variables.
*
* @inputs: the input that was passed through matrix in the previous
*       forward pass
* @dinput: the derivative matrix passed from the next layer
*
* Returns: the partial derivative w.r.t the layers output
*/
matrix::Matrix<double> Dense::backward(matrix::Matrix<double>& inputs, 
	    matrix::Matrix<double>& dinput) {
    dweights = matrix::dot(matrix::transpose(inputs), dinput);
    dbiases = matrix::sum(dinput, 0, true);
    return matrix::dot(dinput, matrix::transpose(weights));
}

/* Dense::get_dweights()
* -----
* Getter for the private member variable dweights.
*
* Returns: the member variable dweights
*/
matrix::Matrix<double> Dense::get_dweights() {
    return dweights;
}

/* Dense::get_dbiases()
* -----
* Getter for the private member variable dbiases.
*
* Returns: the member variable dbiases
*/
matrix::Matrix<double> Dense::get_dbiases() {
    return dbiases;
}

/* Dense::get_biases()
* -----
* Getter for the private member variable biases
*
* Returns: the member variable biases
*/
matrix::Matrix<double> Dense::get_biases() {
    return biases;
}

/* Dense::get_weights()
* -----
* Getter for the private member variable weights
*
* Returns: the member variable weights
*/
matrix::Matrix<double> Dense::get_weights() {
    return weights;
}

/* Dense::set_biases()
* -----
* Setter for the private member variable biases
* 
* @new_biases: the new biases to be set for the layer
*/
void Dense::set_biases(matrix::Matrix<double> new_biases) {
    biases = new_biases;
}

/* Dense::set_weights()
* -----
* Setter for the private member variable weights
* 
* @new_weights: the new weights to be set for the layer
*/
void Dense::set_weights(matrix::Matrix<double> new_weights) {
    weights = new_weights;
}

// ------------- RELU  -------------- //

ReLU::ReLU(void) {}

/* ReLU::forward()
* -----
* Passes the input through the layer. Any values
* less than zero in the parsed input will be set to zero. 
* @input: the input to be passed through the layer 

* Returns: the resulting matrix after performing operations.
*/
matrix::Matrix<double> ReLU::forward(matrix::Matrix<double>& input) {
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            if (input[i][j] < 0) {
            input[i][j] = 0;
            }
        }
    }
    return input;
}

/* ReLU::backward()
* -----
* Computes the partial derivate w.r.t the relus output, utilising 
* the matrix of derivatives. 
*
* @inputs: the input that was passed through matrix in the previous
*       forward pass
* @dinput: the derivative matrix passed from the next layer
*
* Returns: the partial derivative w.r.t the relus output
*/
matrix::Matrix<double> ReLU::backward(matrix::Matrix<double>& inputs, 
	    matrix::Matrix<double>& dinput) {
    for (int i = 0; i < dinput.rows; i++) {
        for (int j = 0; j < dinput.cols; j++) {
            if (inputs[i][j] <= 0) {
            dinput[i][j] = 0;
            }
        }
    }
    return dinput;
}

// ------------- SOFTMAX -------------- //

Softmax::Softmax(void) {}

/* SoftMax::forward()
* -----
* Passes the input through the layer.  Converts each vector 
* of K real numbers into a probability distribution of K possible 
* outcomes. Generally, should be used as the final activation 
* function in Categorical cases.
*
* @input: the input to be passed through the layer 
*
* Returns: the resulting matrix after performing operations.
*/
matrix::Matrix<double> Softmax::forward(matrix::Matrix<double>& input) {
    matrix::Matrix<double> temp = matrix::exp(matrix::subtract(input, 
		matrix::max(input)));
    return matrix::division(temp, matrix::sum(temp));
}

/* SoftMax::backward()
* -----
* Computes the partial derivate w.r.t the Softmax output, utilising 
* the matrix of derivatives. This function has not been implemented.
* Do not use in current state.
*
* @dinput: the derivative matrix passed from the next layer
*
* Returns: the derivative matrix passed
*/
matrix::Matrix<double> Softmax::backward(matrix::Matrix<double>& dinput) {
    return dinput;
}

// ------------- SOFTMAX CROSSENTROPY  -------------- //

/* SoftMaxCrossEntropy::SoftMaxCrossEntropy()
* -----
* Combines the Softmax and CrossEntropyLoss into one layer. The purpose
* of this connection is simplify the calculations of partial derivative. 
*/
SoftmaxCrossEntropy::SoftmaxCrossEntropy(void): softmax(), crossEntropy() {}

/* SoftmaxCrossEntropy::forward()
* -----
* Passes the input through the layer and computes loss. Values 
* are first passed through Softmax, then the loss is 
* calculated using CategoricalCrossEntropy. Loss is stored
* in private member variable loss.
* 
* @input: the input to be passed through the layer 
* @y_true: the class labels for the input, used in loss calculation

* Returns: the resulting matrix from softmax 
*/
matrix::Matrix<double> SoftmaxCrossEntropy::forward(matrix::Matrix<double>& input, 
	    matrix::Matrix<double>& y_true) {
    matrix::Matrix<double> out = softmax.forward(input);
    loss = crossEntropy.calculateLoss(out, y_true);
    return out;
}


/* SoftmaxCrossEntropy::backward()
* -----
* Computes the partial derivate w.r.t output of this layer.
*
* @dinput: the output from the forward pass of this layer.
* @y_true: the class labels for the input
*
* Returns: the partial derivative w.r.t the layers output
*/
matrix::Matrix<double> SoftmaxCrossEntropy::backward(matrix::Matrix<double>& dinput, 
	    matrix::Matrix<double>& y_true) {
    // Expects y_true to be one hot encoded
    matrix::Matrix<int> converted = matrix::argmax(y_true);
    for (int i = 0; i < dinput.rows; i++) {
        dinput[i][converted[i][0]] -= 1;
    }
    matrix::Matrix<double> temp(1, 1, dinput.rows);
    return matrix::division(dinput, temp);
}

/* SoftmaxCrossEntropy::get_loss()
* -----
* Getter for the private member variable loss
*
* Returns: the member variable loss
*/
double SoftmaxCrossEntropy::get_loss() {
    return loss;
}
