#include <cstdlib>
#include <random>
#include <cstdio>
#include "matrix.h"


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
template<typename dtype, typename vtype>
Dense<dtype, vtype>::Dense(int n_inputs, int n_neurons, bool randomize): 
            weights(n_inputs, n_neurons), biases(1, n_neurons, 1, true), 
            dweights(n_inputs, n_neurons), dbiases(1, n_neurons) 
            { 
       randomize_weights(randomize);
}

/* Dense::randomize_weights()
* -----
* Randomizes the layers weights using a uniform distribution.
*/
template<typename dtype, typename vtype>
void Dense<dtype, vtype>::randomize_weights(bool randomize) {
    std::default_random_engine generator;
    std::uniform_real_distribution<dtype> distribution(0.0,1.0);
    for (int i = 0; i < weights.rows; i++) {
        for (int j = 0; j < weights.cols; j++) {
            weights[i*weights.cols + j] = randomize ? (double) distribution(generator) : (double) 0.001 * i;
        }
    }
}

/* Dense::randomize_weights()
* -----
* Randomizes the layers weights using a uniform distribution.
*/
template<>
void Dense<double, double*>::randomize_weights(bool randomize) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    std::vector<double> randomized(weights.rows * weights.cols);
    for (int i = 0; i < weights.rows; i++) {
        for (int j = 0; j < weights.cols; j++) {
            randomized[i*weights.cols + j] = randomize ? distribution(generator) : (double) 0.001 * i;
        }
    }
    weights.set_matrix(&(randomized[0]));
}


/* Dense::forward()
* -----
* Passes the input through the layer. Computes the dot product
* between input and transposed weights, then add biases. 
*
* @input: the input to be passed through the layer 

* Returns: the resulting matrix after performing operations.
*/
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> Dense<dtype, vtype>::forward(matrix::Matrix<dtype, vtype>& input) {
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
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> Dense<dtype, vtype>::backward(matrix::Matrix<dtype, vtype>& inputs, 
	    matrix::Matrix<dtype, vtype>& dinput) {
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
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> Dense<dtype, vtype>::get_dweights() {
    return dweights;
}

/* Dense::get_dbiases()
* -----
* Getter for the private member variable dbiases.
*
* Returns: the member variable dbiases
*/
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> Dense<dtype, vtype>::get_dbiases() {
    return dbiases;
}

/* Dense::get_biases()
* -----
* Getter for the private member variable biases
*
* Returns: the member variable biases
*/
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> Dense<dtype, vtype>::get_biases() {
    return biases;
}

/* Dense::get_weights()
* -----
* Getter for the private member variable weights
*
* Returns: the member variable weights
*/
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> Dense<dtype, vtype>::get_weights() {
    return weights;
}

/* Dense::set_biases()
* -----
* Setter for the private member variable biases
* 
* @new_biases: the new biases to be set for the layer
*/
template<typename dtype, typename vtype>
void Dense<dtype, vtype>::set_biases(matrix::Matrix<dtype, vtype> new_biases) {
    biases = new_biases;
}

/* Dense::set_biases()
* -----
* Setter for the private member variable biases
* 
* @new_biases: the new biases to be set for the layer
*/
template<>
void Dense<double, double*>::set_biases(matrix::Matrix<double, double*> new_biases) {
    cuda::checkError(cudaFree(biases.get_matrix()));
    biases = new_biases.copy();
}


/* Dense::set_weights()
* -----
* Setter for the private member variable weights
* 
* @new_weights: the new weights to be set for the layer
*/
template<typename dtype, typename vtype>
void Dense<dtype, vtype>::set_weights(matrix::Matrix<dtype, vtype> new_weights) {
    weights = new_weights;
}

/* Dense::set_weights()
* -----
* Setter for the private member variable weights
* 
* @new_weights: the new weights to be set for the layer
*/
template<>
void Dense<double, double*>::set_weights(matrix::Matrix<double, double*> new_weights) {
    cuda::checkError(cudaFree(weights.get_matrix()));
    weights = new_weights.copy();
}


// ------------- RELU  -------------- //
template<typename dtype, typename vtype>
ReLU<dtype, vtype>::ReLU(void) {}

/* ReLU::forward()
* -----
* Passes the input through the layer. Any values
* less than zero in the parsed input will be set to zero. 
* Note for memory efficency, this edits the passed matrix.
* @input: the input to be passed through the layer 
*
* Returns: the resulting matrix after performing operations.
*/
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> ReLU<dtype, vtype>::forward(matrix::Matrix<dtype, vtype>& input) {
    matrix::Matrix<dtype, vtype> out(input.rows, input.cols);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            if (input[i*input.cols + j] < 0) {
                out[i*input.cols + j] = 0;
            } else {
                out[i*input.cols + j]  = input[i*input.cols + j];
            }
        }
    }
    return out;
}

/* ReLU::forward()
* -----
* Passes the input through the layer. Any values
* less than zero in the parsed input will be set to zero. 
* @input: the input to be passed through the layer 

* Returns: the resulting matrix after performing operations.
*/
template<>
matrix::Matrix<double, double*> ReLU<double, double*>::forward(matrix::Matrix<double, double*>& input) {
    return matrix::relu_fwd(input);
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
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> ReLU<dtype, vtype>::backward(matrix::Matrix<dtype, vtype>& inputs, 
	    matrix::Matrix<dtype, vtype>& dinput) {
    for (int i = 0; i < dinput.rows; i++) {
        for (int j = 0; j < dinput.cols; j++) {
            if (inputs[i*inputs.cols + j] <= 0) {
            dinput[i*dinput.cols + j] = 0;
            }
        }
    }
    return dinput;
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
template<>
matrix::Matrix<double, double*> ReLU<double, double*>::backward(matrix::Matrix<double, double*>& inputs, 
	    matrix::Matrix<double, double*>& dinput) {
    return matrix::relu_bwd(inputs, dinput);
}


// ------------- SOFTMAX -------------- //
template<typename dtype, typename vtype>
Softmax<dtype, vtype>::Softmax(void) {}

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
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> Softmax<dtype, vtype>::forward(matrix::Matrix<dtype, vtype>& input) {
    matrix::Matrix<dtype, vtype> temp = matrix::exp(matrix::subtract(input, matrix::max(input)));
    return matrix::division(temp, matrix::sum(temp, 1, true));
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
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> Softmax<dtype, vtype>::backward(matrix::Matrix<dtype, vtype>& dinput) {
    return dinput;
}

// ------------- SOFTMAX CROSSENTROPY  -------------- //

/* SoftMaxCrossEntropy::SoftMaxCrossEntropy()
* -----
* Combines the Softmax and CrossEntropyLoss into one layer. The purpose
* of this connection is simplify the calculations of partial derivative. 
*/
template<typename dtype, typename vtype>
SoftmaxCrossEntropy<dtype, vtype>::SoftmaxCrossEntropy(void): softmax(), crossEntropy() {}

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
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> SoftmaxCrossEntropy<dtype, vtype>::forward(matrix::Matrix<dtype, vtype>& input, 
	    matrix::Matrix<dtype, vtype>& y_true) {
    matrix::Matrix<dtype, vtype> out = softmax.forward(input);
    loss = crossEntropy.calculateLoss(y_true, out);
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
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> SoftmaxCrossEntropy<dtype, vtype>::backward(matrix::Matrix<dtype, vtype>& dinput, 
	    matrix::Matrix<dtype, vtype>& y_true) {
    // Expects y_true to be one hot encoded
    matrix::Matrix<int> converted = matrix::argmax(y_true);
    for (int i = 0; i < dinput.rows; i++) {
        dinput[i*dinput.cols + converted[i]] -= 1;
    }
    matrix::Matrix<dtype, vtype> temp(1, 1, dinput.rows);
    return matrix::division(dinput, temp);
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
template<>
matrix::Matrix<double, double*> SoftmaxCrossEntropy<double, double*>::backward(matrix::Matrix<double, double*>& dinput, 
	    matrix::Matrix<double, double*>& y_true) {
    // Expects y_true to be one hot encoded
    matrix::Matrix<int, int*> converted = matrix::argmax(y_true);
    matrix::Matrix<double, double*> temp(1, 1, dinput.rows, true);
    return matrix::division(matrix::softmax_bwd(dinput, converted), temp);
}

/* SoftmaxCrossEntropy::get_loss()
* -----
* Getter for the private member variable loss
*
* Returns: the member variable loss
*/
template<typename dtype, typename vtype>
double SoftmaxCrossEntropy<dtype, vtype>::get_loss() {
    return loss;
}
