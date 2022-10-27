#include "matrix.h"
#include "losses.h"
#include "layers.h"
#include "metrics.h"
#include "optimizers.h"
#include "cuda.h"
#include "data/data.h"

using namespace matrix;

void handle_input(Matrix<double>& x_train, Matrix<double>& y_train, 
		Matrix<double>& x_test, Matrix<double>& y_test, int N);

int handle_N(int N);

/* main()
* -----
* The main entry point for this program. Creates and runs 
* an Artificial Neural Network. Architecture constists of: 
* 			- Dense(2, 16)
* 			- ReLU()
* 			- Dense(16, 2)
* 			- SoftMax()
* 
* A single command line argument can be parsed to control the
* size of the input data to the neural network. Increments
* of 100 from 100-1000. 
*
* @argc: The length of argv
* @argv: Inputs from the command line
*
* Returns: 1 if the program failed, 0 if successful.
*/
int main(int argc, char *argv[]) {
	// Some code to handle command line input
	int N = argc > 1 ? handle_N(std::stoi(argv[1])) : 1000;
	std::cout << "N: " << N << std::endl;
	int train_size = N * 0.8;

	// Define the training and testing Matrixes
    Matrix<double> x_train(train_size, 2);
    Matrix<double> y_train(train_size, 2);
    Matrix<double> x_test(N - train_size, 2);
    Matrix<double> y_test(N - train_size, 2);
	// This function allocates the data to these matrices
	handle_input(x_train, y_train, x_test, y_test, N);



	// Construct our network
    Dense<> layer1(2, 16);
    ReLU<> layer2;
    Dense<> layer3(16, 2);
    SoftmaxCrossEntropy<> layer4;
    optimizer::SGD<> sgd(1.0, 0.001);


	// // START Test code -----------------
	// Dense layer1(2, 4);
	
	// // Create cuda matrices
	// Matrix<double, double*> a(train_size, 2);
	// Matrix<double, double*> b(2, 4);
	// a.set_matrix((double*) &(x_train[0]));
	// b.set_matrix((double*) &(layer1.weights[0]));

	// std::cout << "Cuda Matrix dot(a,b)" << std::endl;
	// Matrix<double, double*> cuda_out = dot(a, b);
	// print(cuda_out);
	// std::cout << "Serial Matrix dot(a,b)" << std::endl;
	// Matrix<double> serial_out = dot(x_train, layer1.weights);
	// print(serial_out);

	// return 0
	// // END Test code -----------------------


	// Main algorithimic loop
    for (int i = 0; i < 2001; i++) {

		Matrix<double> out1 = layer1.forward(x_train);
		Matrix<double> out2 = layer2.forward(out1);
		Matrix<double> out3 = layer3.forward(out2);
		Matrix<double> out4 = layer4.forward(out3, y_train);
		double loss = layer4.get_loss();
		double acc = metric::accuracy(y_train, out4);

		Matrix<double> back4 = layer4.backward(out4, y_train);
		Matrix<double> back3 = layer3.backward(out2, back4);
		Matrix<double> back2 = layer2.backward(out1, back3);
		Matrix<double> back1 = layer1.backward(x_train, back2);
		
		
		sgd.pre_update();
		sgd.update(&layer3);
		sgd.update(&layer1);
		sgd.post_update();
		if (i % 100 == 0) {
			// Let's test the network every 100 iterations
			Matrix<double> outtest1 = layer1.forward(x_test);
			Matrix<double> outtest2 = layer2.forward(outtest1);
			Matrix<double> outtest3 = layer3.forward(outtest2);
			Matrix<double> outtest4 = layer4.forward(outtest3, y_test);
			double losstest = layer4.get_loss();
			double acctest = metric::accuracy(y_test, outtest4);

			std::cout << "epoch: " << i;
			std::cout << ", acc: " << acc;
			std::cout << ", loss: " << loss;
			std::cout << ", acc_test: " << acctest;
			std::cout << ", loss_test: " << losstest;
			std::cout << ", lr: " << std::fixed << sgd.get_lr() << std::endl;
		}

    }


    return 0;
}

/* handle_input()
* -----
* Selects which data should be used in training the neural network.
* Selected conditionally on the size of N.
*
* @x_train: The training data samples
* @y_train: The training data labels
* @x_test: The test data samples
* @y_test: The test data labels
* @N: The amount of data that should be inputted into the Matrix's
*/
void handle_input(Matrix<double>& x_train, Matrix<double>& y_train, 
		Matrix<double>& x_test, Matrix<double>& y_test, int N) {
	if (N <= 100) {
		x_train.set_matrix(x_train_raw_100);
		y_train.set_matrix(y_train_raw_100);
		x_test.set_matrix(x_test_raw_100);
		y_test.set_matrix(y_test_raw_100);
	} else if (N <= 200) {
		x_train.set_matrix(x_train_raw_200);
		y_train.set_matrix(y_train_raw_200);
		x_test.set_matrix(x_test_raw_200);
		y_test.set_matrix(y_test_raw_200);
	} else if (N <= 300) {
		x_train.set_matrix(x_train_raw_300);
		y_train.set_matrix(y_train_raw_300);
		x_test.set_matrix(x_test_raw_300);
		y_test.set_matrix(y_test_raw_300);
	} else if (N <= 400) {
		x_train.set_matrix(x_train_raw_400);
		y_train.set_matrix(y_train_raw_400);
		x_test.set_matrix(x_test_raw_400);
		y_test.set_matrix(y_test_raw_400);
	} else if (N <= 500) {
		x_train.set_matrix(x_train_raw_500);
		y_train.set_matrix(y_train_raw_500);
		x_test.set_matrix(x_test_raw_500);
		y_test.set_matrix(y_test_raw_500);
	} else if (N <= 600) {
		x_train.set_matrix(x_train_raw_600);
		y_train.set_matrix(y_train_raw_600);
		x_test.set_matrix(x_test_raw_600);
		y_test.set_matrix(y_test_raw_600);
	} else if (N <= 700) {
		x_train.set_matrix(x_train_raw_700);
		y_train.set_matrix(y_train_raw_700);
		x_test.set_matrix(x_test_raw_700);
		y_test.set_matrix(y_test_raw_700);
	} else if (N <= 800) {
		x_train.set_matrix(x_train_raw_800);
		y_train.set_matrix(y_train_raw_800);
		x_test.set_matrix(x_test_raw_800);
		y_test.set_matrix(y_test_raw_800);
	} else if (N <= 900) {
		x_train.set_matrix(x_train_raw_900);
		y_train.set_matrix(y_train_raw_900);
		x_test.set_matrix(x_test_raw_900);
		y_test.set_matrix(y_test_raw_900);
	} else if (N <= 1000) {
		x_train.set_matrix(x_train_raw_1000);
		y_train.set_matrix(y_train_raw_1000);
		x_test.set_matrix(x_test_raw_1000);
		y_test.set_matrix(y_test_raw_1000);
	}else if (N <= 1000) {
		x_train.set_matrix(x_train_raw_1000);
		y_train.set_matrix(y_train_raw_1000);
		x_test.set_matrix(x_test_raw_1000);
		y_test.set_matrix(y_test_raw_1000);
	} else if (N <= 2000) {
		x_train.set_matrix(x_train_raw_2000);
		y_train.set_matrix(y_train_raw_2000);
		x_test.set_matrix(x_test_raw_2000);
		y_test.set_matrix(y_test_raw_2000);
	} else if (N <= 5000) {
		x_train.set_matrix(x_train_raw_5000);
		y_train.set_matrix(y_train_raw_5000);
		x_test.set_matrix(x_test_raw_5000);
		y_test.set_matrix(y_test_raw_5000);
	} else if (N <= 10000) {
		x_train.set_matrix(x_train_raw_10000);
		y_train.set_matrix(y_train_raw_10000);
		x_test.set_matrix(x_test_raw_10000);
		y_test.set_matrix(y_test_raw_10000);
	} else if (N <= 20000) {
		x_train.set_matrix(x_train_raw_20000);
		y_train.set_matrix(y_train_raw_20000);
		x_test.set_matrix(x_test_raw_20000);
		y_test.set_matrix(y_test_raw_20000);
	}
}

int handle_N(int N) {
	if (N <= 100) {
		return 100;
	} else if (N <= 200)   {
		return 200;
	} else if (N <= 300)   {
		return 300;
	} else if (N <= 400)  {
		return 400;
	} else if (N <= 500) {
		return 500;
	} else if (N <= 600) {
		return 600;
	} else if (N <= 700) {
		return 700;
	} else if (N <= 800)  {
		return 800;
	} else if (N <= 900) {
		return 900;
	} else if (N <= 1000) {
		return 1000;
	} else if (N <= 2000) {
		return 2000;
	} else if (N <= 5000) {
		return 5000;
	} else if (N <= 10000) {
		return 10000;
	} else {
		return 20000;
	}
}
