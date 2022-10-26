#include "matrix.h"
#include "losses.h"
#include "layers.h"
#include "metrics.h"
#include "optimizers.h"
#include "cuda.h"
#include "data/data.h"

#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>

template <class T>
std::string type_name();

using namespace matrix;

void handle_input(Matrix<double, double*>& x_train, Matrix<double, double*>& y_train, 
		Matrix<double, double*>& x_test, Matrix<double, double*>& y_test, int N);

void handle_input(Matrix<double>& x_train, Matrix<double>& y_train, 
		Matrix<double>& x_test, Matrix<double>& y_test, int N);

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
	int N = 100;
	if (argc > 1) {
		N = std::stoi(argv[1]);
		if (N > 1000 || N < 100) {
			N = 1000;
		}
	}


	//std::cout << "N: " << N << std::endl;
	int train_size = N * 0.80;

	// Define the training and testing Matrixes
    Matrix<double, double*> x_train(train_size, 2);
    Matrix<double, double*> y_train(train_size, 2);
    Matrix<double, double*> x_test(N - train_size, 2);
    Matrix<double, double*> y_test(N - train_size, 2);
	// This function allocates the data to these matrices
	handle_input(x_train, y_train, x_test, y_test, N);

    Matrix<double> x_train_s(train_size, 2);
    Matrix<double> y_train_s(train_size, 2);
    Matrix<double> x_test_s(N - train_size, 2);
    Matrix<double> y_test_s(N - train_size, 2);
	handle_input(x_train_s, y_train_s, x_test_s, y_test_s, N);

	// Construct our network
    Dense<double, double*> layer1(2, 16, false);
    ReLU<double, double*> layer2;
    Dense<double, double*> layer3(16, 2, false);
    Softmax<double, double*> layer4;

	// Serial
	Dense<double> layer1_s(16, 3, false);
    ReLU<double> layer2_s;
    Dense<double> layer3_s(16, 2, false);
    Softmax<double> layer4_s;
	
	// --------------- Tests -------------------------
	// std::cout << "CUDA IMPLEMENTATION" << std::endl;
	// print(dot(x_train, layer1.weights));
	// print(layer1.biases);
	// print(add(dot(x_train, layer1.weights), layer1.biases));

	// // print(layer1.weights);
	// // print(x_train);
	// // print(dot(x_train, layer1.weights));
	// std::cout << "SERIAL IMPLEMENTATION" << std::endl;
	// print(dot(x_train_s, layer1_s.weights));
	// print(layer1_s.biases);
	// print(add(dot(x_train_s, layer1_s.weights), layer1_s.biases));

	// --------------- Tests -------------------------
	// return 0;

	Matrix<double, double*> out1 = layer1.forward(x_train);
	Matrix<double, double*> out2 = layer2.forward(out1);
	Matrix<double, double*> out3 = layer3.forward(out2); 
	Matrix<double, double*> out4 = layer4.forward(out3); 
	std::cout << "CUDA IMPLEMENTATION" << std::endl;
	print(out4);

	Matrix<double> out1_s = layer1_s.forward(x_train_s);
	Matrix<double> out2_s = layer2_s.forward(out1_s);
	Matrix<double> out3_s = layer3_s.forward(out2_s); 
	Matrix<double> out4_s = layer4_s.forward(out3_s); 
	std::cout << "SERIAL IMPLEMENTATION" << std::endl;
	print(out4);
	return 0;

	//print(max(out3));
	//print(out4);
	// std::cout << "decltype(i) is " << type_name<decltype(layer2)>() << '\n';
	// std::cout << "decltype(i) is " << type_name<decltype(layer3)>() << '\n';
	// std::cout << "decltype(i) is " << type_name<decltype(layer4)>() << '\n';
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
    // for (int i = 0; i < 2001; i++) {

	// 	Matrix<double> out1 = layer1.forward(x_train);
	// 	Matrix<double> out2 = layer2.forward(out1);
	// 	Matrix<double> out3 = layer3.forward(out2);
	// 	Matrix<double> out4 = layer4.forward(out3, y_train);
	// 	double loss = layer4.get_loss();
	// 	double acc = metric::accuracy(y_train, out4);

	// 	Matrix<double> back4 = layer4.backward(out4, y_train);
	// 	Matrix<double> back3 = layer3.backward(out2, back4);
	// 	Matrix<double> back2 = layer2.backward(out1, back3);
	// 	Matrix<double> back1 = layer1.backward(x_train, back2);
		
		
	// 	sgd.pre_update();
	// 	sgd.update(&layer3);
	// 	sgd.update(&layer1);
	// 	sgd.post_update();
	// 	if (i % 100 == 0) {
	// 		// Let's test the network every 100 iterations
	// 		Matrix<double> outtest1 = layer1.forward(x_test);
	// 		Matrix<double> outtest2 = layer2.forward(outtest1);
	// 		Matrix<double> outtest3 = layer3.forward(outtest2);
	// 		Matrix<double> outtest4 = layer4.forward(outtest3, y_test);
	// 		double losstest = layer4.get_loss();
	// 		double acctest = metric::accuracy(y_test, outtest4);

	// 		std::cout << "epoch: " << i;
	// 		std::cout << ", acc: " << acc;
	// 		std::cout << ", loss: " << loss;
	// 		std::cout << ", acc_test: " << acctest;
	// 		std::cout << ", loss_test: " << losstest;
	// 		std::cout << ", lr: " << std::fixed << sgd.get_lr() << std::endl;
	// 	}

    // }

    // return 0;
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
void handle_input(Matrix<double, double*>& x_train, Matrix<double, double*>& y_train, 
		Matrix<double, double*>& x_test, Matrix<double, double*>& y_test, int N) {
	if (N == 100) {
		x_train.set_matrix(&(x_train_raw_100[0]));
		y_train.set_matrix(&(y_train_raw_100[0]));
		x_test.set_matrix(&(x_test_raw_100[0]));
		y_test.set_matrix(&(y_test_raw_100[0]));
	} else if (N == 200)   {
		x_train.set_matrix(&(x_train_raw_200[0]));
		y_train.set_matrix(&(y_train_raw_200[0]));
		x_test.set_matrix(&(x_test_raw_200[0]));
		y_test.set_matrix(&(y_test_raw_200[0]));
	} else if (N == 300)   {
		x_train.set_matrix(&(x_train_raw_300[0]));
		y_train.set_matrix(&(y_train_raw_300[0]));
		x_test.set_matrix(&(x_test_raw_300[0]));
		y_test.set_matrix(&(y_test_raw_300[0]));
	} else if (N == 400)  {
		x_train.set_matrix(&(x_train_raw_400[0]));
		y_train.set_matrix(&(y_train_raw_400[0]));
		x_test.set_matrix(&(x_test_raw_400[0]));
		y_test.set_matrix(&(y_test_raw_400[0]));
	} else if (N == 500) {
		x_train.set_matrix(&(x_train_raw_500[0]));
		y_train.set_matrix(&(y_train_raw_500[0]));
		x_test.set_matrix(&(x_test_raw_500[0]));
		y_test.set_matrix(&(y_test_raw_500[0]));
	} else if (N == 600) {
		x_train.set_matrix(&(x_train_raw_600[0]));
		y_train.set_matrix(&(y_train_raw_600[0]));
		x_test.set_matrix(&(x_test_raw_600[0]));
		y_test.set_matrix(&(y_test_raw_600[0]));
	} else if (N == 700) {
		x_train.set_matrix(&(x_train_raw_700[0]));
		y_train.set_matrix(&(y_train_raw_700[0]));
		x_test.set_matrix(&(x_test_raw_700[0]));
		y_test.set_matrix(&(y_test_raw_700[0]));
	} else if (N == 800)  {
		x_train.set_matrix(&(x_train_raw_800[0]));
		y_train.set_matrix(&(y_train_raw_800[0]));
		x_test.set_matrix(&(x_test_raw_800[0]));
		y_test.set_matrix(&(y_test_raw_800[0]));
	} else if (N == 900) {
		x_train.set_matrix(&(x_train_raw_900[0]));
		y_train.set_matrix(&(y_train_raw_900[0]));
		x_test.set_matrix(&(x_test_raw_900[0]));
		y_test.set_matrix(&(y_test_raw_900[0]));
	} else if (N == 1000) {
		x_train.set_matrix(&(x_train_raw_1000[0]));
		y_train.set_matrix(&(y_train_raw_1000[0]));
		x_test.set_matrix(&(x_test_raw_1000[0]));
		y_test.set_matrix(&(y_test_raw_1000[0]));
	}
}


void handle_input(Matrix<double>& x_train, Matrix<double>& y_train, 
		Matrix<double>& x_test, Matrix<double>& y_test, int N) {
	if (N == 100) {
		x_train.set_matrix(x_train_raw_100);
		y_train.set_matrix(y_train_raw_100);
		x_test.set_matrix(x_test_raw_100);
		y_test.set_matrix(y_test_raw_100);
	} else if (N == 200)   {
		x_train.set_matrix(x_train_raw_200);
		y_train.set_matrix(y_train_raw_200);
		x_test.set_matrix(x_test_raw_200);
		y_test.set_matrix(y_test_raw_200);
	} else if (N == 300)   {
		x_train.set_matrix(x_train_raw_300);
		y_train.set_matrix(y_train_raw_300);
		x_test.set_matrix(x_test_raw_300);
		y_test.set_matrix(y_test_raw_300);
	} else if (N == 400)  {
		x_train.set_matrix(x_train_raw_400);
		y_train.set_matrix(y_train_raw_400);
		x_test.set_matrix(x_test_raw_400);
		y_test.set_matrix(y_test_raw_400);
	} else if (N == 500) {
		x_train.set_matrix(x_train_raw_500);
		y_train.set_matrix(y_train_raw_500);
		x_test.set_matrix(x_test_raw_500);
		y_test.set_matrix(y_test_raw_500);
	} else if (N == 600) {
		x_train.set_matrix(x_train_raw_600);
		y_train.set_matrix(y_train_raw_600);
		x_test.set_matrix(x_test_raw_600);
		y_test.set_matrix(y_test_raw_600);
	} else if (N == 700) {
		x_train.set_matrix(x_train_raw_700);
		y_train.set_matrix(y_train_raw_700);
		x_test.set_matrix(x_test_raw_700);
		y_test.set_matrix(y_test_raw_700);
	} else if (N == 800)  {
		x_train.set_matrix(x_train_raw_800);
		y_train.set_matrix(y_train_raw_800);
		x_test.set_matrix(x_test_raw_800);
		y_test.set_matrix(y_test_raw_800);
	} else if (N == 900) {
		x_train.set_matrix(x_train_raw_900);
		y_train.set_matrix(y_train_raw_900);
		x_test.set_matrix(x_test_raw_900);
		y_test.set_matrix(y_test_raw_900);
	} else if (N == 1000) {
		x_train.set_matrix(x_train_raw_1000);
		y_train.set_matrix(y_train_raw_1000);
		x_test.set_matrix(x_test_raw_1000);
		y_test.set_matrix(y_test_raw_1000);
	}
}

template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
#ifndef _MSC_VER
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
#else
                nullptr,
#endif
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}