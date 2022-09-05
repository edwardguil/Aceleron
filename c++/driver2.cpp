#include "matrix.h"
#include "losses.h"
#include "layers.h"
#include "metrics.h"
#include "optimizers.h"
#include "data/data.h"

using namespace matrix;

void handle_input(Matrix<double>& x_train, Matrix<double>& y_train, 
		Matrix<double>& x_test, Matrix<double>& y_test, int N);

int main(int argc, char *argv[]) {
	int N = 100;
	if (argc > 1) {
		N = std::stoi(argv[1]);
		if (N > 1000) {
			N = 1000;
		}
	}

	std::cout << "N: " << N << std::endl;
	int train_size = N * 0.8;

    Matrix<double> x_train(train_size, 2);
    Matrix<double> y_train(train_size, 2);
    Matrix<double> x_test(N - train_size, 2);
    Matrix<double> y_test(N - train_size, 2);
	handle_input(x_train, y_train, x_test, y_test, N);

    Dense layer1(2, 16);
    ReLU layer2;
    Dense layer3(16, 2);
    SoftmaxCrossEntropy layer4;
    optimizer::SGD sgd(1.0, 0.001);

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


    return 1;
}

void handle_input(Matrix<double>& x_train, Matrix<double>& y_train, 
		Matrix<double>& x_test, Matrix<double>& y_test, int N) {
	if (N == 100) {
		x_train.set_matrix(x_train_raw_100);
		y_train.set_matrix(y_train_raw_100);
		x_test.set_matrix(x_test_raw_100);
		y_test.set_matrix(y_test_raw_100);
	} else if (N == 200) {
		x_train.set_matrix(x_train_raw_200);
		y_train.set_matrix(y_train_raw_200);
		x_test.set_matrix(x_test_raw_200);
		y_test.set_matrix(y_test_raw_200);
	} else if (N == 300) {
		x_train.set_matrix(x_train_raw_300);
		y_train.set_matrix(y_train_raw_300);
		x_test.set_matrix(x_test_raw_300);
		y_test.set_matrix(y_test_raw_300);
	} else if (N == 400) {
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
	} else if (N == 800) {
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
