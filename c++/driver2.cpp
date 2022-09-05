#include "matrix.h"
#include "losses.h"
#include "layers.h"
#include "metrics.h"
#include "optimizers.h"
#include "data/data1000.h"

using namespace matrix;

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
	int test_size = N - train_size;
    Matrix<double> x_train(train_size, 2);
    x_train.set_matrix(std::vector<std::vector<double>>(x_train_raw.begin(), x_train_raw.end() - 800 + train_size));
    Matrix<double> y_train(train_size, 2);
    y_train.set_matrix(std::vector<std::vector<double>>(y_train_raw.begin(), y_train_raw.end() - 800 + train_size));

    Matrix<double> x_test(test_size, 2);
    x_test.set_matrix(std::vector<std::vector<double>>(x_test_raw.begin(), x_test_raw.end() - 200 + test_size));
    Matrix<double> y_test(test_size, 2);
    y_test.set_matrix(std::vector<std::vector<double>>(y_test_raw.begin(), y_test_raw.end() - 200 + test_size));

    Dense layer1(2, 32);
    ReLU layer2;
    Dense layer3(32, 2);
    SoftmaxCrossEntropy layer4;
    optimizer::SGD sgd(1.0, 0.001);

    for (int i = 0; i < 1801; i++) {

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
