#include "matrix.h"
#include "losses.h"
#include "layers.h"
#include "metrics.h"
#include "optimizers.h"
#include "data/data1k.h"

using namespace matrix;

int main() {
    
    Matrix<double> x_train(800, 2);
    x_train.set_matrix(x_train_raw);
    Matrix<double> y_train(800, 2);
    y_train.set_matrix(y_train_raw);


    Matrix<double> x_test(200, 2);
    x_test.set_matrix(x_train_raw);
    Matrix<double> y_test(200, 2);
    y_test.set_matrix(y_train_raw);

    Dense layer1(2, 32);
    ReLU layer2;
    Dense layer3(32, 2);
    SoftmaxCrossEntropy layer4;
    optimizer::SGD sgd(1, 0.001);

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
	    std::cout << ", lr: " << sgd.get_lr() << std::endl;
	}

    }


    return 1;
}
