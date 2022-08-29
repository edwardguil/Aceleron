#include "matrix.h"
#include "losses.h"
#include "layers.h"
#include "metrics.h"
#include "optimizers.h"
#include "data.h"

using namespace matrix;

int main() {
    Matrix<float> x_train(600, 2);
    x_train.set_matrix(x_train_raw);
    Matrix<float> y_train(600, 2);
    y_train.set_matrix(y_train_raw);


    Matrix<float> x_test(200, 2);
    x_test.set_matrix(x_train_raw);
    Matrix<float> y_test(200, 2);
    y_test.set_matrix(y_train_raw);

    Dense layer1(2, 32);
    ReLU layer2;
    Dense layer3(32, 2);
    SoftmaxCrossEntropy layer4;
    optimizer::SGD sgd(1, 0.001);

    //for (int i = 0; i < 10001; i++) {

	Matrix<float> out1 = layer1.forward(x_train);
	Matrix<float> out2 = layer2.forward(out1);
	Matrix<float> out3 = layer3.forward(out2);
	Matrix<float> out4 = layer4.forward(out3, y_train);
	float loss = layer4.get_loss();
	float acc = metric::accuracy(y_train, out4);

	Matrix<float> back4 = layer4.backward(out4, y_train);
	Matrix<float> back3 = layer3.backward(out2, back4);
	Matrix<float> back2 = layer2.backward(out1, back3);
	Matrix<float> back1 = layer1.backward(x_train, back2);
	
	sgd.pre_update();
	sgd.update(&layer3);
	sgd.update(&layer1);
	sgd.post_update();

	/*
	if (i % 100 == 0) {
	    Matrix<float> outtest1 = layer1.forward(x_test);
	    Matrix<float> outtest2 = layer2.forward(outtest1);
	    Matrix<float> outtest3 = layer3.forward(outtest2);
	    Matrix<float> outtest4 = layer4.forward(outtest3, y_test);
	    float losstest = layer4.get_loss();
	    float acctest = metric::accuracy(y_train, outtest4);

	    std::cout << "epoch: " << i;
	    std::cout << ", acc: " << acc;
	    std::cout << ", loss: " << loss;
	    std::cout << ", acc_test: " << acctest;
	    std::cout << ", loss_test: " << losstest << std::endl;
	}

    }
    */


    return 1;
}
