#include "matrix.h"
#include "losses.h"
#include "layers.h"
#include "metrics.h"
#include "optimizers.h"

using namespace matrix;

int main() {
    // Lets setup our data
    std::vector<double> in { 
	                       1.0, 2.0, 3.0,
			       -4.0, -5.0, -6.0,  
			       7.0, 8.0, 9.0};
    
    std::vector<double> true_in {0, 1,
					   1, 0,
					   0, 1};
    Matrix<double> X(3, 3);
    X.set_matrix(in);
    Matrix<double> y_true(3, 2);
    y_true.set_matrix(true_in);

    Dense layer1(3, 2);
    ReLU layer2;
    Dense layer3(2, 2);
    SoftmaxCrossEntropy layer4;
    print(X);

    Matrix<double> out1 = layer1.forward(X);
    print(out1);
    Matrix<double> out2 = layer2.forward(out1);
    print(out2);
    Matrix<double> out3 = layer3.forward(out2);
    print(out3);
    std::cout << "Start of softmax \n";
    Matrix<double> out4 = layer4.forward(out3, y_true);
    print(out4);
    std::cout << layer4.get_loss() << std::endl;
    std::cout << metric::accuracy(y_true, out4) << "\n";

    std::cout << "Start of backprop relu and dense" << std::endl;

    Matrix<double> test1 = layer2.backward(out1, out2);
    print(test1);
    Matrix<double> test2 = layer1.backward(X, test1);
    print(test2);
    print(layer1.get_dbiases());
    print(layer1.get_dweights());

    std::cout << "START OF TEST" << std::endl;

    std::vector<double> test_in {0.7, 0.1, 0.2,
					   0.1, 0.5, 0.4,
					   0.02, 0.9, 0.08};
    Matrix<double> softmaxOut(3, 3);
    softmaxOut.set_matrix(test_in);
    Matrix<double> test5 = layer4.backward(softmaxOut, y_true);
    print(test5);
    
    std::cout << "Start of backprop full" << std::endl;
    Matrix<double> back4 = layer4.backward(out4, y_true);
    print(back4);
    Matrix<double> back3 = layer3.backward(out2, back4);
    print(back3);
    Matrix<double> back2 = layer2.backward(out1, back3);
    print(back2);
    Matrix<double> back1 = layer1.backward(X, back2);
    print(back1);
    
    std::cout << "Start Of optimizer test" << std::endl;
    std::cout << "      Weights pre: " << std::endl;
    print(layer3.get_weights());
    print(layer3.get_dweights());
    print(layer1.get_weights());
    print(layer1.get_dweights());
    std::cout << "      Biases pre: " << std::endl;
    print(layer3.get_biases());
    print(layer3.get_dbiases());
    print(layer1.get_biases());
    print(layer1.get_dbiases());
    optimizer::SGD sgd(1, 0);
    sgd.pre_update();
    sgd.update(&layer3);
    sgd.update(&layer1);
    sgd.post_update();
    std::cout << "      Weights post: " << std::endl;
    print(layer3.get_weights());
    print(layer1.get_weights());
    std::cout << "      Biases post: " << std::endl;
    print(layer3.get_biases());
    print(layer1.get_biases());
    return 1;
}


