#include "matrix.h"
#include "losses.h"
#include "layers.h"
#include "metrics.h"

using namespace matrix;

int main() {
    // Lets setup our data
    std::vector<std::vector<float>> in { 
	                       {1.0, 2.0, 3.0},
			       {-4.0, -5.0, -6.0},  
			       {7.0, 8.0, 9.0} };
    
    std::vector<std::vector<float>> true_in {{0, 1},
					   {1, 0},
					   {0, 1}};
    Matrix<float> X(3, 3);
    X.set_matrix(in);
    Matrix<float> y_true(3, 2);
    y_true.set_matrix(true_in);

    Dense layer1(3, 3);
    ReLU layer2;
    Dense layer3(3, 2);
    SoftmaxCrossEntropy layer4;
    print(X);
    Matrix<float> out1 = layer1.forward(X);
    print(out1);
    Matrix<float> out2 = layer2.forward(out1);
    print(out2);
    Matrix<float> out3 = layer3.forward(out2);
    print(out3);
    std::cout << "Start of softmax \n";
    Matrix<float> out4 = layer4.forward(out3, y_true);
    print(out4);
    std::cout << layer4.get_loss() << std::endl;
    std::cout << metric::accuracy(y_true, out4) << "\n";

    std::cout << "Start of backprop relu and dense" << std::endl;

    Matrix<float> back1 = layer2.backward(out2);
    print(back1);
    Matrix<float> back2 = layer1.backward(back1);
    print(back2);
    print(layer1.get_dbiases());
    print(layer1.get_dweights());

    std::cout << "START OF TEST" << std::endl;

    std::vector<std::vector<float>> test_in {{0.7, 0.1, 0.2},
					   {0.1, 0.5, 0.4},
					   {0.02, 0.9, 0.08}};
    Matrix<float> softmaxOut(3, 3);
    softmaxOut.set_matrix(test_in);
    Matrix<float> test5 = layer4.backward(softmaxOut, y_true);
    print(test5);
    std::cout << "Start of backprop full" << std::endl;


    return 1;
}


