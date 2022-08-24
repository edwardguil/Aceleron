#include "matrix.h"
#include "losses.h"
#include "layers.h"

using namespace matrix;

int main() {
    // Lets setup our data
    std::vector<std::vector<float>> in { 
	                       {1.0, 2.0, 3.0},
			       {-4.0, -5.0, -6.0},  
			       {7.0, 8.0, 9.0} };
    
    std::vector<std::vector<float>> true_in { {0.0, 1.0},
					   {1.0, 0.0},
					   {0.0, 1.0} };
    Matrix<float> X(3, 3);
    X.set_matrix(in);
    Matrix<float> y_true(3, 2);
    y_true.set_matrix(true_in);

    Dense layer1(3, 3);
    ReLU layer2;
    Dense layer3(3, 2);
    Softmax layer4;
    CategoricalCrossentropy loss;
    print(X);
    Matrix<float> out1 = layer1.forward(X);
    print(out1);
    Matrix<float> out2 = layer2.forward(out1);
    print(out2);
    Matrix<float> out3 = layer3.forward(out2);
    print(out3);
    Matrix<float> out4 = layer4.forward(out3);
    print(out4);
    print(loss.loss(y_true, out4));

    return 1;
}


