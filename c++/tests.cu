#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <iostream>
#include <string>
#include <cassert>

#include "matrix.h"
#include "losses.h"
#include "layers.h"
#include "metrics.h"
#include "optimizers.h"
#include "cuda.h"
#include "data/data.h"


using namespace matrix;
// ----------------------------------- SETUP ------------------------------------------------------ //
int N = 100;
int train_size = N * 0.8;
double DBL_EPSILON = std::numeric_limits<double>::epsilon();

Matrix<double, double*> x_train_device(train_size, 2);
Matrix<double, double*> y_train_device(train_size, 2);
Matrix<double, double*> weights_device(2, 4);
Matrix<double, double*> biases_device(1, 4, 1);

Matrix<double> x_train(train_size, 2);
Matrix<double> y_train(train_size, 2);
Matrix<double> weights(2, 4);
Matrix<double> biases(1, 4, 1);

template <class T>
std::string type_name();

bool equals(Matrix<double, std::vector<double, std::allocator<double>>>& a, 
        Matrix<double, std::vector<double, std::allocator<double>>>& b);
bool equals(Matrix<double, std::vector<double, std::allocator<double>>>& a, 
        Matrix<double, double*>& b);
bool nearly_equal(double a, double b);
void randomize(Matrix<double>& a);
// ----------------------------------- ------------------------------------------------------ //

// -----------------------------------TESTS ------------------------------------------------------ //
void test_dot();

void test_add_case1();
void test_add_case2();
void test_add_case3();

void test_subtract_case1();
void test_subtract_case2();
void test_subtract_case3();

void test_mul_case1();
void test_mul_case2();
void test_mul_case3();

void test_divide_case1();
void test_divide_case2();
void test_divide_case3();

void test_sum_keepdims_0();
void test_sum_keepdims_1();
void test_sum_reduce();
void test_exp();
void test_log();
// ------------------------------------------------------------------------------------------------ //





/* main()
* -----
*/
int main(int argc, char *argv[]) {
	// This function allocates the data to these matrices
    x_train_device.set_matrix(&(x_train_raw_100[0]));
    y_train_device.set_matrix(&(y_train_raw_100[0]));
    x_train.set_matrix(x_train_raw_100);
    y_train.set_matrix(y_train_raw_100);
    randomize(weights);
    
    weights_device.set_matrix((double*) &(weights.get_matrix()[0]));


    // assert(equals(x_train, x_train_device));
    // assert(equals(y_train, y_train_device));
    // assert(equals(weights, weights_device));

    test_dot();
}

void test_dot() {
    Matrix<double> actual = dot(x_train, weights);
    Matrix<double, double*> attempt = dot(x_train_device, weights_device);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
}


bool equals(Matrix<double, std::vector<double, std::allocator<double>>>& a, 
        Matrix<double, std::vector<double, std::allocator<double>>>& b) {
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            if (!(std::fabs(a[i*a.rows + j] - b[i*a.rows + j]) < DBL_EPSILON)) {
                std::cout << "a != b - " << a[i*a.rows + j] << " != " << b[i*a.rows + j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool equals(Matrix<double, std::vector<double, std::allocator<double>>>& a, Matrix<double, double*>& b) {
    //Matrix<double> compare(a.rows, a.cols);
    //b.get_matrix(compare);
    double* compare = (double*) malloc(sizeof(double) * a.size());
	cuda::checkError(cudaMemcpy(compare, b.get_matrix(), sizeof(double) * a.size(), cudaMemcpyDeviceToHost));
    std::cout << a.size() << std::endl;
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            double a_i = a[i*a.rows + j];
            double b_i = compare[i*a.rows + j];
            if (!(std::fabs(a_i - b_i) < (double) 10000)) {
                std::cout << "i: " << i << " j: " << j << std::endl;
                std::cout << "Error: a != b : " << a_i << " != " << b_i << std::endl;
                //std::cout << "Error: a != b : " << a[i*a.rows + j] << " != " << compare[i*a.rows + j] << std::endl;
                //print(a);
                //print(compare);
                return false;
            }
        }
    }
    return true;
}

void randomize(Matrix<double>& a) {
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            a[i*a.rows + j] = i;
        }
    }
}

bool nearly_equal(double a, double b) {
  return std::nextafter(a, std::numeric_limits<double>::lowest()) <= b
    && std::nextafter(a, std::numeric_limits<double>::max()) >= b;
}


template <class T>
std::string type_name() {
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