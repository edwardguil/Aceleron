/*
* Author: Edward Guilfoyle
* Note: This file is to test the PARALLEL implementation against SERIAL implementation.
*/
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <memory>
#include <cmath>
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
double N = 100;
double DBL_EPSILON = std::numeric_limits<double>::epsilon();
template <class T>
std::string type_name();
// ----------------------------------- Helper Functions--------------------------------------------- //
bool equals(Matrix<double, std::vector<double, std::allocator<double>>>& a, 
        Matrix<double, double*>& b);
bool equals(Matrix<int, std::vector<int, std::allocator<int>>>& a, 
        Matrix<int, int*>& b);


bool nearly_equal(double a, double b);
void randomize(Matrix<double>& a);
double round_up(double value, int decimal_places);
// ----------------------------------- ------------------------------------------------------ //

// -----------------------------------TESTS ------------------------------------------------------ //
void test_dot();

void test_max();

void test_sum_keepdims_0();
void test_sum_keepdims_1();
void test_sum_reduce();

void test_transpose();

void test_add_case1();
void test_add_case2();
void test_add_case3();
void test_add_case4();

void test_subtract_case1();
void test_subtract_case2();
void test_subtract_case3();
void test_subtract_case4();

void test_division_case1();
void test_division_case2();
void test_division_case3();
void test_division_case4();

void test_mul_case1();
void test_mul_case2();
void test_mul_case3();
void test_mul_case4();

void test_equals_case1();
void test_equals_case2();
void test_equals_case3();
void test_equals_case4();

void test_exp();
void test_log();
void test_mul_const();
void test_argmax();

// ------------------------------------------------------------------------------------------------ //


/* main()
* -----
*/
int main(int argc, char *argv[]) {
    test_dot();

    test_max();
    test_sum_keepdims_0();
    test_sum_keepdims_1();
    test_sum_reduce();
    test_transpose();
        
    test_add_case1();
    test_add_case2();
    test_add_case3();
    test_add_case4();

    test_subtract_case1();
    test_subtract_case2();
    test_subtract_case3();
    test_subtract_case4();

    test_division_case1();
    test_division_case2();
    test_division_case3();
    test_division_case4();

    test_mul_case1();
    test_mul_case2();
    test_mul_case3();
    test_mul_case4();

    test_equals_case1();
    test_equals_case2();
    test_equals_case3();
    test_equals_case4();

    test_exp();
    test_log();
    test_mul_const();
    test_argmax();
    std::cout << "Well Done! All tests have passed.\n" <<   R"V0G0N(
          __
     w  c(..)o   (
      \__(-)    __)
          /\   (
         /(_)___)
         w /|
          | \
          m  m


    )V0G0N" << "Go For A Boogey!" << std::endl;

}

void test_dot() {
    Matrix<double> a(10, 4, 1);
    Matrix<double> b(4, 10, 2);
    Matrix<double, double*> a_d(10, 4, 1, true);
    Matrix<double, double*> b_d(4, 10, 2, true);

    Matrix<double> actual = dot(a, b);
    Matrix<double, double*> attempt = dot(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_max() {
    Matrix<double> a(4, 4);
    randomize(a);
    Matrix<double, double*> a_d(4, 4);
    a_d.set_matrix(&a.get_matrix()[0]);

    Matrix<double> actual = max(a);
    Matrix<double, double*> attempt = max(a_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}


void test_sum_keepdims_0() {
    Matrix<double> a(4, 4);
    randomize(a);
    Matrix<double, double*> a_d(4, 4);
    a_d.set_matrix(&a.get_matrix()[0]);

    Matrix<double> actual = sum(a, 0, true);
    Matrix<double, double*> attempt = sum(a_d, 0, true);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}


void test_sum_keepdims_1() {
    Matrix<double> a(4, 4);
    randomize(a);
    Matrix<double, double*> a_d(4, 4);
    a_d.set_matrix(&a.get_matrix()[0]);

    Matrix<double> actual = sum(a, 1, true);
    Matrix<double, double*> attempt = sum(a_d, 1, true);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_sum_reduce() {
    Matrix<double> a(4, 4);
    randomize(a);
    Matrix<double, double*> a_d(4, 4);
    a_d.set_matrix(&a.get_matrix()[0]);

    Matrix<double> actual = sum(a, 1, false);
    Matrix<double, double*> attempt = sum(a_d, 1, false);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_transpose() {
    Matrix<double> a(1, 4);
    randomize(a);
    Matrix<double, double*> a_d(1, 4);
    a_d.set_matrix(&a.get_matrix()[0]);

    Matrix<double> actual = transpose(a);
    Matrix<double, double*> attempt = transpose(a_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

/* test_add_case1()
* -----
* Tests the CUDA add against the serial implementation. Tests 
* case 1 of the add function, that is when a.rows = b.rows 
* && a.cols == b.cols e.g. element wise add. Exits if test 
* fails, returns void otherwise. 
*/
void test_add_case1() {
    // Setup matrices
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(4, 4, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(4, 4, 2, true);

    // Perform calculation
    Matrix<double> actual = add(a, b);
    Matrix<double, double*> attempt = add(a_d, b_d);

    // Compare
    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}


void test_add_case2() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(1, 4, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(1, 4, 2, true);

    Matrix<double> actual = add(a, b);
    Matrix<double, double*> attempt = add(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_add_case3() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(4, 1, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(4, 1, 2, true);

    Matrix<double> actual = add(a, b);
    Matrix<double, double*> attempt = add(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_add_case4() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(1, 1, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(1, 1, 2, true);

    Matrix<double> actual = add(a, b);
    Matrix<double, double*> attempt = add(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_subtract_case1() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(4, 4, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(4, 4, 2, true);

    Matrix<double> actual = subtract(a, b);
    Matrix<double, double*> attempt = subtract(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}


void test_subtract_case2() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(1, 4, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(1, 4, 2, true);

    Matrix<double> actual = subtract(a, b);
    Matrix<double, double*> attempt = subtract(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_subtract_case3() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(4, 1, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(4, 1, 2, true);

    Matrix<double> actual = subtract(a, b);
    Matrix<double, double*> attempt = subtract(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_subtract_case4() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(1, 1, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(1, 1, 2, true);

    Matrix<double> actual = subtract(a, b);
    Matrix<double, double*> attempt = subtract(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_division_case1() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(4, 4, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(4, 4, 2, true);

    Matrix<double> actual = division(a, b);
    Matrix<double, double*> attempt = division(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}


void test_division_case2() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(1, 4, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(1, 4, 2, true);

    Matrix<double> actual = division(a, b);
    Matrix<double, double*> attempt = division(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_division_case3() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(4, 1, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(4, 1, 2, true);

    Matrix<double> actual = division(a, b);
    Matrix<double, double*> attempt = division(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_division_case4() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(1, 1, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(1, 1, 2, true);

    Matrix<double> actual = division(a, b);
    Matrix<double, double*> attempt = division(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_mul_case1() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(4, 4, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(4, 4, 2, true);

    Matrix<double> actual = mul(a, b);
    Matrix<double, double*> attempt = mul(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}


void test_mul_case2() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(1, 4, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(1, 4, 2, true);

    Matrix<double> actual = mul(a, b);
    Matrix<double, double*> attempt = mul(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_mul_case3() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(4, 1, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(4, 1, 2, true);

    Matrix<double> actual = mul(a, b);
    Matrix<double, double*> attempt = mul(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_mul_case4() {
    Matrix<double> a(4, 4, 1);
    Matrix<double> b(1, 1, 2);
    Matrix<double, double*> a_d(4, 4, 1, true);
    Matrix<double, double*> b_d(1, 1, 2, true);

    Matrix<double> actual = mul(a, b);
    Matrix<double, double*> attempt = mul(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_equals_case1() {
    Matrix<int> a(4, 4, 1);
    Matrix<int> b(4, 4, 2);
    Matrix<int, int*> a_d(4, 4, 1, true);
    Matrix<int, int*> b_d(4, 4, 2, true);

    Matrix<int> actual = equals(a, b);
    Matrix<int, int*> attempt = equals(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}


void test_equals_case2() {
    Matrix<int> a(4, 4, 1);
    Matrix<int> b(1, 4, 2);
    Matrix<int, int*> a_d(4, 4, 1, true);
    Matrix<int, int*> b_d(1, 4, 2, true);

    Matrix<int> actual = equals(a, b);
    Matrix<int, int*> attempt = equals(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_equals_case3() {
    Matrix<int> a(4, 4, 1);
    Matrix<int> b(4, 1, 2);
    Matrix<int, int*> a_d(4, 4, 1, true);
    Matrix<int, int*> b_d(4, 1, 2, true);

    Matrix<int> actual = equals(a, b);
    Matrix<int, int*> attempt = equals(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_equals_case4() {
    Matrix<int> a(4, 4, 1);
    Matrix<int> b(1, 1, 2);
    Matrix<int, int*> a_d(4, 4, 1, true);
    Matrix<int, int*> b_d(1, 1, 2, true);

    Matrix<int> actual = equals(a, b);
    Matrix<int, int*> attempt = equals(a_d, b_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_exp() {
    Matrix<double> a(10, 4, 1);
    Matrix<double, double*> a_d(10, 4, 1, true);

    Matrix<double> actual = exp(a);
    Matrix<double, double*> attempt = exp(a_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}

void test_log() {
    Matrix<double> a(10, 4, 1);
    Matrix<double, double*> a_d(10, 4, 1, true);

    Matrix<double> actual = log(a);
    Matrix<double, double*> attempt = log(a_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}



void test_mul_const() {
    Matrix<double> a(10, 4, 1);
    Matrix<double, double*> a_d(10, 4, 1, true);

    Matrix<double> actual = mul_const(a, 3.5);
    Matrix<double, double*> attempt = mul_const(a_d, 3.5);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}


void test_argmax() {
    Matrix<double> a(10, 4, 1);
    Matrix<double, double*> a_d(10, 4, 1, true);

    Matrix<int> actual = argmax(a);
    Matrix<int, int*> attempt = argmax(a_d);

    assert(actual.rows == attempt.rows && actual.cols == attempt.cols);
    assert(equals(actual, attempt));
    _free();
}



bool equals(Matrix<double, std::vector<double, std::allocator<double>>>& a, Matrix<double, double*>& b) {
    std::vector<double> compare(a.rows * a.cols);
	cuda::checkError(cudaMemcpy(&compare[0], b.get_matrix(), sizeof(double) * a.size(), cudaMemcpyDeviceToHost));
    for (double i = 0; i < a.rows; ++i) {
        for (double j = 0; j < a.cols; ++j) {
            if ((std::fabs(a[i*a.cols + j] - compare[i*a.cols + j])) > DBL_EPSILON) {
                std::cout << "i: " << i << " j: " << j << std::endl;
                std::cout << a[i*a.cols + j] << " != " << compare[i*a.cols + j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool equals(Matrix<int, std::vector<int, std::allocator<int>>>& a, Matrix<int, int*>& b) {
    std::vector<int> compare(a.rows * a.cols);
	cuda::checkError(cudaMemcpy(&compare[0], b.get_matrix(), sizeof(int) * a.size(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            if (a[i*a.cols + j] != compare[i*a.cols + j]) {
                std::cout << "i: " << i << " j: " << j << std::endl;
                std::cout << a[i*a.cols + j] << " != " << compare[i*a.cols + j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

void randomize(Matrix<double>& a) {
    for (double i = 0; i < a.rows; ++i) {
        for (double j = 0; j < a.cols; ++j) {
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

double round_up(double value, int decimal_places) {
    const double multiplier = std::pow(10.0, decimal_places);
    return std::ceil(value * multiplier) / multiplier;
}
