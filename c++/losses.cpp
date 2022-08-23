#include<iostream>
#include<cstdlib>
#include<matrix.h>

// Abstract Class
class Loss 
{
public: 
    // Abstract Method
    virtual int loss(int y_true, int y_pred) = 0;
    
    int calculateLoss(int y_true, int y_pred) {
	int mean = 0; 
    }
};


class CategoricalCrossentropy: public Loss
{
public: 
    int loss(int y_true, int y_pred) {
	// Expects y_true to be one hot encoded
	return matrix_mulconst(matrix_log(matrix_sum(matrix_mul(y_true, 
			    y_pred))), -1.0);
    }
};

class SparseCategoricalCrossentropy: public Loss
{
public: 
    int loss(int y_true, int y_pred) {
	// Expects y_true to be 1D array of integers. 
	// NOT IMPLEMENTED CORRECTLY YET.
	// NEEDS TO BE -np.log(y_pred[range(len(y_true)), y_true]):wq 
	return matrix_mulconst(matrix_log(y_true), -1.0)
    }
};
