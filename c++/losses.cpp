#include<iostream>
#include<cstdlib>

// Abstract Class
class Loss 
{
public: 
    // Abstract Method
    virtual int loss(int y_true, int y_pred) = 0;
    
    int calculateLoss(int y_true, int y_pred) {
	int mean = 0; 
    }
}


class SparseCategoricalCrossentropy: public Loss
{
public: 
    int loss(int y_true, int y_pred) {
	return y_true;
    }
}
