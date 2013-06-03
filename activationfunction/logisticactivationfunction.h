#ifndef __LOGISTICACTIVATIONFUNCTION_H__
#define __LOGISTICACTIVATIONFUNCTION_H__

#include <cmath>
#include "activationfunction.h"

/**
 * Derived class from ActivationFunction that implements a linear function.
 */
template <class Type>
class LogisticFunction : public ActivationFunction<Type>
{
	
public:
	
	/**
	 * Method that implements the linear function between two arrays.
	 * The formula is given by:
	 * \f$ linear(weights, input, arrayLength) = 1 \f$
	 * @param weights First array of weights
	 * @param input Second array of weights
	 * @param arrayLength Length of the arrays
	 * @param threshold 
	 */
	
	void exec(Type value, Type beta = 1)
	{
	
        
		if (beta != 0)
		{
			if(value > 100) this->result = 1;
        	else if(value < -100) this->result = 0;
			else this->result = (Type)(1/(1 + pow(M_E, -value*beta)));
		}	
		
		
	}
	
	void derived(Type value, Type beta = 1)
	{
        if (beta != 0)
		{
			if(value > 100) this->derivedResult = 1;
        	else if(value < -100) this->derivedResult = 0;
			else this->derivedResult = (Type)(pow(M_E, -value*beta)/pow(1 + pow(M_E, -value*beta), 2));
		}	
	}
	
};

#endif
