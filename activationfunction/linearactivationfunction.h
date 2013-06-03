#ifndef __LINEARACTIVATIONFUNCTION_H__
#define __LINEARNACTIVATIONFUNCTION_H__

#include <cmath>
#include "activationfunction.h"

/**
 * Derived class from ActivationFunction that implements a linear function.
 */
template <class Type>
class LinearActivationFunction : public ActivationFunction<Type>
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
	void exec(Type value, Type threshold)
	{
		//cout << "value: " << value << " - threshold: " << threshold << endl;
		if (value > threshold)
			this->result = value;
		else
		{
			this->result = 0; //?
		}
		/*
		 * this->result = value; direto?
		 */
		
	}
	
	void derived(Type value, Type threshold)
	{
		this->derivedResult = 1;
	}
	
};

#endif
