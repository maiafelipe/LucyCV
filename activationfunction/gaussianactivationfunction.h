#ifndef __GAUSSIANACTIVATIONFUNCTION_H__
#define __GAUSSIANACTIVATIONFUNCTION_H__

#include <cmath>
#include "activationfunction.h"

/**
 * Derived class from ActivationFunction that implements a gaussian function.
 */
template <class Type>
class GaussianActivationFunction : public ActivationFunction<Type>
{
	
public:
	
	/**
	 * Method that implements the gaussian function between two arrays.
	 * The formula is given by:
	 * \f$ gaussian(weights, input, arrayLength) = 1 \f$
	 * @param weights First array of weights
	 * @param input Second array of weights
	 * @param arrayLength Length of the arrays
	 * @param sigma Sigma value for the gaussian function
	 */
	void exec(Type distance, Type sigma)
	{
		sigma *= 2;
		this->result = exp(-1*distance/sigma);
		if (this->result < 0.000001)
			this->result = 0;
	}
	
};

#endif
