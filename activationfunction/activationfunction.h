#ifndef __ACTIVATIONFUNCTION_H__
#define __ACTIVATIONFUNCTION_H__

#include <iostream>

using namespace std;


/**
 * A base distance class used to evaluate the winner neuron in the Kohonen Self-organizing Map.
 */

template <class Type>
class ActivationFunction
{
public:
	/**
	 * The constructor sets the result distance to 0.
	 */
	ActivationFunction()
	{
		result = 0;
		derivedResult = 0;
	}
	
	/**
	 * This method will be derived to calculate the distance between two arrays of the same length.
	 * @param array1 First array of weights
	 * @param array2 Second array of weights
	 * @param arrayLength Length of the arrays
	 */
	virtual void exec(Type, Type){}
	
	/**
	 * This method will be derived to calculate the distance between two arrays of the same length.
	 * @param array1 First array of weights
	 * @param array2 Second array of weights
	 * @param arrayLength Length of the arrays
	 */
	virtual void derived(Type, Type){}
	
	
	/**
	 * Returns the value of the variable result.
	 * @return A double
	 */
	virtual Type getResult()
	{
		return result;
	}
	
	virtual Type getDerivedResult()
	{
		return derivedResult;
	}
	
	
protected:
	Type result; /** The value of the activation function according to the function used */
	Type derivedResult; /** The value of the derived activation function according to the function used */
};


#endif
