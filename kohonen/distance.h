#ifndef __DISTANCE_H__
#define __DISTANCE_H__

#include <iostream>

using namespace std;


/**
 * A base distance class used to evaluate the winner neuron in the Kohonen Self-organizing Map.
 */

template <class Type>
class Distance
{
public:
	/**
	 * The constructor sets the result distance to 0.
	 */
	Distance()
	{
		clean = false;
		result = 0;
	}
	
	/**
	 * This method will be derived to calculate the distance between two arrays of the same length.
	 * @param array1 First array of weights
	 * @param array2 Second array of weights
	 * @param arrayLength Length of the arrays
	 */
	virtual void exec(Type*, Type*, int){}
	
	
	/**
	 * Returns the value of the variable result.
	 * @return A double
	 */
	virtual Type getResult()
	{
		return result;
	}
	
	/**
	 * Returns the array distanceVector.
	 * @return An array of doubles
	 */
	virtual Type* getDistanceVector()
	{
		return distanceVector;
	}
	
protected:
	bool clean;
  	Type* distanceVector; /** The resulting array of the subtraction between two arrays */
	Type result; /** The value of the distanceVector according to the distance used */
};


#endif