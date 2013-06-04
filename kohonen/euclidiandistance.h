#ifndef __EUCLIDIANDISTANCE_H__
#define __EUCLIDIANDISTANCE_H__

#include <cmath>
#include "distance.h"

/**
 * Derived class from Distance that implements an euclidian distance.
 */
template <class Type>
class EuclidianDistance : public Distance<Type>
{
	
public:
	/**
	 * Deletes the distanceVector variable releasing memory
	 */
	~EuclidianDistance()
	{
		if (this->clean == true)
			delete[] this->distanceVector;
	}
	
	/**
	 * Method that implements the euclidian distance between two arrays.
	 * The formula is given by:
	 * \f$ distance(array1, array2, arrayLength) = \sqrt{\sum_{i=1}^{arrayLength} (array1_i - array2_i)} \f$
	 * @param array1 First array of weights
	 * @param array2 Second array of weights
	 * @param arrayLength Length of the arrays
	 */
	void exec(Type* array1, Type* array2, int arrayLength)
	{
		this->distanceVector = new Type[arrayLength];
		this->clean = true;
		for (int i = 0; i < arrayLength; i++)
		{
			this->distanceVector[i] = (array1[i] - array2[i]);
			this->result += pow(this->distanceVector[i], 2);
		}
		this->result = sqrt(this->result);
		if (this->result < 0.000001)
			this->result = 0;
	}
	
};

#endif
