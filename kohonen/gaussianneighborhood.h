#ifndef __GAUSSIANNEIGHBORHOODFUNCTION_H__
#define __GAUSSIANNEIGHBORHOODFUNCTION_H__

#include <cmath>
#include "neighborhoodfunction.h"


/**
 * A derived class from Neighborhood that implements the gaussian function.
 */
template <class Type>	
class GaussianNeighborhoodFunction : public NeighborhoodFunction < Type>
{
public:
	
	/**
	 * This method calculates result of the gaussian function to a certain distance giving the current neighborhood size.
	 * The formula of the function is: 
	 * GaussianNeighborhood(distance, neighborhoodSize) = \f$ e^{\frac{- distance^2}{2 \times neighborhoodSize}} \f$.
	 * @param distance The distance from my current position
	 * @param neighborhoodSize the size of the neighborhood
	 */
	void exec(int distance, Type neighborhoodSize)
	{
		//cout << distance << " , " << neighborhoodSize << endl;
		if (distance <= neighborhoodSize)
		{
			neighborhoodSize = 2*pow(neighborhoodSize,2);
			distance *= distance;
			this->neighborhoodValue = exp(((double) -distance)/ neighborhoodSize);
			//cout << "neighborhoodValue: " << this->neighborhoodValue << endl;
		}
		else
			this->neighborhoodValue = 0;
	}


};

#endif

