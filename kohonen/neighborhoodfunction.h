#ifndef __NEIGHBORHOODFUNCTION_H__
#define __NEIGHBORHOODFUNCTION_H__



template <class Type>
class NeighborhoodFunction
{
public:
	virtual void exec(int, Type){}
	/**
	 * Returns the value of the variable neighborhood.
	 * @return A template type
	 */
	virtual Type getNeighborhoodValue() 
	{
		return this->neighborhoodValue;
	}

protected:
	Type neighborhoodValue;
};

#endif