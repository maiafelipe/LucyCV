#ifndef __NEIGHBORHOOD_H__
#define __NEIGHBORHOOD_H__

#include <iostream>
#include <vector>
#include "templateDefinitions.h"

using namespace std;


/**
 * The neighborhood class implements a base neighborhood that will represent the function used by the Kohonen Self-Organizing Map.
 */
template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction,
			distanceTemplate Distance, 
			typename Type>
class Neighborhood
{
public:
	
	
	/**
	 * The constructor sets the variable neighborhood to 0.
	 */
	Neighborhood(int numWeights = 0)
	{
		this->numWeights = 0;
		this->neighborhoodValue = 0;
		if (numWeights != 0)
			initialize(numWeights);
	}
	
	/**
	 * Cleans the connections vector releasing memory.
	 */
	virtual ~Neighborhood() 
	{
		if (connections.empty() == false)
			connections.clear();
		if (this->numWeights != 0)
			delete central;
	}
	
	/**
	 * 
	 */
	virtual void initialize(int);
	
	
	/**
	 * Adds the central neuron of the neighbohood.
	 * @param neuron the central neuron
	 */
	virtual void addCentral(Neuron<Distance, Type>*);
	
	/**
	 * Adds a neuron connection to the vector
	 * @param neuron the new neuron connection
	 */
	virtual void addConnection(Neuron<Distance, Type>*);
	
	
	/**
	 * Adjust weights of the neighbohood's neurons using a breadth first search
	 * @param learningRate 
	 * @param distance the distance to the winning neuron
	 * @param neighborhoodSize 
	 */
	virtual void adjustWeights(double, int, double);
	
	
	/**
	 * 
	 */
	virtual Type distanceTo(Type*);
	
	
	/**
	 * Return the vector of the neuron's connections
	 * 
	 */
	virtual vector< Neuron<Distance, Type>* > getConnections();
	
	
	/**
	 * 
	 */
	virtual void reinitializeAdjusted();
	
	/**
	 * 
	 */
	virtual Neuron<Distance, Type>* getCentral();

	/**
	 *
	 */
	virtual void setCluster(int);

	/**
	 *
	 */
	virtual int getCluster();

	/**
	 *
	 */
	virtual void setId(int);

	/**
	 *
	 */
	virtual int getId();

	virtual void print();

	virtual void printConnections();
	
protected:
	Type neighborhoodValue; /** The value of neighborhood */
	vector< Neuron<Distance, Type>* > connections; /** The connections of the central neuron of the neighbohood */
	Neuron<Distance, Type>* central; /** The central neuron of the neighborhood */
	NeighborhoodFunction<Type> neighborhoodFunction;
	int numWeights;
	
	
};

template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>	
void Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::initialize(int numWeights)
{
	this->numWeights = numWeights;
	central = new Neuron<Distance, Type>;
	central->initialize(numWeights);
}

template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>	
void Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::addCentral (Neuron<Distance, Type>* neuron)
{
	central = neuron;
}

template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>	
void Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::addConnection (Neuron<Distance, Type>* neuron)
{
	connections.push_back(neuron);
}


template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>	
void Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::adjustWeights (double learningRate, int distance, double neighborhoodSize)
{
	if (central->getAdjusted() == false)
	{	
		neighborhoodFunction.exec(distance, neighborhoodSize);
		//cout << "Valor de vizinhança: " << neighborhoodFunction.getNeighborhoodValue() << endl;
		central->adjustWeights(learningRate, neighborhoodFunction.getNeighborhoodValue());
		central->setAdjusted(true);
	}	
}


template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>	
Type Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::distanceTo (Type* input)
{
	return central->distanceTo(input);
}

template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>	
vector< Neuron<Distance, Type>* > Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::getConnections ()
{
	return connections;
}

template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>	
void Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::reinitializeAdjusted()
{
	central->setAdjusted(false);
}

template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>	
Neuron<Distance, Type>* Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::getCentral()
{
	return central;
}

template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>
void Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::setCluster(int cluster)
{
	central->setCluster(cluster);
}

template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>
int Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::getCluster()
{
	return central->getCluster();
}

template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>
void Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::setId(int id)
{
	central->setId(id);
}

template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>
int Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::getId()
{
	return central->getId();
}

template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>
void Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::print()
{
	central->print();
}

template <neuronTemplate Neuron, neighborhoodFunctionTemplate NeighborhoodFunction, distanceTemplate Distance, typename Type>
void Neighborhood <Neuron, NeighborhoodFunction, Distance, Type>::printConnections()
{
	for (int i = 0; i < connections.size(); i++)
		cout << connections[i]->getId() << " ";
	cout << endl;
}


#endif
