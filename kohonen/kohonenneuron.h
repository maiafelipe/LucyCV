#ifndef __KOHONENNEURON_H__
#define __KOHONENNEURON_H__

#include <cstdlib>

#include "templateDefinitions.h"
#include "distance.h"

#define FACTOR 10000


template <distanceTemplate Distance, class Type>
class KohonenNeuron
{

public:
	KohonenNeuron(int);
	~KohonenNeuron();
	void initialize(int);
	void setRanking(int);
	Type distanceTo(Type*);
	void adjustWeights(double, Type);
	void print();
	void setWeights(Type*);
	bool getAdjusted();
	void setAdjusted(bool);
	int getId();
	void setId(int);
	int getCluster();
	void setCluster(int);
	
protected:
	Type* weights;
	int numWeights;
	int ranking;
	int id;
	bool adjusted;
	int cluster;
	Distance<Type> distance;
	void initializeWeights();
	void normalizeWeights();
};

template <distanceTemplate Distance, class Type>
KohonenNeuron<Distance, Type>::KohonenNeuron(int weights = 0)
{
	this->numWeights = 0;
	if (weights != 0)
		initialize(weights);
}

template <distanceTemplate Distance, class Type>
KohonenNeuron<Distance, Type>::~KohonenNeuron()
{
	if (this->numWeights != 0)
		delete[] this->weights;
}

template <distanceTemplate Distance, class Type>
void KohonenNeuron<Distance, Type>::initialize(int weights)
{
	this->adjusted = false;
	this->numWeights = weights;
	this->weights = new Type[weights];
	initializeWeights();
	
}

template <distanceTemplate Distance, class Type>
void KohonenNeuron<Distance, Type>::setRanking(int ranking)
{
	this->ranking = ranking;
}

template <distanceTemplate Distance, class Type>
Type KohonenNeuron<Distance, Type>::distanceTo(Type* input)
{
	distance.exec(this->weights, input, this->numWeights);
	return distance.getResult(); 
}

template <distanceTemplate Distance, class Type>
void KohonenNeuron<Distance, Type>::adjustWeights(double learningRate, Type neighborhoodValue)
{
	Type* distanceVetor = distance.getDistanceVector();
	Type learningFactor = learningRate * neighborhoodValue;
	//cout << "Neuron" << endl;
	for (int i = 0; i < this->numWeights; i++)
	{
		this->weights[i] += learningFactor * distanceVetor[i];
		//cout << learningFactor << " * " << distanceVetor[i] << endl;
	}
	normalizeWeights();
}

template <distanceTemplate Distance, class Type>
void KohonenNeuron<Distance, Type>::normalizeWeights()
{
	Type norm = 0;
	for (int i = 0; i < this->numWeights; i++)
		norm += this->weights[i]*weights[i];
	norm = sqrt(norm);
	for (int i = 0; i < this->numWeights; i++)
		this->weights[i] /= norm;
}

template <distanceTemplate Distance, class Type>
void KohonenNeuron<Distance, Type>::print()
{
	for (int i = 0; i < this->numWeights; i++)
		cout << this->weights[i] << " ";
	cout << endl;
}

template <distanceTemplate Distance, class Type>
void KohonenNeuron<Distance, Type>::initializeWeights()
{
	for (int i = 0; i < this->numWeights; i++)
		this->weights[i] = (rand()%FACTOR)/(double) FACTOR;
}

template <distanceTemplate Distance, class Type>
void KohonenNeuron<Distance, Type>::setWeights(Type* weights)
{
	for (int i = 0; i < this->numWeights; i++)
		this->weights[i] = weights[i];
}

template <distanceTemplate Distance, class Type>
bool KohonenNeuron<Distance, Type>::getAdjusted()
{
	return this->adjusted;
}

template <distanceTemplate Distance, class Type>
void KohonenNeuron<Distance, Type>::setAdjusted(bool newAdjusted)
{
	this->adjusted = newAdjusted;
}

template <distanceTemplate Distance, class Type>
int KohonenNeuron<Distance, Type>::getId()
{
	return this->id;
}

template <distanceTemplate Distance, class Type>
void KohonenNeuron<Distance, Type>::setId(int newId)
{
	this->id = newId;
}

template <distanceTemplate Distance, class Type>
int KohonenNeuron<Distance, Type>::getCluster()
{
	return this->cluster;
}

template <distanceTemplate Distance, class Type>
void KohonenNeuron<Distance, Type>::setCluster(int newCluster)
{
	this->cluster = newCluster;
}


#endif
