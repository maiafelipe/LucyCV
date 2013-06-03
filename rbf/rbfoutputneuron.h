#ifndef __RBFOUTPUTNEURON_H__
#define __RBFOUTPUTNEURON_H__

#include "templateDefinitions.h"
// #include "activationfunction.h"
#include <cstdlib>


#define FACTOR 10000


template <activationTemplate Activation, class Type>
class RBFOutputNeuron
{

public:
	RBFOutputNeuron(int);
	~RBFOutputNeuron();
	void initialize(int);
	void initializeWeights();
	Type activationFunction(Type*);
	void adjustWeights(Type*, Type, double);
	void setThreshold(Type);
	void print();
	
protected:
	Type* weights;
	Type threshold;
	int numWeights;
	Activation<Type> activation;
	
};

template <activationTemplate Activation, class Type>
RBFOutputNeuron<Activation, Type>::RBFOutputNeuron(int weights = 0)
{
	this->numWeights = 0;
	if (weights != 0)
		initialize(weights);
}

template <activationTemplate Activation, class Type>
RBFOutputNeuron<Activation, Type>::~RBFOutputNeuron()
{
	if (this->numWeights != 0)
		delete[] this->weights;
}

template <activationTemplate Activation, class Type>
void RBFOutputNeuron<Activation, Type>::initialize(int weights)
{
	this->numWeights = weights;
	this->weights = new Type[weights];
	initializeWeights();
}


template <activationTemplate Activation, class Type>
Type RBFOutputNeuron<Activation, Type>::activationFunction(Type* input)
{
	Type result = 0;
	for (int i = 0; i < this->numWeights; i++)
		result += this->weights[i] * input[i];
	
	this->activation.exec(result, this->threshold);
	return this->activation.getResult(); 
}

template <activationTemplate Activation, class Type>
void RBFOutputNeuron<Activation, Type>::adjustWeights(Type* input, Type intended, double learningRate)
{
	Type result = 0;
	for (int i = 0; i < this->numWeights; i++)
		result += this->weights[i] * input[i];

	this->activation.derived(result, this->threshold);
	Type delta = (intended - this->activation.getResult() ) * this->activation.getDerivedResult();

	Type factor = learningRate * delta;

	for (int i = 0; i < this->numWeights; i++)
	{
		this->weights[i] += factor * input[i];
	}
}

template <activationTemplate Activation, class Type>
void RBFOutputNeuron<Activation, Type>::setThreshold(Type value)
{
	this->threshold = value;
}

template <activationTemplate Activation, class Type>
void RBFOutputNeuron<Activation, Type>::print()
{
	for (int i = 0; i < this->numWeights; i++)
		cout << this->weights[i] << " ";
	cout << endl;
}

template <activationTemplate Activation, class Type>
void RBFOutputNeuron<Activation, Type>::initializeWeights()
{
	for (int i = 0; i < this->numWeights; i++)
		this->weights[i] = (rand()%FACTOR)/(double) FACTOR;
}

#endif
