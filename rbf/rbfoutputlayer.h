#ifndef __RBFOUTPUTLAYER_H__
#define __RBFOUTPUTLAYER_H__

#include <cfloat>
#include <cstdlib>
#include <algorithm>
#include <vector>

#include "templateDefinitions.h"
// #include "activationfunction.h"

#define INFINITE DBL_MAX


template < neuronTemplate Neuron, activationTemplate Activation, class Type>
class RBFOutputLayer
{
public:
	RBFOutputLayer(int, int);
	~RBFOutputLayer();
	void initialize(int, int);
	void train(Type**, Type**, int);
	Type* answer(Type*);
	void setLearningRate(double);
	void setMinimumError(double);
	void setMaximumEpochs(int);
	void setThreshold(Type);
	void print();
	void randWeights();
private:
	Neuron<Activation, Type>* neurons;
	int numNeurons;
	int numWeights;
	double learningRate;
	double minimumError;
	int maximumEpochs;
	
};

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
RBFOutputLayer <Neuron, Activation, Type>::RBFOutputLayer(int numNeurons = 0, int numWeights = 0)
{
	this->numNeurons = 0;
	if (numNeurons != 0)
		initialize(numNeurons, numWeights);
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFOutputLayer <Neuron, Activation, Type>::initialize(int numNeurons, int numWeights)
{
	this->numNeurons = numNeurons;
	this->numWeights = numWeights;
	this->neurons = new Neuron<Activation, Type>[numNeurons];
	for (int i = 0; i < numNeurons; i++)
		this->neurons[i].initialize(numWeights);
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
RBFOutputLayer <Neuron, Activation, Type>::~RBFOutputLayer()
{
	
	if (this->numNeurons != 0)
		delete[] this->neurons;
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
Type* RBFOutputLayer <Neuron, Activation, Type>::answer(Type* input)
{
	Type* answer = new Type[this->numNeurons];
	for (int i = 0; i < this->numNeurons; i++)
		answer[i] = this->neurons[i].activationFunction(input);
	return answer;
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFOutputLayer <Neuron, Activation, Type>::train(Type** inputSet, Type** intended , int numInputs)
{
	bool condition = true;
	double previousError = 0;
	double currentError;
	int epochs = 0;
	
	Type** obtained;
	obtained = new Type*[numInputs];
	for (int i = 0; i < numInputs; i++)
		obtained[i] = new Type[this->numNeurons];
	
	cout << endl << endl << "_____" << "Output layer" << "_____" << endl << endl;
	
	cout << "Neurons: " << endl;
	
	for (int i = 0; i < this->numNeurons; i++)
		neurons[i].print();
	
	
	while(condition)
	{
		
		//shuffle(inputSet, numInputs, this->numWeights);
		
		/**
		cout << endl << "New iteration" << endl;
		for (int i = 0; i < numInputs; i++)
		{
			for (int j = 0; j < numWeights; j++)
				cout << inputSet[i][j] << " ";
			cout << endl;
		}
		cout << endl;
		*/
		
		/**
		 * Bloco de ajuste de pesos
		 */
		
		
		for (int i = 0; i < numInputs; i++)
		{
			for (int j = 0; j < this->numNeurons; j++)
			{
				obtained[i][j] = this->neurons[j].activationFunction(inputSet[i]);
				this->neurons[j].adjustWeights(inputSet[i], intended[i][j], this->learningRate);
			}
		}
		
		/** Calculando o erro medio */

		double mediumError = 0;
		for (int i = 0; i < numInputs; i++)
		{
			double localError = 0;
			for (int j = 0; j < this->numNeurons; j++)
			{
				localError += pow((intended[i][j] - obtained[i][j]),2);
			}
			localError /= 2;
			mediumError += localError;
		}
		mediumError /= numInputs;
		
		/** pesos e condicao de parada */
		currentError = mediumError; //Calcular erro medio;
		if ( ((abs(currentError - previousError) < this->minimumError)) ||  (epochs > this->maximumEpochs))
			condition = false;
		epochs += 1;
		previousError = currentError; // Calcular erro medio;

	}
	
	for (int i = 0; i < numInputs; i++)
		delete[] obtained[i];
	delete[] obtained;
	
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFOutputLayer <Neuron, Activation, Type>::setLearningRate(double learningRate)
{
	this->learningRate = learningRate;
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFOutputLayer <Neuron, Activation, Type>::setMinimumError(double minimumError)
{
	this->minimumError = minimumError;
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFOutputLayer <Neuron, Activation, Type>::setMaximumEpochs(int maximumEpochs)
{
	this->maximumEpochs = maximumEpochs;
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFOutputLayer <Neuron, Activation, Type>::setThreshold(Type threshold)
{
	for (int i = 0; i < this->numNeurons; i++)
		this->neurons[i].setThreshold(threshold);
}


template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFOutputLayer <Neuron, Activation, Type>::print()
{
	for (int i = 0; i < this->numNeurons; i++)
	{
		cout << "Neuron" << i << ":" << endl;
		this->neurons[i].print();
		cout << endl;
	}
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFOutputLayer <Neuron, Activation, Type>::randWeights()
{
	for (int i = 0; i < this->numNeurons; i++)
	{
		this->neurons[i].initializeWeights();
	}
}

#endif
