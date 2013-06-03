#ifndef __RBFHIDDENLAYER_H__
#define __RBFHIDDENLAYER_H__

#include <cfloat>
#include <cstdlib>
#include <algorithm>
#include <vector>

#include "templateDefinitions.h"
// #include "activationfunction.h"

#define INFINITE DBL_MAX


template < neuronTemplate Neuron, activationTemplate Activation, class Type>
class RBFHiddenLayer
{
public:
	RBFHiddenLayer(int, int);
	~RBFHiddenLayer();
	void initialize(int, int);
	void train(Type**, int);
	Type* answer(Type*);
	void print();
	void randWeights();
	void setWeightsNeuron(int, Type*);
private:
	Neuron<Activation, Type>* neurons;
	int numNeurons;
	int numWeights;
	
};

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
RBFHiddenLayer <Neuron, Activation, Type>::RBFHiddenLayer(int numNeurons = 0, int numWeights = 0)
{
	this->numNeurons = 0;
	if (numNeurons != 0)
		initialize(numNeurons, numWeights);
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFHiddenLayer <Neuron, Activation, Type>::initialize(int numNeurons, int numWeights)
{
	this->numNeurons = numNeurons;
	this->numWeights = numWeights;
	this->neurons = new Neuron<Activation, Type>[numNeurons];
	for (int i = 0; i < numNeurons; i++)
		this->neurons[i].initialize(numWeights);
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
RBFHiddenLayer <Neuron, Activation, Type>::~RBFHiddenLayer()
{
	
	if (this->numNeurons != 0)
		delete[] this->neurons;
}


template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
Type* RBFHiddenLayer <Neuron, Activation, Type>::answer(Type* input)
{
	Type* answer = new Type[this->numNeurons];
	//Type answer[this->numNeurons];
	
	for (int i = 0; i < this->numNeurons; i++)
	{
		answer[i] = this->neurons[i].activationFunction(input);
		//cout << answer[i] << " ";
	}
	//cout << endl;
	return answer;
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFHiddenLayer <Neuron, Activation, Type>::train(Type** inputSet, int numInputs)
{
	
	bool condition = true;
	bool changed;
	vector< vector<int> > oldSets(this->numNeurons);
	for (int i = 0; i < this->numNeurons; i++)
		oldSets[i] = vector<int>(1,-1);
	int k = 1;
	
	
	/*cout << "Neurons: " << endl;
	for (int i = 0; i < this->numNeurons; i++)
		neurons[i].print();
	
	cout << endl << endl;*/
	
	while (condition == true)
	{
		
		/*cout << endl << "New iteration" << endl;
		for (int i = 0; i < numInputs; i++)
		{
			for (int j = 0; j < numWeights; j++)
				cout << inputSet[i][j] << " ";
			cout << endl;
		}
		cout << endl;
		*/
		
		vector< vector<int> > sets(this->numNeurons);
		for (int i = 0; i < numInputs; i++)
		{
			int indexMinimum = 0;
			Type minimuResult =  this->neurons[0].euclidianDistance(inputSet[i]);
			for (int j = 1; j < this->numNeurons; j++)
			{
				Type temporaryResult = this->neurons[j].euclidianDistance(inputSet[i]);
				if (temporaryResult < minimuResult)
				{
					minimuResult = temporaryResult;
					indexMinimum = j;
				}
			}
			sets[indexMinimum].push_back(i);
		}
		
		for (int i = 0; i < this->numNeurons; i++)
		{
			if (sets[i].size() != 0)
			{
				/** Creating the sets of the inputs to each neuron */
				
				Type** auxiliaryMatrix;
				auxiliaryMatrix = new Type*[sets[i].size()];
				for (int j = 0; j < sets[i].size(); j++)
					auxiliaryMatrix[j] = new Type[this->numWeights];
				
				for (int j = 0; j < sets[i].size(); j++)
					for (int k = 0; k < this->numWeights; k++)
						auxiliaryMatrix[j][k] = inputSet[ sets[i][j] ][k];
					
				/*cout << "Neuron " <<  i << " : " <<  sets[i].size() << endl;
				for (int j = 0; j < sets[i].size(); j++)
				{
					for (int k = 0; k < this->numWeights; k++)
						cout << auxiliaryMatrix[j][k] << " ";
					cout << endl;
				}*/
					
				
				this->neurons[i].adjustWeights(auxiliaryMatrix, sets[i].size());
				
				/** deleting the auxiliary matrix */
				for (int j = 0; j < sets[i].size(); j++)
					delete[] auxiliaryMatrix[j];
				delete[] auxiliaryMatrix;
			}
		}
		
		for (int i = 0; i < this->numNeurons; i++)
		{
			changed = false;
			
			
			if (equal(sets[i].begin(), sets[i].end(), oldSets[i].begin()) == false)
				changed = true;
			oldSets[i] = sets[i];
		}
		
		if (!changed)
				condition = false;
		k++;
	}
	/** Calculating the sigma */
	for (int i = 0; i < this->numNeurons; i++)
	{
		//this->neurons[i].print();
		Type sigma = 0;
		Type** auxiliaryMatrix;
		auxiliaryMatrix = new Type*[oldSets[i].size()];
		for (int j = 0; j < oldSets[i].size(); j++)
			auxiliaryMatrix[j] = new Type[this->numWeights];

		for (int j = 0; j < oldSets[i].size(); j++)
			for (int k = 0; k < this->numWeights; k++)
				auxiliaryMatrix[j][k] = inputSet[ oldSets[i][j] ][k];
		
		for (int j = 0; j < oldSets[i].size(); j++)
		{
			Type distance = this->neurons[i].euclidianDistance(auxiliaryMatrix[j]);

			sigma += pow(distance, 2);
		}

		
		if (oldSets[i].size() == 0)
			cout << "Neuron: " << i << " - has no cluster"  << endl;
		
		if (oldSets[i].size() == 0)
			sigma = 1;
		else
			sigma /= oldSets[i].size();
		

		this->neurons[i].setSigma(sigma);
			
		/** deleting the auxiliary matrix */
				for (int j = 0; j < oldSets[i].size(); j++)
					delete[] auxiliaryMatrix[j];
				delete[] auxiliaryMatrix;
	}
	
	cout << endl << endl << "Neurons: " << endl;
	for (int i = 0; i < this->numNeurons; i++)
		neurons[i].print();
	
}


template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFHiddenLayer <Neuron, Activation, Type>::setWeightsNeuron(int index, Type* weights)
{
	neurons[index].setWeights(weights);
}




template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFHiddenLayer <Neuron, Activation, Type>::print()
{
	for (int i = 0; i < this->numNeurons; i++)
	{
		cout << "Neuron" << i << ":" << endl;
		this->neurons[i].print();
		cout << endl;
	}
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
void RBFHiddenLayer <Neuron, Activation, Type>::randWeights()
{
	for (int i = 0; i < this->numNeurons; i++)
	{
		this->neurons[i].initializeWeights();
	}
}


#endif
