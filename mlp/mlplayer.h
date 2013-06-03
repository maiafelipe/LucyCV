#ifndef __MLPLAYER_H__
#define __MLPLAYER_H__

#include <cfloat>
#include <cstdlib>
#include <algorithm>
#include <vector>

#include "templateDefinitions.h"
//#include "activationfunction.h"

#define INFINITE DBL_MAX

/**
 * This class is responsable for the control of the Layer of the Neural Network.
 * It will contain the data structs responsable for the neurons of the layers.
 */
template < neuronTemplate Neuron, activationTemplate Activation, class Type>
class MLPLayer
{

private:
	Neuron<Activation, Type>* neurons;
	int numNeurons;
	int numWeights;
	double learningRate;
	double momentumRate;
	Type betaActivationFuction;
	
public:
    /**
	 * The default constructor does nothing. If you use it, you will have to initialize;  
	 */
	MLPLayer(){};
	/**
	 * This other constructor initialize the layer according to the following parameters:
	 * @param numNeurons number or neurons in the layer; 
	 * @param numWeights number of weights in each neuron of the layer.
	 */
	MLPLayer(int numNeurons, int numWeights);
	/**
	 * The default destructor deletes the Neurons array
	 */
	~MLPLayer();
	/**
	 * This method also can initialize the layer according to the following parameters:
	 * @param numNeurons number or neurons in the layer; 
	 * @param numWeights number of weights in each neuron of the layer.
	 */
	void initialize(int numNeurons, int numWeights);
	/**
	 * Randomize all the weights.
	 */
	void randomizeWeight();
	/**
	 * This method sets the parameters that can be set after the initialation of the network and alse reset anytime
	 * @param learningRate the learning rate used in the network 
	 * @param momentumRate used in the learning process, may be zero if you don't want Momentum
	 * @param betaActivationFuction parameter Beta used in the activation function.
	 */
	void setParameters(double learningRate, double momentumRate, Type betaActivationFuction);
	/**
	 * This method will execute the layer.
	 * @param in vector with the data input for the layer
	 */
	Type* answer(Type* in);
	/**
	 * This method will return some weight of some neuron.
	 * @param n the neuron target;
	 * @param w the weight target.
	 */
	Type getWeight(int n, int w);
	/**
	 * This method will return number of neurons of the layer.
	 */
	int getNumNeurons();
	/**
	 * This method will set some weight of some neuron.
	 * @param n the neuron target;
	 * @param w the weight target;
     * @param value the value to be set.
	 */
	void setWeight(int n, int w, Type value);
	/**
	 * This method will calculate the delta based on the delta of the previous layer.
	 * @param previousDelta a vector with the deltas of the previous layer;
	 * @param in the input of the layer;
	 */
	Type* getDelta(Type* previousDelta, Type* in);
    /**
	 * This method will fit the weights based on following parameters:
	 * @param delta a vector with the deltas of the layer;
	 * @param in the input of the layer;
	 */
	void fitWeights(Type* delta, Type* in);
	
};

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
MLPLayer <Neuron, Activation, Type>::MLPLayer(int numNeurons, int numWeights)
{
	if (numNeurons != 0 && numWeights != 0)
		initialize(numNeurons, numWeights);
}

template <neuronTemplate Neuron, activationTemplate Activation, typename Type>	
MLPLayer <Neuron, Activation, Type>::~MLPLayer()
{
	if (this->numNeurons != 0)
		delete[] this->neurons;
}

template <neuronTemplate Neuron, activationTemplate Activation, class Type>	
void MLPLayer <Neuron, Activation, Type>::initialize(int numNeurons, int numWeights)
{
	this->numNeurons = numNeurons;
	this->numWeights = numWeights;
	this->neurons = new Neuron<Activation, Type>[numNeurons];
	for (int i = 0; i < numNeurons; i++)
		this->neurons[i].initialize(numWeights);
}

template <neuronTemplate Neuron, activationTemplate Activation, class Type>	
void MLPLayer <Neuron, Activation, Type>::randomizeWeight()
{
	for (int i = 0; i < numNeurons; i++)
		this->neurons[i].initializeWeights();
}

template <neuronTemplate Neuron, activationTemplate Activation, class Type>	
void MLPLayer <Neuron, Activation, Type>::setParameters(double learningRate, double momentumRate, Type betaActivationFuction = 1)
{
	this->learningRate = learningRate;
	this->momentumRate = momentumRate;
	this->betaActivationFuction = betaActivationFuction;
}

template <neuronTemplate Neuron, activationTemplate Activation, class Type>	
Type* MLPLayer <Neuron, Activation, Type>::answer(Type* in)
{
	Type* out;
	out = new Type[this->numNeurons];
	for(int i = 0; i < numNeurons; i++)
	{
		out[i] = this->neurons[i].answer(in);
	}
	return out;
}

template <neuronTemplate Neuron, activationTemplate Activation, class Type>	
Type MLPLayer <Neuron, Activation, Type>::getWeight(int n, int w)
{
	return this->neurons[n].getWeight(w);
}

template <neuronTemplate Neuron, activationTemplate Activation, class Type>	
void MLPLayer <Neuron, Activation, Type>::setWeight(int n, int w, Type value)
{
	this->neurons[n].setWeight(w, value);
}


template <neuronTemplate Neuron, activationTemplate Activation, class Type>	
Type* MLPLayer <Neuron, Activation, Type>::getDelta(Type* previousDelta, Type* in)
{
	Type* delta;
	delta = new Type[this->numNeurons];
	for(int i = 0; i < numNeurons; i++)
	{
		delta[i] = previousDelta[i] * this->neurons[i].derivedAnswer(in);
	}
	return delta;
}

template <neuronTemplate Neuron, activationTemplate Activation, class Type>	
void MLPLayer <Neuron, Activation, Type>::fitWeights(Type* delta, Type* in)
{
	for(int n = 0; n < this->numNeurons; n++)
	{
		for(int w = 0; w < this->neurons[n].getNumWeight(); w++){
			double momentum = this->neurons[n].getMomentum(w); 
			Type newWeight = this->neurons[n].getWeight(w) + momentum + this->learningRate * delta[n] * in[w];
			this->neurons[n].setWeight(w, newWeight);
		}
		int w = this->neurons[n].getNumWeight();
		double momentum = this->neurons[n].getMomentum(w); 
		Type newWeight = this->neurons[n].getWeight(w) + momentum + this->learningRate * delta[n] * (-1);
		this->neurons[n].setWeight(w, newWeight);
		
	}
}
template <neuronTemplate Neuron, activationTemplate Activation, class Type>	
int MLPLayer <Neuron, Activation, Type>::getNumNeurons()
{
	return this->numNeurons;
}





#endif
