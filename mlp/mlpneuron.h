#ifndef __MLPNEURON_H__
#define __MLPNEURON_H__

#include "templateDefinitions.h"
//#include "activationfunction.h"
#include <cstdlib>

#define FACTOR 10000

#define activationTemplate template <class> class

/**
 * This class is responsable for the control of the Neurons of the Neural Network.
 * It will contain the data structs responsable for the weights of the neurons.
 */

template <activationTemplate Activation, class Type>
class MLPNeuron
{

public:
	/**
	 * The default constructor does nothing. If you use it, you will have to initialize;  
	 */
	MLPNeuron(){}
	/**
	 * This other constructor initialize the layer according to the following parameters:
	 * @param numWeights number of weights in of the neuron.
	 */
	MLPNeuron(int numWeights);
	/**
	 * The default destructor deletes the weights array
	 */
	~MLPNeuron();
	/**
	 * This method also can initialize the layer according to the following parameters:
	 * @param numWeights number of weights in of the neuron.
	 */
	void initialize(int numWeights);
	/**
	 * This method will execute the neuron.
	 * @param input vector with the data input for the neuron.
	 */
	Type answer(Type* input);
	/**
	 * This method will return the answer of the neuron using the derived of the activation function.
	 * @param input vector with the data input for the neuron.
	 */
	Type derivedAnswer(Type* input);
	/**
	 * This method will set some weight of the neuron.
	 * @param w the weight target;
     * @param value the value to be set.
	 */
	void setWeight(int w, Type weight);
	/**
	 * This method will return some weight of the neuron.
	 * @param w the weight target.
	 */
	Type getWeight(int w);
	/**
	 * This method will return number of weights of the neuron.
	 */
	int getNumWeight();
	/**
	 * Initialize the weights of neuron randomly.
	 */
	void initializeWeights();
	/**
	 * Set the momentum of the neuron.
	 * @param momentumRate the rate of the momentum.
	 */
	void setMomentumRate(double momentumRate);
	/**
	 * Return the momentum of the neuron.
	 */
	double getMomentum(int w);
	/**
	 * Set the beta of the activation function.
	 * @param betaActivationFuction the beta of the activation function.
	 */
	void setBetaActivationFunction(Type betaActivationFuction);
	
protected:
	Type* weights;
	Type* oldWeights; //utilizado no momentum
	Type threshold;
	int numWeights;
	Activation<Type> activation;
	double momentumRate;
	Type betaActivationFuction;
	
};

template <activationTemplate Activation, class Type>
MLPNeuron<Activation, Type>::MLPNeuron(int numWeights)
{
	if (weights != 0)
		initialize(numWeights);
}

template <activationTemplate Activation, class Type>
MLPNeuron<Activation, Type>::~MLPNeuron()
{
	if (this->numWeights != 0)
		delete[] this->weights;
}

template <activationTemplate Activation, class Type>
void MLPNeuron<Activation, Type>::initialize(int numWeights)
{
	this->numWeights = numWeights;
	this->momentumRate = 0;
	this->weights = new Type[numWeights+1];
	this->betaActivationFuction = 1;
	initializeWeights();
}

template <activationTemplate Activation, class Type>
void MLPNeuron<Activation, Type>::setMomentumRate(double momentumRate)
{
	this->momentumRate = momentumRate;
	if(this->momentumRate != 0.0)
	{
		this->oldWeights = new Type[this->numWeights+1];
		for(int w = 0; w <= numWeights; w++)
		{
			this->oldWeights[w] = weights[w]; 		
		}
	}
}

template <activationTemplate Activation, class Type>
void MLPNeuron<Activation, Type>::initializeWeights()
{
	for (int i = 0; i <= this->numWeights; i++)
	{
		this->weights[i] = (2*(rand()%FACTOR)/(double) FACTOR) - 1;
		//cout<<" "<<weights[i]<<" ";
	}
	//cout<<endl;
}

template <activationTemplate Activation, class Type>
Type MLPNeuron<Activation, Type>::answer(Type* input)
{
	Type result = 0;
	for (int i = 0; i < this->numWeights; i++)
		result += this->weights[i] * input[i];
	
	result -= this->weights[numWeights+1];
	
	this->activation.exec(result, betaActivationFuction);
	return this->activation.getResult(); 
}

template <activationTemplate Activation, class Type>
Type MLPNeuron<Activation, Type>::derivedAnswer(Type* input)
{
	Type result = 0;
	for (int i = 0; i < this->numWeights; i++)
		result += this->weights[i] * input[i];
		
	result -= this->weights[numWeights+1];
	
	this->activation.derived(result, betaActivationFuction);
	return this->activation.getDerivedResult(); 
}

template <activationTemplate Activation, class Type>
void MLPNeuron<Activation, Type>::setWeight(int i, Type weight)
{
	if(this->momentumRate){
		this->oldWeights[i] = this->weights[i];
	}
	this->weights[i] = weight;
}

template <activationTemplate Activation, class Type>
Type MLPNeuron<Activation, Type>::getWeight(int w){
    return this->weights[w];
}

template <activationTemplate Activation, class Type>
double MLPNeuron<Activation, Type>::getMomentum(int w){
	if(momentumRate != 0 )
	{
    double momentum = this->momentumRate * (weights[w] - oldWeights[w]);
    return momentum;
    }
    return 0.0; 
}

template <activationTemplate Activation, class Type>
void MLPNeuron<Activation, Type>::setBetaActivationFunction(Type betaActivationFuction){
    this->betaActivationFuction = betaActivationFuction;
}

template <activationTemplate Activation, class Type>
int MLPNeuron<Activation, Type>::getNumWeight(){
    return this->numWeights;
}

#endif
