#ifndef __RBFNETWORK_H__
#define __RBFHNETWORK_H__

//#include "normalize.cpp"
#include "network.h"
#include "templateDefinitions.h"


template < layerTemplate LayerHidden, 
			neuronTemplate NeuronHidden, 
			activationTemplate ActivationHidden,
			layerTemplate LayerOutput, 
			neuronTemplate NeuronOutput, 
			activationTemplate ActivationOutput,
			class Type >
			
class RBFNetwork : public Network<Type>
{
public:
	/**
	 * The default constructor sets the network as non-initialized.
	 */
	RBFNetwork();
	/**
	 * This other constructor initialize the network according to the 
	 * following parameters:
	 * @param numWeightsInput number of neurons in the input layer; 
	 * @param numHiddenNeurons number of neurons in the hidden layer;
	 * @param numOutputNeurons number of neurons in the output layer.
	 */
	RBFNetwork(int numWeightsInput, int numHiddenNeurons, int numOutputNeurons);
	/**
	 * The default destructor don't need to deletes nothing, because 
	 * the neurons in the layers will be deleteds by the layers destructors.
	 */
	~RBFNetwork();
	/**
	 * This method also can initialize the network according to the 
	 * following parameters:
	 * @param numWeightsInput number of neurons in the input layer;
	 * @param numHiddenNeurons number of neurons in the hidden layer; 
	 * @param numOutputNeurons number of neurons in the output layer.
	 */
	void initialize(int numWeightsInput, int numHiddenNeurons, int numOutputNeurons);
	/**
	 * This method executes the network training.
	 * @param inputSet matrix with the data input for the network
	 * @param intended vector with the data intented output for the network
	 * @param numInputs of instances used on the training
	 */
	void train(Type** inputSet, Type** intended, int numInputs);
	/**
	 * This method will execute the network.
	 * @param in vector with the data input for the network
	 */
	Type* answer(Type* input);
	/**
	 * This method also will execute the network, but for a set of inputs.
	 * @param in matrix with the data input for the network
	 * @param inSize number of inputs that will be executed
	 */
	Type** answer(Type** input, int inSize);
	/**
	 * This method sets the parameters that can be set after the 
	 * initialation of the network and also reset anytime:
	 * @param learningRate the learning rate used in the network 
	 * @param minimumError representes the diference between the medium 
	 * square erro of two successive epochs and is used as stopping condition
	 * @param maximumEpochs maximum number of epochs it can train
	 * @param threshold sets the threshold in the outputLayer.
	 */
	void setParameters(double learningRate, double minimumError, int maximumEpochs, Type threshold);
	/**
	 * This method sets the weights in a neuron of the hidden layer.
	 * @param index the neuron of the hidden layer that you want to set; 
	 * @param weights vector with the values ti be sets.
	 */
	void setWeightsHiddenNeuron(int index, Type* weights);
	/**
	 * Randomize all the weights.
	 */
	void randomizeWeight();
	/**
	 * Saves the network into a file, whit all the parameters and weight 
	 * of all the neurons.
	 * @param filename name of the file
	 */
	void save(const char* filename);
	/**
	 * Recover the network from file. The network must be non-initialized.
	 * @param filename name of the file
	 */
	void load(const char* filename);
	double getAccuracy(){}
	void setAccuracy(double){}
	
	
protected:
	LayerHidden < NeuronHidden, ActivationHidden, Type > hiddenLayer;
	LayerOutput < NeuronOutput, ActivationOutput, Type> outputLayer;
	int numWeightsInput;
	int numHiddenNeurons;
	int numOutputNeurons;
	double learningRate;
	double minimumError;
	int maximumEpochs;
	Type threshold;
};

template < layerTemplate LayerHidden,neuronTemplate NeuronHidden, activationTemplate ActivationHidden,
			layerTemplate LayerOutput, neuronTemplate NeuronOutput, activationTemplate ActivationOutput,
			class Type >
RBFNetwork<LayerHidden, NeuronHidden, ActivationHidden, LayerOutput, NeuronOutput, ActivationOutput, Type>::RBFNetwork(int numWeightsInput, int numHiddenNeurons , int numOutputNeurons)
{
	if ( (numWeightsInput != 0) && (numHiddenNeurons != 0) && (numOutputNeurons != 0) )
		initialize(numWeightsInput, numHiddenNeurons, numOutputNeurons);
}

template < layerTemplate LayerHidden,neuronTemplate NeuronHidden, activationTemplate ActivationHidden,
			layerTemplate LayerOutput, neuronTemplate NeuronOutput, activationTemplate ActivationOutput,
			class Type >
RBFNetwork<LayerHidden, NeuronHidden, ActivationHidden, LayerOutput, NeuronOutput, ActivationOutput, Type>::~RBFNetwork()
{

}

template < layerTemplate LayerHidden,neuronTemplate NeuronHidden, activationTemplate ActivationHidden,
			layerTemplate LayerOutput, neuronTemplate NeuronOutput, activationTemplate ActivationOutput,
			class Type >
void RBFNetwork<LayerHidden, NeuronHidden, ActivationHidden, LayerOutput, NeuronOutput, ActivationOutput, Type>::initialize(int numWeightsInput, int numHiddenNeurons , int numOutputNeurons)
{
	this->numWeightsInput = numWeightsInput;
	this->numHiddenNeurons = numHiddenNeurons;
	this->numOutputNeurons = numOutputNeurons;
	hiddenLayer.initialize(numHiddenNeurons, numWeightsInput);
	outputLayer.initialize(numOutputNeurons, numHiddenNeurons);
}


template < layerTemplate LayerHidden,neuronTemplate NeuronHidden, activationTemplate ActivationHidden,
			layerTemplate LayerOutput, neuronTemplate NeuronOutput, activationTemplate ActivationOutput,
			class Type >
void RBFNetwork<LayerHidden, NeuronHidden, ActivationHidden, LayerOutput, NeuronOutput, ActivationOutput, Type>::setParameters(double learningRate, double minimumError, int maximumEpochs, Type threshold)
{
	this->learningRate = learningRate;
	this->minimumError = minimumError;
	this->maximumEpochs = maximumEpochs;
	this->threshold = threshold;
	outputLayer.setMinimumError(minimumError);
	outputLayer.setLearningRate(learningRate);
	outputLayer.setMaximumEpochs(maximumEpochs);
	outputLayer.setThreshold(threshold);
}


template < layerTemplate LayerHidden,neuronTemplate NeuronHidden, activationTemplate ActivationHidden,
			layerTemplate LayerOutput, neuronTemplate NeuronOutput, activationTemplate ActivationOutput,
			class Type >
void RBFNetwork<LayerHidden, NeuronHidden, ActivationHidden, LayerOutput, NeuronOutput, ActivationOutput, Type>::setWeightsHiddenNeuron(int index, Type* weights)
{
	hiddenLayer.setWeightsNeuron(index, weights);
}

template < layerTemplate LayerHidden,neuronTemplate NeuronHidden, activationTemplate ActivationHidden,
			layerTemplate LayerOutput, neuronTemplate NeuronOutput, activationTemplate ActivationOutput,
			class Type >
void RBFNetwork<LayerHidden, NeuronHidden, ActivationHidden, LayerOutput, NeuronOutput, ActivationOutput, Type>::randomizeWeight()
{
	hiddenLayer.randWeights();
	outputLayer.randWeights();
}




template < layerTemplate LayerHidden,neuronTemplate NeuronHidden, activationTemplate ActivationHidden,
			layerTemplate LayerOutput, neuronTemplate NeuronOutput, activationTemplate ActivationOutput,
			class Type >
void RBFNetwork<LayerHidden, NeuronHidden, ActivationHidden, LayerOutput, NeuronOutput, ActivationOutput, Type>::train(Type** inputSet, Type** intended, int numInputs)
{
	cout << "Treinando a camada hidden" << endl;
	hiddenLayer.train(inputSet, numInputs);
	//cout << "Treinou o hidden" << endl;
	Type** hiddenOutput;
	hiddenOutput = new Type* [numInputs];
	for (int i = 0; i < numInputs; i++)
		hiddenOutput[i] = new Type[this->numHiddenNeurons];
	
	for (int i = 0; i < numInputs; i++)
		hiddenOutput[i] = hiddenLayer.answer(inputSet[i]);
	
	//normalize(hiddenOutput, numInputs, this->numHiddenNeurons);
	
	/** Normalizar os vstores de entrada agora */

	cout << "Treinando a camada output" << endl;
	outputLayer.train(hiddenOutput, intended, numInputs);

	for (int i = 0; i < numInputs; i++)
		delete[] hiddenOutput[i];
	delete[] hiddenOutput;
}

template < layerTemplate LayerHidden,neuronTemplate NeuronHidden, activationTemplate ActivationHidden,
			layerTemplate LayerOutput, neuronTemplate NeuronOutput, activationTemplate ActivationOutput,
			class Type >
Type* RBFNetwork<LayerHidden, NeuronHidden, ActivationHidden, LayerOutput, NeuronOutput, ActivationOutput, Type>::answer(Type* input)
{
	//cout << "First answer" << endl;
	Type* hiddenOutput =  hiddenLayer.answer(input);

	cout << "Answer hidden" << endl;
	for (int i = 0; i < this->numHiddenNeurons; i++)
		cout << hiddenOutput[i] << " ";
	cout << endl;
	
	
	/** Normalizar os vetores de entrada agora */
	
	Type minimum = INFINITE, maximum = 0, diference;
	for (int j = 0; j < this->numHiddenNeurons; j++)
	{
		if (hiddenOutput[j] < minimum)
			minimum = hiddenOutput[j];
		if (hiddenOutput[j] > maximum)
			maximum = hiddenOutput[j];
	}
	diference = maximum - minimum;
	for (int j = 0; j < this->numHiddenNeurons; j++)
		hiddenOutput[j] = (hiddenOutput[j] - minimum) /diference;
	
	//outputLayer.print();
	
	
	Type* networkOutput = outputLayer.answer(hiddenOutput);
	
	/*cout << "saida" << endl;
	for (int i = 0; i < this->numOutputNeurons; i++)
		cout << networkOutput[i] << " ";
	cout << endl;*/
	

	delete[] hiddenOutput;
	return networkOutput;
}

template < layerTemplate LayerHidden,neuronTemplate NeuronHidden, activationTemplate ActivationHidden,
			layerTemplate LayerOutput, neuronTemplate NeuronOutput, activationTemplate ActivationOutput,
			class Type >
Type** RBFNetwork<LayerHidden, NeuronHidden, ActivationHidden, LayerOutput, NeuronOutput, ActivationOutput, Type>::answer(Type** input, int inSize)
{
    Type** out;
    out = new Type*[inSize];
    for(int a = 0; a  < inSize; a++)
    {
        out[a] = this->answer(input[a]);
    }
    return out;	
}

template < layerTemplate LayerHidden,neuronTemplate NeuronHidden, activationTemplate ActivationHidden,
			layerTemplate LayerOutput, neuronTemplate NeuronOutput, activationTemplate ActivationOutput,
			class Type >
void RBFNetwork<LayerHidden, NeuronHidden, ActivationHidden, LayerOutput, NeuronOutput, ActivationOutput, Type>::save(const char* filename)
{
	cout << "Second Answer" << endl;
}


template < layerTemplate LayerHidden,neuronTemplate NeuronHidden, activationTemplate ActivationHidden,
			layerTemplate LayerOutput, neuronTemplate NeuronOutput, activationTemplate ActivationOutput,
			class Type >
void RBFNetwork<LayerHidden, NeuronHidden, ActivationHidden, LayerOutput, NeuronOutput, ActivationOutput, Type>::load(const char* filename)
{
	cout << "Second Answer" << endl;
}




#endif
