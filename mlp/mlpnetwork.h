#ifndef __MLPNETWORK_H__
#define __MLPNETWORK_H__

//#include "normalize.cpp"
#include "network.h"
#include "templateDefinitions.h"
#include "fstream"

#define activationTemplate template <class> class


/**
 * This class is responsable for the control of the Neural Network.
It will contain the data structs responsable for the layers.
 */

template < layerTemplate LayerT, 
			neuronTemplate NeuronT, 
			activationTemplate ActivationT,
			class Type >
			
class MLPNetwork : public Network<Type>
{
public:

	/**
	 * The default constructor sets the network as non-initialized.
	 */
	MLPNetwork();
	
	/**
	 * This other constructor initialize the network according to the following parameters:
	 * @param numWeightsInput number of neurons in the input layer 
	 * @param numLayers number of layers in the network;
	 * @param numNeurons array with the number or neurons in each layer.
	 */
	MLPNetwork(int numWeightsInput, int numLayers, int *numNeurons);
	
	/**
	 * The default destructor deletes the layer array and the  numNeurons array
	 */
	~MLPNetwork();
	
	/**
	 * This method also can initialize the network according to the following parameters:
	 * @param numWeightsInput number of neurons in the input layer 
	 * @param numLayers number of layers in the network;
	 * @param numNeurons array with the number or neurons in each layer.
	 */
	bool initialize(int numWeightsInput, int numLayers, int *numNeurons);
	
	/**
	 * Reinicializa os pesos aleatoriamente.
	 */
	void randomizeWeight();
	
	
	/**
	 * This method sets the parameters that can be set after the initialation of the network and alse reset anytime
	 * @param learningRate the learning rate used in the network 
	 * @param minimumError representes the diference between the medium square erro of two successive epochs and is used as stopping condition
	 * @param momentumRate used in the learning process, may be zero if you don't want Momentum
	 * @param maximumEpochs maximum number of epochs it can train
	 * @param betaActivationFuction parameter Beta used in the activation function, may be disconsidered
	 */
	void setParameters(double learningRate, double minimumError, double momentumRate, int maximumEpochs, Type betaActivationFuction = 1);
	
	/**
	 * This method will execute the network.
	 * @param in vector with the data input for the network
	 */
	Type* answer(Type* in);
	
	/**
	 * This method also will execute the network, but for a set of inputs.
	 * @param in matrix with the data input for the network
	 * @param inSize number of inputs that will be executed
	 */
	Type** answer(Type** in, int inSize);
	
	/**
	 * This method executes the network training.
	 * @param inputSet matrix with the data input for the network
	 * @param intended vector with the data intented output for the network
	 * @param numInputs of instances used on the training
	 */
	int train(Type** inputSet, Type** intended, int numInputs);
	
	/**
	 * Saves the network into a file, whit all the parameters and weight of all the neurons.
	 * @param filename name of the file
	 */
	void saveNetwork(string filename);
	
	/**
	 * Recover the network from file. The network must be non-initialized.
	 * @param filename name of the file
	 */
	bool loadNetwork(string filename);
	
protected:
	LayerT < NeuronT, ActivationT, Type> *layer;
	int numWeightsInput;
	int numLayers;
	int *numNeurons;
	double learningRate;
	double momentumRate;
	double minimumError;
	Type betaActivationFuction;
	int maximumEpochs;
	bool initialized;
	
	
};

template < layerTemplate LayerT, neuronTemplate NeuronT, 
           activationTemplate ActivationT, class Type >
MLPNetwork<LayerT, NeuronT, ActivationT,Type>::MLPNetwork(int numWeightsInput, int numLayers, int *numNeurons)
{
	initialized = false;
	if ((numWeightsInput != 0) && (numLayers != 0) && (numNeurons[0] != 0))
	{
		this->initialize(numWeightsInput, numLayers, numNeurons);
	}
}

template < layerTemplate LayerT, neuronTemplate NeuronT, 
           activationTemplate ActivationT, class Type >
MLPNetwork<LayerT, NeuronT, ActivationT,Type>::MLPNetwork()
{
	initialized = false;
}

template < layerTemplate LayerT, neuronTemplate NeuronT, 
           activationTemplate ActivationT, class Type >
MLPNetwork<LayerT, NeuronT, ActivationT,Type>::~MLPNetwork()
{
	if(numLayers != 0)
	{ 
	    delete[] layer;
	    delete[] numNeurons;
	}   
}

template < layerTemplate LayerT, neuronTemplate NeuronT, 
           activationTemplate ActivationT, class Type >
bool MLPNetwork<LayerT, NeuronT, ActivationT,Type>::initialize(int numWeightsInput, int numLayers, int *numNeurons)
{
	if(initialized) return false;
	srand(time(NULL));
	this->numWeightsInput = numWeightsInput;
	this->numLayers = numLayers;
	
	this->numNeurons = new int[numLayers];

	for(int i = 0; i < numLayers; i++)
	{
		this->numNeurons[i] = numNeurons[i];	    
	}

	this->layer = new LayerT < NeuronT, ActivationT, Type> [numLayers];
	
	/* Here you will have to initialize all the layer, whit something like this:*/

	this->layer[0].initialize(numNeurons[0], numWeightsInput);
	for(int i = 1; i < numLayers; i++){
		this->layer[i].initialize(numNeurons[i], numNeurons[i-1]);
	}
	initialized = true;
	return true;
}

template < layerTemplate LayerT, neuronTemplate NeuronT, 
           activationTemplate ActivationT, class Type >
void MLPNetwork<LayerT, NeuronT, ActivationT,Type>::randomizeWeight()
{
	if(initialized)
	{
		for(int i = 1; i < numLayers; i++){
			this->layer[i].randomizeWeight();
		}	
	}
}

template < layerTemplate LayerT, neuronTemplate NeuronT, 
           activationTemplate ActivationT, class Type >
void MLPNetwork<LayerT, NeuronT, ActivationT,Type>::setParameters(double learningRate, double minimumError, double momentumRate,int maximumEpochs, Type betaActivationFuction)
{	
	this->minimumError = minimumError;
	this->maximumEpochs = maximumEpochs;
	this->learningRate = learningRate;
	this->momentumRate = momentumRate;
	this->betaActivationFuction = betaActivationFuction;
	for(int l = 0; l < numLayers; l++)
	{
	    this->layer[l].setParameters(this->learningRate, this->momentumRate);
	}
}

template < layerTemplate LayerT, neuronTemplate NeuronT, 
           activationTemplate ActivationT, class Type >
Type* MLPNetwork<LayerT, NeuronT, ActivationT,Type>::answer(Type* in)
{
    Type* out;
    Type* previousOut;
    out = layer[0].answer(in);
    for(int l = 1; l  < numLayers; l++)
    {
        previousOut = out;
        out = layer[l].answer(previousOut);
        delete[] previousOut;
    }
    return out;
}

template < layerTemplate LayerT, neuronTemplate NeuronT, 
           activationTemplate ActivationT, class Type >
Type** MLPNetwork<LayerT, NeuronT, ActivationT,Type>::answer(Type** in, int inSize)
{
    Type** out;
    out = new Type*[inSize];
    for(int a = 0; a  < inSize; a++)
    {
        out[a] = this->answer(in[a]);
    }
    return out;
}

template < layerTemplate LayerT, neuronTemplate NeuronT, 
           activationTemplate ActivationT, class Type >
int MLPNetwork<LayerT, NeuronT, ActivationT,Type>::train(Type** inputSet, Type** intended, int numInputs)
{
    Type** layersOut = new Type*[this->numLayers+1];
    Type* delta;
    Type* previousDelta;
    Type erro = 0;
    Type previousErro;
    for(int epoch = 0; epoch < this->maximumEpochs; epoch++ )
    {
    	if(epoch%100 == 0)cout<<"epoch: "<<epoch<<"\n";
        previousErro = erro;
        erro = 0;
        for(int currentInput = 0; currentInput < numInputs; currentInput++)
        {
            layersOut[0] = new Type[numWeightsInput];
            for(int i = 0; i < numWeightsInput; i++)
            {
                layersOut[0][i] = inputSet[currentInput][i];
               // cout<<layersOut[0][i]<<" ";
            }
            //cout<<" -> ";
            for(int l = 0; l < this->numLayers; l++)
            {
                layersOut[l+1] = this->layer[l].answer(layersOut[l]);        
            }
            for(int n = 0; n < layer[this->numLayers-1].getNumNeurons(); n++)
            {
               // cout<<" "<<intended[currentInput][n]<<"/"<<layersOut[this->numLayers][n]<<" ";
            }
            //cout<<endl;
            
            //here begins the backpropagation
            int currentLayer = this->numLayers - 1;
            //to the last layer:
            //cout<<currentLayer<<endl;
            previousDelta = new Type[this->layer[currentLayer].getNumNeurons()]; 
            for(int n = 0; n < this->layer[currentLayer].getNumNeurons(); n++)
            {learningRate;
                 erro+=pow(intended[currentInput][n]-layersOut[currentLayer+1][n],2);
                 previousDelta[n] = intended[currentInput][n]-layersOut[currentLayer+1][n];
            }
            delta = this->layer[currentLayer].getDelta(previousDelta, layersOut[currentLayer]);
            
            //aqui use o delta pra corrigir os pesos da currentlayer
            this->layer[currentLayer].fitWeights(delta, layersOut[currentLayer]);
            
            delete[] previousDelta;
            currentLayer--;
            //cout<<currentLayer;
            while(currentLayer >= 0)
            {
                previousDelta = new Type[this->layer[currentLayer].getNumNeurons()]; 
                for(int n = 0; n < this->layer[currentLayer].getNumNeurons(); n++)
                {
                     previousDelta[n] = 0;
                     for(int w = 0; w < this->layer[currentLayer+1].getNumNeurons(); w++)
                     {
                         previousDelta[n] += delta[w]*this->layer[currentLayer+1].getWeight(w,n);
                     }
                }
                delete[] delta;
                delta = this->layer[currentLayer].getDelta(previousDelta, layersOut[currentLayer]);
                
                //aqui use o delta pra corrigir os pessos da currentLayer
                this->layer[currentLayer].fitWeights(delta, layersOut[currentLayer]);
                
                delete[] previousDelta;
                currentLayer--;
            }
            for(int l = 0; l <= this->numLayers; l++)
            {
                delete[] layersOut[l];
            }
            delete[] delta;
        }
        erro/=numInputs;
        if(abs(erro-previousErro) <= this->minimumError)
        {
            cout<<"A rede apresentou erro: "<<abs(erro-previousErro)<<" apos: "<< epoch<<" interações."<<endl;
            delete[] layersOut;
            return epoch;
        }
    }
    delete[] layersOut;
    cout<<"A rede estourou o limite de épocas apresentando um erro: "<<erro<<". "<<endl;
    return this->maximumEpochs;
}

template < layerTemplate LayerT, neuronTemplate NeuronT, 
           activationTemplate ActivationT, class Type >
void MLPNetwork<LayerT, NeuronT, ActivationT,Type>::saveNetwork(string filename)
{
	// tem que salvar learningRate, double minimumError, momentumRate, maximumEpochs, numLayers, numNeurons de cada layer, pesos de cada neuronios de cada layer. 
	fstream File(filename.c_str(), ios::app);
	File.close();
	File.open(filename.c_str());
	File<<"MLPLucyCV ";
	File<<this->learningRate<<" ";
	File<<this->minimumError<<" ";
	File<<this->momentumRate<<" ";
	File<<this->maximumEpochs<<" ";
	File<<this->numLayers<<" ";
	File<<this->numWeightsInput<<" ";
	File<<endl;
	
	File<<numNeurons[0]<<" ";
	for(int nn = 0; nn < numNeurons[0]; nn++)
	{
		for(int nw = 0; nw < numWeightsInput; nw++)
		{
			File<<this->layer[0].getWeight(nn,nw)<<" ";
		}
	}
	File<<endl;
	
	
	for(int nl = 1; nl < this->numLayers; nl++)
	{
		File<<numNeurons[nl]<<" ";
		for(int nn = 0; nn < numNeurons[nl]; nn++)
		{
			for(int nw = 0; nw < numNeurons[nl-1]; nw++)
			{
				File<<this->layer[nl].getWeight(nn,nw)<<" ";
			}
		}
		File<<endl;
	}
	
	File.close();
}


template < layerTemplate LayerT, neuronTemplate NeuronT, 
           activationTemplate ActivationT, class Type >
bool MLPNetwork<LayerT, NeuronT, ActivationT,Type>::loadNetwork(string filename)
{
	// tem que recuperar learningRate, double minimumError, momentumRate, maximumEpochs, numLayers, numNeurons de cada layer, pesos de cada neuronios de cada layer.
	fstream File;
	File.open(filename.c_str());
	if(!File.is_open()) return false;
	string nome;
	File>>nome;
	if(nome == "MLPLucyCV")
	{
		File>>this->learningRate;
		cout<<"this->learningRate "<<this->learningRate<<endl;
		File>>this->minimumError;
		cout<<"this->minimumError "<<this->minimumError<<endl;
		File>>this->momentumRate;
		cout<<"this->momentumRate "<<this->momentumRate<<endl;
		File>>this->maximumEpochs;
		cout<<"this->maximumEpochs "<<this->maximumEpochs<<endl;
	
		if(initialized) return false;
		File>>this->numLayers;
		cout<<"this->numLayers "<<this->numLayers<<endl;
		File>>this->numWeightsInput;
		cout<<"this->numWeightsInput "<<this->numWeightsInput<<endl;
		this->layer = new LayerT < NeuronT, ActivationT, Type> [this->numLayers];
		this->numNeurons = new int[numLayers];
	
		File>>numNeurons[0];
	
		this->layer[0].initialize(numNeurons[0], numWeightsInput);
	
		for(int nn = 0; nn < numNeurons[0]; nn++)
		{
			for(int nw = 0; nw < numWeightsInput; nw++)
			{
				Type newWeight;
				File>>newWeight;
				this->layer[0].setWeight(nn,nw, newWeight);
			}
		}
		
		for(int nl = 1; nl < this->numLayers; nl++)
		{
			File>>numNeurons[nl];
			this->layer[nl].initialize(numNeurons[nl], numNeurons[nl-1]);
			for(int nn = 0; nn < numNeurons[nl]; nn++)
			{
				for(int nw = 0; nw < numNeurons[nl-1]; nw++)
				{
					Type newWeight;
					File>>newWeight;
					this->layer[nl].setWeight(nn,nw, newWeight);
				}
			}
		}
		
		for(int l = 0; l < this->numLayers; l++)
		{
	   		this->layer[l].setParameters(this->learningRate, this->momentumRate);
		}
	}
	File.close();
	return true;
}
#endif
