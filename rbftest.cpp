#include <iostream>
#include <fstream>
#include <cstdlib>


#include "activationfunction/gaussianactivationfunction.h"
#include "activationfunction/linearactivationfunction.h"
#include "rbf/rbfhiddenneuron.h"
#include "rbf/rbfoutputneuron.h"
#include "rbf/rbfhiddenlayer.h"
#include "rbf/rbfoutputlayer.h"
#include "rbf/rbfnetwork.h"
#include "input/input.h"

//#include "../Output/output.cpp"

using namespace std;

int main()
{
	srand(time(NULL));

	////////////////////////////////////////////// começa a cuidar da matriz de entrada de dados
		
	Input<double> Set; //cria o conjunto de entrada.
	Set.readData("data/zoo/zoo.data", true); //le o conjunto do arquivo.
	Set.shuffle(); // embaralha.
	Set.normalize();
	Set.setTestProportion(30);
	
	double **inTrain = Set.getTrainingData(); //pega a matriz de entrada do treinamento.
	double **desTrain = Set.getTrainingIntendedClasses(); //pega a matriz de saida do treinamento.
	
	double **inTest = Set.getTestData(); //pega a matriz de entrada do treinamento.
	double **desTest = Set.getTestIntendedClasses(); //pega a matriz de saida do treinamento.
	
	int numTrainingInputs = Set.getNumTrainingInputs(); //pega o numero de exemplos do treinamento.
	int numTestInputs = Set.getNumTestInputs(); //pega o numero de exemplos do teste.
	int numClasses = Set.getNumClasses(); //pega o numero de classes de saida possiveis da rede, deve ser igual ao numero de neuronios da camada de saida.
	int numAttributes = Set.getNumAttributes(); //pega o numero de atributos de entrada de um exemplo, deve ser igual ou numero de neuronios da camada de entrada.
	
    ////////////////////////////////////////////////termina de cuidar da matriz de entrada de dados
	
	////////////////////////////////////////////////começa a criação e configuração da rede RBF	
	
	RBFNetwork <RBFHiddenLayer, RBFHiddenNeuron, GaussianActivationFunction, RBFOutputLayer, RBFOutputNeuron, LinearActivationFunction, double > rbf1(numAttributes,numClasses , numClasses);

	rbf1.setParameters(0.1, 0.000001, 10000, 0.0);
	
	/*////////////////////////////////////////////////termina a criação e configuração da rede RBF
	
	////////////////////////////////////////////////começa o treinamento da rede RBF
	
	cout<<"vai treinar"<<endl;

	rbf1.train(inTrain, desTrain, numTrainingInputs);
	
	cout << endl << "TREINOU TREINOU COM VÁRIAS INSTÂNCIAS DE TESTE: " << numTrainingInputs << endl << endl <<endl;
	
	cout<<"treinou"<<endl;
	
	////////////////////////////////////////////////termina o treinamento da rede RBF
	
	////////////////////////////////////////////////começa o teste da rede RBF
	
	cout<<"Vai Executar"<<endl;
	
	double **obtained;
	obtained = rbf1.answer(inTest, numTestInputs);
	
	cout<<"Taxa de acerto:"<<Set.getRate(obtained, desTest, numTestInputs)<<endl;
	
	////////////////////////////////////////////////termina o teste da rede RBF*/
	
	int ** matriz;
	double rate;
	int ** melhorMatriz = new int*[numClasses];
	double melhorRate = 0;
	int ** piorMatriz= new int*[numClasses];
	double piorRate = 1000;
	
	for(int c = 0; c < numClasses; c++)
    {
    	melhorMatriz[c] = new int[numClasses];
        piorMatriz[c] = new int[numClasses];
    }
	
	double* hitRate = new double[11];

	for(int tr = 0; tr < 10; tr++)
	{
		rbf1.randomizeWeight();
		cout<<"treinando mlp";
		Set.shuffle();
		cout<<endl<<"-------------Inicio do treinamento "<<tr<<"--------------"<<endl<<endl;
		rbf1.train(inTrain, desTrain, numTrainingInputs);
		cout<<"treinou"<<endl;

		cout<<"testando"<<endl;
		double **out = rbf1.answer(inTest, numTestInputs);
		rate = Set.getRate(out, desTest, numTestInputs);
		matriz = Set.getConfMatrix();

		hitRate[tr] = rate;

		if(rate < piorRate){
			piorRate = rate;
			for(int c = 0; c < numClasses; c++)
			{
				 for(int cc = 0; cc < numClasses; cc++)
				 {
				     piorMatriz[c][cc] = matriz[c][cc];
				 }
			}
		}

		if(rate > melhorRate){
			melhorRate = rate;
			for(int c = 0; c < numClasses; c++)
			{
				 for(int cc = 0; cc < numClasses; cc++)
				 {
				     melhorMatriz[c][cc] = matriz[c][cc];
				 }
			}
		}

		cout<<"testou mlp"<<endl;              
    }
	
	cout<<endl<<"----------------------Resultados----------------------"<<endl;
	
	cout<<"Melhor: "<<melhorRate<<endl;
    for(int cc = 0; cc < numClasses; cc++)
    {
		//cout<<cc<<" -> "<<this->classes[cc]<<endl;
		cout<<cc<<" -> "<<Set.getClassValue(cc)<<";"<<endl;
    }

    for(int cc = 0; cc < numClasses; cc++)
    {
       cout<<"\t"<<cc;
    }
    cout<<endl;
    for(int c = 0; c < numClasses; c++)
    {
   		cout<<c<<"\t";
        for(int cc = 0; cc < numClasses; cc++)
        {
        	cout<<melhorMatriz[c][cc]<<"\t";
        }
        cout<<endl;
     }

     cout<<endl;
	 cout<<"Pior: "<<piorRate<<endl;

     for(int cc = 0; cc < numClasses; cc++)
     {
     	cout<<cc<<" -> "<<Set.getClassValue(cc)<<";"<<endl;
     }

     for(int cc = 0; cc < numClasses; cc++)
     {
     	cout<<"\t"<<cc;
     }
     cout<<endl;
	
	for(int c = 0; c < numClasses; c++)
	{
     	cout<<c<<"\t";
        for(int cc = 0; cc < numClasses; cc++)
        {
        	cout<<piorMatriz[c][cc]<<"\t";
        }
        cout<<endl;
	}

	cout<<endl<<endl;
		    
	//Calcula a media:
	double media = 0.0;
	for(int tr = 0; tr < 10; tr++)
     	media += hitRate[tr]/10;
	
	cout<<"Média: "<<media<<endl;
	
	//Calcula a variancia:
    double variance = 0.0;
    for(int tr = 0; tr < 10; tr++)
    	variance += pow((hitRate[tr] - media), 2)/10;

	cout<<"Variância: "<<variance<<endl;
	
	return 0;
}
