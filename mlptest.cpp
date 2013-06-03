#include "activationfunction/logisticactivationfunction.h"
#include "activationfunction/linearactivationfunction.h"
#include "mlp/mlplayer.h"
#include "mlp/mlpneuron.h"
#include "mlp/mlpnetwork.h"
#include "input/input.h"
#include "preprocessing/preprocessing.h"

#define AMOUNT_PIX 4
#define AMOUNT_CLASS 3

int main(){
	
	Input<double> Set; //cria o conjunto de entrada.
  PreProcessing<double> prePro;
	//Set.readData("data/zoo/zoo.data", true); //le o conjunto do arquivo.
	Set.readDataImageIbI("imagens/img.data", 1, true);
	Set.shuffle(); // embaralha.
	Set.setTestProportion(30);
	
  int max_att = 20;

	 //separa entre treino e teste.
	
	double **in = Set.getTrainingData(); //pega a matriz de entrada do treinamento.
	double **desired = Set.getTrainingIntendedClasses(); //pega a matriz de saida do treinamento.
	int numTrainingInputs = Set.getNumTrainingInputs(); //pega o numero de exemplos do treinamento.
	int numClasses = Set.getNumClasses(); //pega o numero de classes de saida possiveis da rede, deve ser igual ao numero de neuronios da camada de saida.
	int numAttributes = Set.getNumAttributes(); //pega o numero de atributos de entrada de um exemplo, deve ser igual ou numero de neuronios da camada de entrada.
	double** norTraining = prePro.normalize(in, numTrainingInputs, numAttributes);
  double** pcaTraining = prePro.pca(norTraining, numTrainingInputs, numAttributes, max_att);
	
  int numTestInputs = Set.getNumTestInputs(); //pega o numero de exemplos do teste.
	double **test = Set.getTestData(); //pega a matriz de entrada do teste.
	double **desTest = Set.getTestIntendedClasses(); //pega a matriz de saida do teste.
	double** norTest = prePro.normalize(test, numTestInputs, numAttributes, true);
  double** pcaTest = prePro.pca(norTest, numTestInputs, numAttributes, max_att, true);

	int camada[4] = {(max_att+numClasses)/2, numClasses}; //define a camada intermediaria e a camada de saida.
	MLPNetwork<MLPLayer, MLPNeuron, LogisticFunction, double> myNet(max_att , 2, camada); //cria a rede.
	myNet.setParameters(0.5, 0.00001, 0.5, 10000, 100); //seta os parametros da rede.
	
	//cout<<numClasses<<endl;
	//cout<<numTrainingInputs<<" "<<numClasses<<endl;
	//cout<<endl<<"-------------Inicio do treinamento--------------"<<endl<<endl;
	myNet.train(pcaTraining, desired, numTrainingInputs); //realiza o treinamento
	cout<<endl<<"-------------Fim do treinamento--------------"<<endl;
	
	cout<<endl<<"-------------Inicio do teste--------------"<<endl<<endl;

	double **out = myNet.answer(pcaTest, numTestInputs); //calcula a saida a partir da entrada teste pra calcular uma estatistica.
	//cout<<numClasses<<endl;
	//imprime os testes:
	for(int i = 0; i < numTestInputs; i++)
	{
		for(int j = 0; j < max_att; j++)
		{
			cout<<pcaTest[i][j]<<" ";
		}
		cout<<" -> ";
		
		cout<<Set.getClassName(out[i])<<" ";
		
		cout<<endl;
	}

	cout<<"Taxa de acerto:"<<Set.getRate(out, desTest, numTestInputs)<<endl; //calcula e imprime a taxa de acerto.
	cout<<endl<<"-------------Fim do teste--------------"<<endl;

  delete[] norTraining;
  delete[] pcaTraining;

  delete[] norTest;
  delete[] pcaTest;

	/*int ** matriz;
	double rate;
	int ** melhorMatriz = new int*[numClasses];
	double melhorRate = 0;
	int ** piorMatriz= new int*[numClasses];
	double piorRate = 1000;
	int qtTreina = 0;
	for(int c = 0; c < numClasses; c++)
    {
    	melhorMatriz[c] = new int[numClasses];
        piorMatriz[c] = new int[numClasses];
    }
	
	double* hitRate = new double[11];
	
	int numTreinamentos = 20;
	
	for(int tr = 0; tr < numTreinamentos; tr++)
	{
		myNet.randomizeWeight();
		cout<<"treinando mlp";
		Set.shuffle();
		cout<<endl<<"-------------Inicio do treinamento "<<tr<<"--------------"<<endl<<endl;
		qtTreina+=myNet.train(in, desired, numTrainingInputs);
		cout<<"treinou"<<endl;

		cout<<"testando"<<endl;
		double **out = myNet.answer(test, numTestInputs); 
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
	for(int tr = 0; tr < numTreinamentos; tr++)
     	media += hitRate[tr]/numTreinamentos;
	
	cout<<"Média: "<<media<<endl;
	
	//Calcula a variancia:
    double variance = 0.0;
    for(int tr = 0; tr < numTreinamentos; tr++)
    	variance += pow((hitRate[tr] - media), 2)/numTreinamentos;

	cout<<"Variância: "<<variance<<endl;
	
	cout<<"Média da quantidade de Épocas "<<qtTreina/numTreinamentos<<endl;*/
		
	return 0;
}
