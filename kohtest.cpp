#include <iostream>
#include <fstream>
#include <cstdlib>
#include <queue>

#include "kohonen/euclidiandistance.h"
#include "kohonen/kohonenneuron.h"
#include "kohonen/gaussianneighborhood.h"
#include "kohonen/kohonen.h"

#include "input/input.h"

using namespace std;
/////////////////////////////////////////////////////
int main()
{
	 srand(time(NULL));
	 
	 Input<double> Set;
	 Set.readData("data/iris/iris.data", true);
	 Set.shuffle();
	 Set.normalize();
	 Set.setTestProportion(50);
	 
	 int numTrainingInputs = Set.getNumTrainingInputs(); 
	 int numTestInputs = Set.getNumTestInputs();
	 int numAttributes = Set.getNumAttributes();
	 int numClasses = Set.getNumClasses();
	 
	 double** inTrain = Set.getTrainingData();
	 
	 double** inTest = Set.getTestData();
	 double** outTest = Set.getTestIntendedClasses();
	 
	 int numNeur = 5;
	 
	 Kohonen< KohonenNeuron, GaussianNeighborhoodFunction, EuclidianDistance, double> kohonen(numNeur, numAttributes);
	 kohonen.setParameters(0.3, 1);
	 
	 kohonen.train(inTrain, numTrainingInputs);
	 
	 /*int* out = new int[numTestInputs];
	 
	 for (int i = 0; i < numTestInputs; i++)
	{
		cout << "Input: " << i;
		out[i] = kohonen.cluster(inTest[i]);
		cout<<": "<<out[i]<<endl;
	}*/
	double **out;
	out = kohonen.cluster(inTest, numTestInputs);
	for (int i = 0; i < numTestInputs; i++)
	{
		for(int j = 0; j < numNeur; j++)
		{
			//cout<<" "<<out[i][j]<<" ";
		}
		//cout<<endl;
	}
	
	
	
	int **confMatrix = NULL;
	if(!confMatrix)
	{
		confMatrix = new int*[numNeur];
		for(int c = 0; c < numNeur; c++)
		{
			confMatrix[c] = new int[numClasses];
			for(int cc = 0; cc < numClasses; cc++)
			{
				confMatrix[c][cc] = 0;
			}
		}
	}
	//cout<<"-----------"<<endl;
	for(int c = 0; c < numNeur; c++)
		{
			for(int cc = 0; cc < numClasses; cc++)
			{
				confMatrix[c][cc] = 0;
			}
		}
	
    int rights = 0;
    for(int inst = 0; inst < numTestInputs; inst++)
    {
        int classObtaned = 0;
        int classDesired = 0;
        
        for(int cla = 0; cla < numNeur; cla++)
        {
            if(out[inst][classObtaned] < out[inst][cla])
            {
                classObtaned = cla;       
            }
        }
        
        for(int cla = 0; cla < numClasses; cla++)
        {
            if(outTest[inst][classDesired] < outTest[inst][cla])
            {
                classDesired = cla;       
            }
        }
        
        confMatrix[classObtaned][classDesired]++;
    }
    
    /*for(int cc = 0; cc < numClasses; cc++)
	{
		cout<<cc<<" -> "<<this->classes[cc]<<endl;
	}
	
    for(int cc = 0; cc < numClasses; cc++)
	{
		cout<<"\t"<<cc;
	}
    cout<<endl;
    */
    for(int c = 0; c < numNeur; c++)
	{
		//cout<<c<<"\t";
		for(int cc = 0; cc < numClasses; cc++)
		{
			cout<<confMatrix[c][cc]<<"\t";
		}
		cout<<endl;
	}
    //cout<<rights<<" de "<<amount<<" "<<endl;
	return 0;
	 
	
}
