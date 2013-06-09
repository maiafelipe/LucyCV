#ifndef __INPUT_H__
#define __INPUT_H__

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace std;

using namespace cv;

/**
* A class that will be responsible for receiving the data from a text file and saving it in a matrix;
* 
*/

template <class Type>
class Input
{
public:
    /**
	 * The default constructor does nothing.
	 */
    Input();
    /**
	 * The default destructor deallocates all the matrices  
	 */
    ~Input();
	/**
	 * This method will fill the matrices with the values of the file.
	 * @param filename the adress of the file; 
	 * @param supervised inform if the training is supervised.
	 */    
    bool readData(const char*, bool);

    void readDataImageIbI(const char* filename, int numColors, bool supervised);
    
    void readDataImagePbP(const char* filename, int numColors, bool supervised);
    

    /**
     * Return all the input data  for the network.  
     */
    void clear();

    /**
	 * Return all the input data  for the network.  
	 */
    Type** getData();
    /**
	 * Return all the intented output data  for the network.  
	 */
    Type** getIntendedClasses();
    
    /**
	 * Splits the data in training and test.
	 */
    void setTestProportion(double p);
    
    /**
	 * Return the training input data for the network.  
	 */
    Type** getTrainingData();
    /**
	 * Return the training intented output data  for the network.  
	 */
    Type** getTrainingIntendedClasses();
    /**
	 * Return the test input data  for the network.  
	 */
    Type** getTestData();
    /**
	 * Return the test intented output data  for the network.  
	 */
    Type** getTestIntendedClasses();
    
    /**
	 * This method will shuffle the set.
	 */ 
    void shuffle();
    /**
	 * This method will normalize the set.
	 */
    void normalize();
    
    /**
	 * This method will return the amount of all inputs.
	 */
    int getNumInputs();
    /**
	 * This method will return the amount of training inputs.
	 */
    int getNumTrainingInputs();
    /**
	 * This method will return the amount of test inputs.
	 */
    int getNumTestInputs();
    /**
	 * This method will return the amount of attributes of a input. Must be compatible with the input layer of the network.
	 */
    int getNumAttributes();
    /**
	 * This method will return the number of classes of data present in the set. Normaly, must be compatible with the output layer of the network.
	 */
    int getNumClasses();
    
    /**
	 * Return the number of instances in a file.
	 * @param filename the name of the file. 
	 */
    int countInput(const char* filename);
    
    /**
	 * Return the name of a class based on a patter;
	 * @param value the vector with the patter. 
	 */    
    string getClassName(Type* value);
    
    /**
	 * Return the name of a class based on the index of the class;
	 * @param i value of the index. 
	 */
    string getClassValue(int i);
    
    /**
	 * Return the accuracy rate of a test. Also create a confusion matrix.
	 * @param obtaned matrix with the output of the network;
     * @param desired matrix with the desired output;
     * @param amount amount of samples that will be compared.
	 */
    Type  getRate(Type** obtaned,Type** desired, int amount);
    
    /**
	 * This method will return the confusion matrix created in getRate().
	 */
    int** getConfMatrix();
        
private:
    ifstream inputData;
    
    Type** allData;
    Type** allClasses;
    
    Type** testData;
    Type** testClasses;
  
    Type** trainingData;
    Type** trainingClasses;
    
    int numAttributes;
    int numClasses;
    int numAllInputs;
    int numTrainingInputs;
    int numTestInputs;
    
    vector<string> classes;
    
    int ** confMatrix;
};

template <class Type>
Input<Type>::Input()
{
    this->numAttributes = 0;
    this->numClasses = 0;
    this->numAllInputs = 0;
    confMatrix = NULL;
    
    allData = NULL;
    allClasses = NULL;
    
    testData = NULL;
    testClasses = NULL;
  
    trainingData = NULL;
    trainingClasses = NULL;
}


template <class Type>
Input<Type>::~Input()
{
    if (this->numAllInputs != 0)
    {
        for (int i = 0; i < this->numAllInputs; i++)
        {
            delete[] this->allData[i];
        }
        delete[] this->allData;
    }
    
    if (this->numClasses != 0)
    {
        for (int i = 0; i < this->numAllInputs; i++)
        {
            delete[] this->allClasses[i];
        }
        delete[] this->allClasses;
    }

    if (this->numTrainingInputs != 0)
    {
        for (int i = 0; i < this->numTrainingInputs; i++)
        {
            delete[] this->trainingData[i];
        }
        delete[] this->trainingData;
    }
    
    if (this->numClasses != 0)
    {
        for (int i = 0; i < this->numTrainingInputs; i++)
        {
            delete[] this->trainingClasses[i];
        }
        delete[] this->trainingClasses;
    }

    if (this->numTestInputs != 0)
    {
        for (int i = 0; i < this->numTestInputs; i++)
        {
            delete[] this->testData[i];
        }
        delete[] this->testData;
    }
    
    if (this->numClasses != 0)
    {
        for (int i = 0; i < this->numTestInputs; i++)
        {
            delete[] this->testClasses[i];
        }
        delete[] this->testClasses;
    }

	if(confMatrix)
	{	
		for(int c = 0; c < numClasses; c++)
		{
			delete[] confMatrix[c];
		}
		delete[] confMatrix;
	}
}


template <class Type>
bool Input<Type>::readData(const char* filename, bool supervised)
{
    int n = this->countInput(filename);
    bool adaptar = false;
    int numOut = 0;
    ifstream dataFile(filename);
    string word;
    dataFile >> word;
    numAttributes = 0;
    
    while(word != "->"){
        if(word == "d" || word == "i" || word == "f" || word == "s")
        {
            numAttributes++;
            dataFile >> word;
        }
        else return false;
    }
    dataFile >> word;
    while(word != "begin"){
        
        if(word == "s") adaptar = true;
        if(word == "d" || word == "i" || word == "f" || word == "s")
        {
            numOut++;
            dataFile >> word;
        }
        else return false;
    }

    numAllInputs = (n - 2);
    allData = new Type* [numAllInputs];
    Type temporaryClasses[numAllInputs];
    for (int i = 0; i < numAllInputs; i++)
        allData[i] = new Type[numAttributes];
    
    if(supervised && !adaptar){
        this->numClasses = numOut;
        allClasses = new Type*[numAllInputs];
        for (int i = 0; i < numAllInputs; i++)
        {
            allClasses[i] = new Type [numClasses];
        }
    }
    
    for (int i = 0; i < numAllInputs; i++)
    {
        for (int j = 0; j < numAttributes; j++)
            dataFile >> allData[i][j];
        if (supervised && adaptar)
        {
            string temporaryClass;
            
            dataFile >> word;
            while(word == "->"){
                dataFile >> word;
            } 
            temporaryClass = word;
            
            bool inClass = false;
            int indexInClass;
            int j;
            for (j = 0; j < classes.size(); j++)
                if (classes[j] == temporaryClass)
                {
                    inClass = true;
                    indexInClass = j;
                    break;
                }
            if (inClass == false)
            {
                classes.push_back(temporaryClass);
            }
            temporaryClasses[i] = j;
        }
        else if(supervised)
        {
            for (int j = 0; j < numClasses; j++){
                
                dataFile >> word;
                while(word == "->" )
                    dataFile >> word;
                
                allClasses[i][j] = (Type)atof(word.c_str());
            }
        }
    }
    if (supervised && adaptar)
    {
        this->numClasses = classes.size();
        allClasses = new Type*[numAllInputs];
        for (int i = 0; i < numAllInputs; i++)
        {
            allClasses[i] = new Type [numClasses];
        }
        for (int i = 0; i < numAllInputs; i++)
        {
            for (int j = 0; j < numClasses; j++)
                allClasses[i][j] = 0;
            int classIndex = temporaryClasses[i];
            allClasses[i][classIndex] = 1;
        }
    }
    return true;
}

template <class Type>
void Input<Type>::readDataImageIbI(const char* filename, int numColors, bool supervised)
{
    int n = this->countInput(filename);
    bool adaptar = false;
    int numOut = 0;
    ifstream dataFile(filename);
    string word;
    dataFile >> word;
    int numImgs = 0;
    while(word != "->"){
        numImgs++;
        dataFile >> word;
    }
    dataFile >> word;
    while(word != "begin"){
        if(word == "s") adaptar = true;
        numOut++;
        dataFile >> word;
    }
    this->numAllInputs = n-2;
    Type** allDataLocal = new Type* [numAllInputs];
    Type temporaryClasses[numAllInputs];
    if(supervised && !adaptar){
        this->numClasses = numOut;
        allClasses = new Type*[numAllInputs];
        for (int i = 0; i < numAllInputs; i++)
        {
            allClasses[i] = new Type [numClasses];
        }
    }
    //cout<<numAllInputs<<endl;
    
    
    
    for (int i = 0; i < numAllInputs; i++)
    {
	//cout<<i<<endl;
        dataFile >> word;
	//cout<<word<<endl;
        IplImage *full = cvLoadImage(word.c_str(),1);
        int perc = 100;
        IplImage *img = cvCreateImage( cvSize((int)((full->width*perc)/100) , (int)((full->height*perc)/100) ), full->depth, full->nChannels );
        cvResize(full, img);
        numAttributes = img->height * img->width * numColors;
	//cout<<numAttributes<<endl;
        int j = 0;
        allDataLocal[i] = new Type[numAttributes];
        for(int h = 0; h < img->height; h++){
            for(int w = 0; w < img->width; w++){
                CvScalar pont = cvGet2D(img,h,w);
                for(int c = 0; c < numColors; c++){
                    allDataLocal[i][j] = pont.val[c];
                    //cout<<" "<<pont.val[c]<<"-";
                    //cout<<allDataLocal[i][j]<<" ";
                    j++; 
                }
            }
        }
        //cout<<endl;
        if (supervised && adaptar)
        {
        	dataFile >> word;
        	while(word == "->" )
                    dataFile >> word;
            string temporaryClass;
            temporaryClass = word;
        
            bool inClass = false;
            int indexInClass;
            int j;
            for (j = 0; j < classes.size(); j++)
                if (classes[j] == temporaryClass)
                {
                    inClass = true;
                    indexInClass = j;
                    break;
                }
            if (inClass == false)
            {
                classes.push_back(temporaryClass);
            }
            temporaryClasses[i] = j;
        }
        else if(supervised)
        {
        	dataFile >> word;
        	while(word == "->" )
                    dataFile >> word;
            for (int j = 0; j < numClasses; j++){
                allClasses[i][j] = (Type)atof(word.c_str());
                if(j < numClasses - 1)
                	dataFile >> word;
            }
        }
	//cout<<i<<endl;
    }	   
    //cout<<"out"<<endl;
    if (supervised && adaptar)
    {
	//cout<<"S"<<endl;
        this->numClasses = classes.size();
        allClasses = new Type*[numAllInputs];
        for (int i = 0; i < numAllInputs; i++)
        {
	    //cout<<i<<endl;
            allClasses[i] = new Type [numClasses];
        }
        for (int i = 0; i < numAllInputs; i++)
        {
            for (int j = 0; j < numClasses; j++)
                allClasses[i][j] = 0;
            int classIndex = temporaryClasses[i];
            allClasses[i][classIndex] = 1;
        }
    }
    
   /* Mat allImg(numAllInputs, numAttributes, CV_64F);
    
    for(int i = 0; i < numAllInputs; i++)
    {
        for(int j = 0; j < numAttributes; j++)
        {
            allImg.col(j).row(i) = allDataLocal[i][j]; 
        }
    }
    

    numAttributes = numMaxAtt;
    PCA pca(allImg, Mat(), CV_PCA_DATA_AS_ROW, numAttributes);  
 
    for (int i = 0; i < this->numAllInputs; i++)
    {
        delete[] allDataLocal[i];
    }
    
    Mat projecao;
    pca.project(allImg, projecao);
    allData = new Type* [numAllInputs];
    for(int i = 0; i < numAllInputs; i++)
    {
        allData[i] = new Type[numAttributes];
        allDataLocal[i] = projecao.ptr<double>(i); 
    }*/
    allData = new Type* [numAllInputs];
    for(int i = 0; i < numAllInputs; i++)
    {
        allData[i] = new Type[numAttributes];
        
    }
    for(int i = 0; i < numAllInputs; i++)
    {
        for(int j = 0; j < numAttributes; j++)
        {
            allData[i][j] = allDataLocal[i][j]; 
        }
    }
    
    delete[] allDataLocal;
    
}


template <class Type>
void Input<Type>::readDataImagePbP(const char* filename, int numColors, bool supervised)
{
    int n = this->countInput(filename);
    bool adaptar = false;
    int numOut = 0;
    ifstream dataFile(filename);
    string word;
    dataFile >> word;
    int numImgs = 0;
    while(word != "->"){
        numImgs++;
        dataFile >> word;
    }
    dataFile >> word;
    while(word != "begin"){
        if(word == "s") adaptar = true;
        numOut++;
        dataFile >> word;
    }
    
    this->numAllInputs = n-2;
    cout<<numAllInputs<<endl;
    dataFile >> word;
    IplImage* img = cvLoadImage(word.c_str(),1);
    int numPixels = img->height * img->width;
    allData = new Type* [numAllInputs*numPixels];
    numAttributes = numImgs * numColors;
    for(int i = 0; i < numAllInputs*numPixels; i++)
        allData[i] = new Type[numAttributes];
    Type temporaryClasses[numAllInputs*numPixels];
    if(supervised && !adaptar){
        this->numClasses = numOut;
        allClasses = new Type*[numAllInputs*numPixels];
        for (int i = 0; i < numAllInputs*numPixels; i++)
        {
            allClasses[i] = new Type [numClasses];
        }
    }
    for (int i = 0; i < numAllInputs; i++)
    {
        for(int j = 0; j < numImgs; j++){
            int k = 0;
            for(int h = 0; h < img->height; h++){
		        for(int w = 0; w < img->width; w++){
		            CvScalar pont = cvGet2D(img,h,w);
		            for(int c = 0; c < numColors; c++){
		                allData[i*numPixels + k][j*numColors+c] = pont.val[c]; 
		            }
		            k++; 
		        }
            }
            dataFile >> word;
            img = cvLoadImage(word.c_str(),1);
        }
        if (supervised && adaptar)
        {
            dataFile >> word;
			ifstream classFile(word.c_str());
			string temporaryClass;
			
			for(int j = 0; j < numPixels; j++){
				classFile >> temporaryClass;
		        bool inClass = false;
		        int indexInClass;
		        int k;
		        for (k = 0; k < classes.size(); k++)
		            if (classes[k] == temporaryClass)
		            {
		                inClass = true;
		                indexInClass = k;
		                break;
		            }
		        if (inClass == false)
		        {
		            classes.push_back(temporaryClass);
		        }
		        temporaryClasses[i*numPixels + j] = k;
            }
        }
        else if(supervised)
        {
        	dataFile >> word;
        	while(word == "->" )
                 dataFile >> word;
            cout<<word<<endl;
			ifstream classFile(word.c_str());
        	for(int j = 0; j < numPixels; j++){
            	for (int k = 0; k < numClasses; k++){
            		double felipe;
            		classFile >> felipe;
            		allClasses[i*numPixels + j][k] = felipe;
            	}
        	}
        }        
    }	   
    if (supervised && adaptar)
    {
        this->numClasses = classes.size();
        allClasses = new Type*[numAllInputs*numPixels];
        for (int i = 0; i < numAllInputs*numPixels; i++)
        {
            allClasses[i] = new Type [numClasses];
        }
        for (int i = 0; i < numAllInputs*numPixels; i++)
        {
            for (int j = 0; j < numClasses; j++)
                allClasses[i][j] = 0;
            int classIndex = temporaryClasses[i];
            allClasses[i][classIndex] = 1;
        }
    }
    numAllInputs = numAllInputs * numPixels;
}

template <class Type>
void Input<Type>::clear()
{
    if (this->numAllInputs != 0)
    {
        for (int i = 0; i < this->numAllInputs; i++)
        {
            delete[] this->allData[i];
        }
        delete[] this->allData;
    }
    
    if (this->numClasses != 0)
    {
        for (int i = 0; i < this->numAllInputs; i++)
        {
            delete[] this->allClasses[i];
        }
        delete[] this->allClasses;
    }

    if (this->numTrainingInputs != 0)
    {
        for (int i = 0; i < this->numTrainingInputs; i++)
        {
            delete[] this->trainingData[i];
        }
        delete[] this->trainingData;
    }
    
    if (this->numClasses != 0)
    {
        for (int i = 0; i < this->numTrainingInputs; i++)
        {
            delete[] this->trainingClasses[i];
        }
        delete[] this->trainingClasses;
    }

    if (this->numTestInputs != 0)
    {
        for (int i = 0; i < this->numTestInputs; i++)
        {
            delete[] this->testData[i];
        }
        delete[] this->testData;
    }
    
    if (this->numClasses != 0)
    {
        for (int i = 0; i < this->numTestInputs; i++)
        {
            delete[] this->testClasses[i];
        }
        delete[] this->testClasses;
    }

    if(confMatrix)
    {   
        for(int c = 0; c < numClasses; c++)
        {
            delete[] confMatrix[c];
        }
        delete[] confMatrix;
    }
    
    this->numAttributes = 0;
    this->numClasses = 0;
    this->numAllInputs = 0;
    confMatrix = NULL;
    
    allData = NULL;
    allClasses = NULL;
    
    testData = NULL;
    testClasses = NULL;
  
    trainingData = NULL;
    trainingClasses = NULL;
}

template <class Type>
Type** Input<Type>::getData()
{
    return this->allData;
}

template <class Type>
Type** Input<Type>::getIntendedClasses()
{
    return this->allClasses;
}

template <class Type>
void Input<Type>::setTestProportion(double p)
{
	if(p < 0) return;
	if(p > 1 && p <= 100)
	{
		p/=100;
	}
	while(p > 1)
	{
		p/=10;
	}
	numTestInputs = numAllInputs * p;
	numTrainingInputs = numAllInputs - numTestInputs;
	//cout<<numTrainingInputs<<" "<<numTestInputs<<endl;
	if(trainingClasses == NULL)
	{
		trainingClasses = new Type*[numTrainingInputs];
        for (int i = 0; i < numTrainingInputs; i++)
        {
            trainingClasses[i] = new Type [numClasses];
        }
	}
	if(trainingData == NULL)
	{
		trainingData = new Type*[numTrainingInputs];
        for (int i = 0; i < numTrainingInputs; i++)
        {
            trainingData[i] = new Type [numAttributes];
        }
	}
	if(testClasses == NULL)
	{
		testClasses = new Type*[numTestInputs];
        for (int i = 0; i < numTestInputs; i++)
        {
            testClasses[i] = new Type [numClasses];
        }
	}
	if(testData == NULL)
	{
		testData = new Type*[numTestInputs];
        for (int i = 0; i < numTestInputs; i++)
        {
            testData[i] = new Type [numAttributes];
        }
	}
	for(int tr = 0; tr < numTrainingInputs; tr++)
	{
		for(int d = 0; d < numAttributes; d++)
		{
			trainingData[tr][d] = allData[tr][d];
		}
		for(int c = 0; c < numClasses; c++)
		{
			trainingClasses[tr][c] = allClasses[tr][c];	
		}
		
	}
	for(int te = 0; te < numTestInputs; te++)
	{
		for(int d = 0; d < numAttributes; d++)
		{
			testData[te][d] = allData[numTrainingInputs + te][d];
		}
		for(int c = 0; c < numClasses; c++)
		{
			testClasses[te][c] = allClasses[numTrainingInputs + te][c];	
		}
	}
	
	
 ////////////////////////////////////////////////////////////////////////////////
 //    Mat traImg(numTrainingInputs, numAttributes, CV_64F);
    
 //    for(int i = 0; i < numTrainingInputs; i++)
 //    {
 //        for(int j = 0; j < numAttributes; j++)
 //        {
 //            traImg.col(j).row(i) = trainingData[i][j]; 
 //        }
 //    }
    
 //    Mat tesImg(numTestInputs, numAttributes, CV_64F);
    
 //    for(int i = 0; i < numTestInputs; i++)
 //    {
 //        for(int j = 0; j < numAttributes; j++)
 //        {
 //            tesImg.col(j).row(i) = testData[i][j]; 
 //        }
 //    }
    
 //    Type **buffet = new Type* [numAllInputs];
    
 //    numAttributes = maxAtt;
 //    PCA pca(traImg, Mat(), CV_PCA_DATA_AS_ROW, numAttributes);  
 
 //    for (int i = 0; i < this->numAllInputs; i++)
 //    {
 //        delete[] trainingData[i];
 //    }
    
 //    //projetar o training
 //    Mat projecao;
 //    pca.project(traImg, projecao);
 //    for(int i = 0; i < numTrainingInputs; i++)
 //    {
 //        buffet[i] = projecao.ptr<double>(i); 
 //    }
 //    for(int i = 0; i < numTrainingInputs; i++)
 //    {
 //        for(int j = 0; j < numAttributes; j++)
 //        {
 //            trainingData[i][j] = buffet[i][j]; 
 //        }
 //    }
    
 //    //projetar o test
 //    pca.project(tesImg, projecao);
 //    for(int i = 0; i < numTestInputs; i++)
 //    {
 //        buffet[i] = projecao.ptr<double>(i); 
 //    }
 //    for(int i = 0; i < numTestInputs; i++)
 //    {
 //        for(int j = 0; j < numAttributes; j++)
 //        {
 //            testData[i][j] = buffet[i][j]; 
 //        }
 //    }
 //    delete[] buffet;
}

    
template <class Type>
Type** Input<Type>::getTrainingData()
{
	return this->trainingData;
}

template <class Type>
Type** Input<Type>::getTrainingIntendedClasses()
{
	return this->trainingClasses;
}
    
template <class Type>
Type** Input<Type>::getTestData()
{
	return this->testData;
}

template <class Type>
Type** Input<Type>::getTestIntendedClasses()
{
	return this->testClasses;
}


template <class Type>
void Input<Type>::shuffle()
{
    Type* auxInput;
    Type* auxClass;
    auxInput = new Type[numAttributes];
    auxClass = new Type[numClasses];
    int ale = 0;
    for(int i = 0; i < numAllInputs; i++){
    
        ale = i + rand()%(numAllInputs-i);
        
        for(int j = 0; j < numAttributes; j++)
            auxInput[j] = allData[ale][j];
            
        for(int j = 0; j < numClasses; j++)
            auxClass[j] = allClasses[ale][j];
        
        for(int j = 0; j < numAttributes; j++)
            allData[ale][j] = allData[i][j];
            
        for(int j = 0; j < numClasses; j++)
            allClasses[ale][j] = allClasses[i][j];
            
        for(int j = 0; j < numAttributes; j++)
            allData[i][j] = auxInput[j];
            
        for(int j = 0; j < numClasses; j++)
            allClasses[i][j] = auxClass[j];	
    }
}

template <class Type>
void Input<Type>::normalize()
{
    int lines = this->numAllInputs;
    int columns = this->numAttributes;
	Type diference[columns];
	Type minimum[columns];
	for (int j = 0; j < columns; j++)
	{
		minimum[j] = INFINITE;
		Type maximum = 0;
		for (int i = 0; i < lines; i++)
		{
			if (allData[i][j] < minimum[j])
				minimum[j] = allData[i][j];
			if (allData[i][j] > maximum)
				maximum = allData[i][j];
		}
		diference[j] = maximum - minimum[j];
	}
	for (int j = 0; j < columns; j++)
	{
		for (int i = 0; i < lines; i++)
		{
			allData[i][j] = (allData[i][j] - minimum[j]) /diference[j];
		}
	}
}

template <class Type>
int Input<Type>::countInput(const char* filename)
{
    ifstream dataFile(filename);
    char word[256];
    int i = 0;
    while(dataFile.getline(word, 256))
    {
        i++;
    }
    return i;
}

template <class Type>
int Input<Type>::getNumInputs(){
    return numAllInputs;
}

template <class Type>
int Input<Type>::getNumTrainingInputs(){
    return numTrainingInputs;
}

template <class Type>
int Input<Type>::getNumTestInputs(){
    return numTestInputs;
}

template <class Type>
int Input<Type>::getNumAttributes(){
    return numAttributes;
}

template <class Type>
int Input<Type>::getNumClasses(){
    return numClasses;
}

template <class Type>
string Input<Type>::getClassName(Type* value){
    int bigger = 0;
    for(int att = 1; att < numClasses; att++)
    {
        if(value[bigger] < value[att])
        {
            bigger = att;       
        }
    }
    return this->classes[bigger];
}

template <class Type>
string Input<Type>::getClassValue(int i)
{
	return classes[i];
}



template <class Type>
Type Input<Type>::getRate(Type** obtaned,Type** desired, int amount){

	if(!confMatrix)
	{
		confMatrix = new int*[numClasses];
		for(int c = 0; c < numClasses; c++)
		{
			confMatrix[c] = new int[numClasses];
			for(int cc = 0; cc < numClasses; cc++)
			{
				confMatrix[c][cc] = 0;
			}
		}
	}
	
	for(int c = 0; c < numClasses; c++)
		{
			for(int cc = 0; cc < numClasses; cc++)
			{
				confMatrix[c][cc] = 0;
			}
		}
	
    int rights = 0;
    for(int inst = 0; inst < amount; inst++)
    {
        int classObtaned = 0;
        int classDesired = 0;
        
        for(int cla = 0; cla < numClasses; cla++)
        {
            if(obtaned[inst][classObtaned] < obtaned[inst][cla])
            {
                classObtaned = cla;       
            }
        }
        
        for(int cla = 0; cla < numClasses; cla++)
        {
            if(desired[inst][classDesired] < desired[inst][cla])
            {
                classDesired = cla;       
            }
        }
        
        if(classObtaned == classDesired)
        {
            rights++;
        }
        confMatrix[classDesired][classObtaned]++;
    }
    
    for(int cc = 0; cc < numClasses; cc++)
	{
		cout<<cc<<" -> "<<this->classes[cc]<<endl;
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
			cout<<confMatrix[c][cc]<<"\t";
		}
		cout<<endl;
	}
    cout<<rights<<" de "<<amount<<" "<<endl;
    return (Type)rights/(Type)amount;
}

template <class Type>
int** Input<Type>::getConfMatrix(){
	return confMatrix;
}

#endif
