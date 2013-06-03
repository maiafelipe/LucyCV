#ifndef __PREPROC_H__
#define __PREPROC_H__

#include <iostream>
#include <cmath>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace std;

using namespace cv;

template <class Type>
class PreProcessing
{
private:
  Type* norMinimum;
  Type* norDiference;

  Type* norMean;
  Type* norStdDev;

  PCA* objPca; 

  bool initNorMinimum;
  bool initNorDiference;
  bool initNorMean;
  bool initNorStdDev;
  bool initObjPca;

public:
  PreProcessing();
  ~PreProcessing();

  Type** normalize(Type** in, int n_patt, int n_att, bool relative = false);

  Type** normalize_mean(Type** in, int n_patt, int n_att, bool relative = false);

  Type** pca(Type** in, int n_patt, int n_att, int max_att, bool relative = false);
};

template <class Type>
PreProcessing<Type>::PreProcessing()
{
  initNorDiference = false;
  initNorMinimum = false;
  initNorMean = false;
  initNorStdDev = false;
  initObjPca = false;
}

template <class Type>
PreProcessing<Type>::~PreProcessing()
{
  if(initNorMinimum)
  {
    delete[] norMinimum;
  }
  if(initNorDiference)
  {
    delete[] norDiference;
  }
  if(initNorMean)
  {
    delete[] norMean;
  }
  if(initNorStdDev)
  {
    delete[] norStdDev;
  }
  if(initObjPca)
  {
    delete objPca;
  }
}

template <class Type>
Type** PreProcessing<Type>::normalize(Type** in, int n_patt, int n_att, bool relative)
{
  int lines = n_patt;
  int columns = n_att;
  if(!relative)
  {
    if(initNorMinimum)
    {
      delete[] norMinimum;
      initNorMinimum = false; 
    }
    norMinimum = new Type[columns];
    initNorMinimum = true;
    if(initNorDiference)
    {
      delete[] norDiference;
      initNorDiference = false;
    }
    norDiference = new Type[columns];
    initNorDiference = true;
    for (int j = 0; j < columns; j++)
    {
      norMinimum[j] = INFINITE;
      Type maximum = 0;
      for (int i = 0; i < lines; i++)
      {
        if (in[i][j] < norMinimum[j])
          norMinimum[j] = in[i][j];
        if (in[i][j] > maximum)
          maximum = in[i][j];
      }
      norDiference[j] = maximum - norMinimum[j];
    }
  }
  Type** outData = new Type*[n_patt];
  for (int i = 0; i < lines; i++)
  {
    outData[i] = new Type[n_att];
    for (int j = 0; j < columns; j++)
    {
      outData[i][j] = (in[i][j] - norMinimum[j]) /norDiference[j];
    }
  }
  return outData;
}

template <class Type>
Type** PreProcessing<Type>::normalize_mean(Type** in, int n_patt, int n_att, bool relative)
{
  if(!relative)
  {
    if(initNorMean)
    {
      delete[] norMean;
      initNorMean = false; 
    }
    norMean = new Type[n_att];
    initNorMean = true;
    if(initNorStdDev)
    {
      delete[] norStdDev;
      initNorStdDev = false;
    }
    norStdDev = new Type[n_att];
    initNorStdDev = true;
    for (int att = 0; att < n_att; att++)
    {
      Type mean = 0;
      for (int patt = 0; patt < n_patt; patt++)
      {
        mean += in[patt][att];
      }
      norMean[att] = mean/n_patt;
      Type var = 0;
      for (int patt = 0; patt < n_patt; patt++)
      {
        var += ((in[patt][att] - norMean[att]) * (in[patt][att] - norMean[att]));
      }
      norStdDev[att] = sqrt(var/n_patt);
    }
  }
  Type** outData = new Type*[n_patt];
  for (int patt = 0; patt < n_patt; patt++)
  {
    outData[patt] = new Type[n_att];
    for (int att = 0; att < n_att; att++)
    {
      outData[patt][att] = (in[patt][att] - norMean[att])/norStdDev[att];
    }
  }
  return outData;
}

template <class Type>
Type** PreProcessing<Type>::pca(Type** in, int numInputs, int numAttributes, int max_att, bool relative)
{
  Mat inMatrix(numInputs, numAttributes, CV_64F);
    
  for(int i = 0; i < numInputs; i++)
  {
      for(int j = 0; j < numAttributes; j++)
      {
          inMatrix.col(j).row(i) = in[i][j]; 
      }
  }
  if(!relative)
  {
    if(initObjPca)
    {
      delete objPca;
      initObjPca = false;
    }
    objPca = new PCA(inMatrix, Mat(), CV_PCA_DATA_AS_ROW, max_att);  
    initObjPca = true;
  }
    
  if(!objPca)
  {
    return NULL;
  }
  //projetar o training
  Mat projection;
  objPca->project(inMatrix, projection);
  
  Type **buffet = new Type* [numInputs];
  
  for(int i = 0; i < numInputs; i++)
  {
    buffet[i] = projection.ptr<double>(i); 
  }

  Type** outData = new Type*[numInputs];

  for(int i = 0; i < numInputs; i++)
  {
    outData[i] = new Type[numAttributes];
    for(int j = 0; j < numAttributes; j++)
    {
      outData[i][j] = buffet[i][j]; 
    }
  }
  
  return outData;
}

#endif