#ifndef __PREPROC_H__
#define __PREPROC_H__

#include <iostream>
#include <cmath>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace std;

// using namespace cv;

template <class Type>
class PreProcessing
{
private:
  Type* norMinimum;
  Type* norDiference;

  Type* norMean;
  Type* norStdDev;

  cv::PCA* objPca; 

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

  int detectFace(Type** in, Type** desired, int n_patt, int n_att, int n_class, int origHeight, int origWidth, int newHeight, int newWidth);

  IplImage* detectAFace(cv::Mat& img, CascadeClassifier& cascade, double scale, bool tryflip, IplImage* src );
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
  // cout<<n_patt<<endl;
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
        // cout<<i<<endl;
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
  cv::Mat inMatrix(numInputs, numAttributes, CV_64F);
    
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
    objPca = new cv::PCA(inMatrix, Mat(), CV_PCA_DATA_AS_ROW, max_att);  
    initObjPca = true;
  }
    
  if(!objPca)
  {
    return NULL;
  }
  //projetar o training

  cv::Mat projection;
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

template <class Type>
int PreProcessing<Type>::detectFace(Type** in, Type** desired, int n_patt, int n_att, int n_class, int origHeight, int origWidth, int newHeight, int newWidth)
{
  int count = 0;
  Type** resultIn = new Type*[n_patt];
  Type** resultDesired = new Type*[n_patt];
  for (int patt = 0; patt < n_patt; patt++)
  {
    resultIn[patt] = new Type[newHeight*newWidth];
    resultDesired[patt] = new Type[n_class];
    for (int cl = 0; cl < n_class; cl++)
    {
      resultDesired[patt][cl] = 0;
    }
  }
  int numColors = 1;
  for (int patt = 0; patt < n_patt; patt++)
  {
    // cout<<"patt "<<patt<<endl;
    int att = 0;
    Mat imag(origHeight, origWidth, CV_8UC1);
    for (int h = 0; h < origHeight; h++)
    {
      for (int w = 0; w < origWidth; w++)
      {
        imag.col(w).row(h) = in[patt][att];  
        // cout<<imag.col(w).row(h)<<" ";
        att++;
      }
      // cout<<endl;
    }
    // cout<<"v1"<<endl;
    // cout<<endl<<endl;
    // cout<<patt<<endl;
    // namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    // imshow( "Display window", imag );   
    // waitKey(0);
 
    IplImage imagIPL = imag;  
    char nome[50];
    // sprintf(nome, "Janela %d", patt);
    // cvShowImage(nome, &imagIPL);
    
    CascadeClassifier cascade;
    string cascadeName = "haarcascades/haarcascade_frontalface_alt.xml";
    if( !cascade.load(cascadeName) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return 0;
    }
    // cout<<"v2"<<endl;
    IplImage* face = NULL;
    face = detectAFace(imag, cascade, 1, false, &imagIPL);
    // cout<<"v3"<<endl;
    IplImage *rface = cvCreateImage( cvSize(newWidth, newHeight), imagIPL.depth, imagIPL.nChannels );
    // cout<<"v4"<<endl;

    if(face){
        cvResize(face, rface);
         // sprintf(nome, "Janela %d 2", patt);
         // cvShowImage(nome, rface);
         // waitKey(0);
        count++;
        // cout<<"2"<<endl;
    }else{
        string cascadeName = "haarcascades/haarcascade_frontalface_default.xml";
        if( !cascade.load(cascadeName) )
        {
            cerr << "ERROR: Could not load classifier cascade" << endl;
            return 0;
        }
        IplImage* face = NULL;
        face = detectAFace(imag, cascade, 1, false, &imagIPL);
        if(face){
            cvResize(face, rface);
            // sprintf(nome, "Janela %d 2", patt);
            // cvShowImage(nome, rface);
            // waitKey(0);
            count++;
            // cout<<"3"<<endl;   
        }else{
            string cascadeName = "haarcascades/haarcascade_frontalface_alt2.xml";
            if( !cascade.load(cascadeName) )
            {
                cerr << "ERROR: Could not load classifier cascade" << endl;
                return 0;
            }
            IplImage* face = NULL;
            face = detectAFace(imag, cascade, 1, false, &imagIPL);
            if(face){
                cvResize(face, rface);
                // sprintf(nome, "Janela %d 2", patt);
                // cvShowImage(nome, rface);
                // waitKey(0);
                count++;
                // cout<<"4"<<endl;   
            }else{
                string cascadeName = "haarcascades/haarcascade_profileface.xml";
                if( !cascade.load(cascadeName) )
                {
                    cerr << "ERROR: Could not load classifier cascade" << endl;
                    return 0;
                }
                IplImage* face = NULL;
                face = detectAFace(imag, cascade, 1, false, &imagIPL);
                if(face){
                    cvResize(face, rface);
                    // sprintf(nome, "Janela %d 2", patt);
                    // cvShowImage(nome, rface);
                    // waitKey(0);
                    count++;
                    // cout<<"5"<<endl;                     
                }
            }
        }
    }
    if(face){
        // cout<<"detec: 1"<<endl;   
        int j = 0;
        for(int h = 0; h < newHeight; h++){
            for(int w = 0; w < newWidth; w++){
                CvScalar pont = cvGet2D(rface,h,w);
                for(int c = 0; c < numColors; c++){
                    resultIn[count-1][j] = pont.val[c];
                    j++; 
                }
                // cout<<h<<" "<<w<<" "<<j<<" "<<rface->height<<" "<<rface->width<<endl;
            }
        }
        // cout<<"detec: 2"<<endl;
        for (int cl = 0; cl < n_class; cl++)
        {
          resultDesired[count-1][cl] = desired[patt][cl];
        }
        // cout<<"detec: 3"<<endl;

    }else{
        // int j = 0;
        // for(int h = 0; h < newHeight; h++){
        //     for(int w = 0; w < newWidth; w++){
        //         for(int c = 0; c < numColors; c++){
        //             resultIn[patt][j] = 0;
        //             j++; 
        //         }
        //     }
        // }
    }
  }
  in = resultIn;
  desired = resultDesired;
  cout<<count<<"/"<<n_patt<<endl;
  return count;
}

template <class Type>
IplImage*  PreProcessing<Type>::detectAFace( Mat& img, CascadeClassifier& cascade,
                                          double scale, bool tryflip, IplImage* src )
{
    int i = 0;
    double t = 0;
    vector<Rect> faces, faces2;

    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    // cvtColor( img, gray, CV_BGR2GRAY );
    gray = img;  
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CV_HAAR_FIND_BIGGEST_OBJECT
                                 //|CV_HAAR_DO_ROUGH_SEARCH
                                 |CV_HAAR_SCALE_IMAGE
                                 ,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)cvGetTickCount() - t;
    // printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    if(faces.size() == 0) return NULL;
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
      
        IplImage* cropped = cvCreateImage(cvSize(r->width*scale,r->height*scale), src->depth, src->nChannels ) ;
        cvSetImageROI( src,  cvRect( r->x*scale,r->y*scale ,  r->width*scale,r->height*scale) );
      
        // Do the copy
        cvCopy( src, cropped );
        cvResetImageROI( src ); 
        char nome[50];
        // sprintf(nome, "Janela %d", i);
        // cvShowImage(nome, cropped );
        
        //cvSaveImage("test.jpg", cropped);
        return cropped; 
    }
    return NULL;
}

#endif
