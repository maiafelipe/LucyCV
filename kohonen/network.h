#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <iostream>

using namespace std;

template <class Type>
class Network
{
public:
	virtual void setAccuracy(double){}
	virtual double getAccuracy(){}
	virtual void train(Type**, Type**, int){}
	virtual Type* answer(Type*){}
	virtual int cluster(Type*){}
	virtual void print(){cout << "Sou network" << endl;}
	virtual void save(const char*){}
	virtual void load(const char*){}
};

#endif