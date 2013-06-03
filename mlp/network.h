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
	virtual int train(Type**, Type**, int){}
	virtual Type* answer(Type*){}
	virtual void print(){cout << "Sou network" << endl;}
};

#endif
