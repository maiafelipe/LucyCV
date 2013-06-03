#ifndef __USEFUL_DEFINITIONS_H__
#define __USEFUL_DEFINITIONS_H__

#define activationTemplate template <class> class
#define distanceTemplate template <class> class
#define neighborhoodFunctionTemplate template <class> class
#define combinerTemplate template<class> class
#define trainingTemplate template<class> class
#define neuronTemplate template < distanceTemplate, class> class
#define neighborhoodTemplate template < neuronTemplate, neighborhoodFunctionTemplate , distanceTemplate, class> class
#define layerTemplate template < neuronTemplate, activationTemplate, class > class
#endif
