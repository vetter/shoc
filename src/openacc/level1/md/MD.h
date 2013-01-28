#ifndef MD_H__
#define MD_H__

#include "constants.h"
#include "OptionParser.h"
#include "ResultDatabase.h"

// Forward Declarations
template <class T, class forceVecType, class posVecType>
void runTest(const string& testName, ResultDatabase& resultDB,
          OptionParser& op);

template <class T, class posVecType>
inline T distance(const posVecType* position, const int i, const int j);

template <class T>
inline void insertInOrder(std::list<T>& currDist, std::list<int>& currList,
        const int j, const T distIJ, const int maxNeighbors);

template <class T, class posVecType>
inline int buildNeighborList(const int nAtom, const posVecType* position,
        int* neighborList);

template <class T>
inline int populateNeighborList(std::list<T>& currDist,
        std::list<int>& currList, const int j, const int nAtom,
        int* neighborList);

// C Calls To OpenACC kernels
extern "C" void ljSingle(const unsigned int iters,
                         const int nAtom,
                         const int maxNeighbors,
                         float3* force,
                         const float4* position,
                         const int*    neighborList,
                         double* itersTime,
                         double* totalTime);

extern "C" void ljDouble(const unsigned int iters,
                         const int nAtom,
                         const int maxNeighbors,
                         double3* force,
                         const double4* position,
                         const int*    neighborList,
                         double* itersTime,
                         double* totalTIme);

template <class T, class T2>
void lj(const unsigned int iters,
        const int nAtom,
        const int maxNeighbors,
        T* force,
        const T2*  position,
        const int* neighborList,
        double* itersTime,
        double* totalTime)
{
    ;
}

template <>
void lj(const unsigned int iters,
        const int nAtom,
        const int maxNeighbors,
        float3* force,
        const float4* position,
        const int*    neighborList,
        double* itersTime,
        double* totalTime) 
{
    ljSingle(iters, nAtom, maxNeighbors, force, position, neighborList, 
            itersTime, totalTime);
}

template <>
void lj(const unsigned int iters,
        const int nAtom,
        const int maxNeighbors,
        double3* force,
        const double4* position,
        const int*    neighborList,
        double* itersTime,
        double* totalTime) 
{
    ljDouble(iters, nAtom, maxNeighbors, force, position, neighborList, 
            itersTime, totalTime);
}

#endif // _MD_H__
