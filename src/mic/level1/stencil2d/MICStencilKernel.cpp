// This example from an alpha release of the Scalable HeterOgeneous Computing
// (SHOC) Benchmark Suite Alpha v1.1.4a-mic for Intel MIC architecture
// Contact: Jeffrey Vetter <vetter@ornl.gov>
//          Rezaur Rahman <rezaur.rahman@intel.com>
//
// Copyright (c) 2011-2013, UT-Battelle, LLC
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//   
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Oak Ridge National Laboratory, nor UT-Battelle, LLC, 
//    nor the names of its contributors may be used to endorse or promote 
//    products derived from this software without specific prior written 
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
// THE POSSIBILITY OF SUCH DAMAGE.

#if defined(__APPLE__)
#if _GLIBCXX_ATOMIC_BUILTINS == 1
#undef _GLIBCXX_ATOMIC_BUILTINS
#endif // _GLIBCXX_ATOMIC_BUILTINS
#endif // __APPLE__

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "omp.h"
#include "math.h"
#include "offload.h"
#include "Timer.h"
#include "MICStencil.cpp"

#define MAX_OMP_THREADS 256

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel Author:   Valentin Andrei - valentin.andrei@intel.com (SSG/SSD/PTAC/PAC Power & Characterization)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
void
MICStencil<T>::operator()( Matrix2D<T>& mtx, unsigned int nIters )
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    unsigned int uOMP_Threads = 240;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    __declspec(target(mic), align(sizeof(T))) T* rarr1 = mtx.GetFlatData();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    __declspec(target(mic), align(4)) unsigned int  nrows           = mtx.GetNumRows();
    __declspec(target(mic), align(4)) unsigned int  ncols           = mtx.GetNumColumns();
    __declspec(target(mic), align(4)) unsigned int  nextralines     = (nrows - 2) % uOMP_Threads;
    __declspec(target(mic), align(4)) unsigned int  uLinesPerThread = (unsigned int)floor((double)(nrows - 2) / uOMP_Threads);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    __declspec(target(mic), align(sizeof(T)))   T wcenter   = this->wCenter;
    __declspec(target(mic), align(sizeof(T)))   T wdiag     = this->wDiagonal;
    __declspec(target(mic), align(sizeof(T)))   T wcardinal = this->wCardinal;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    unsigned int len = nrows * ncols;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    #pragma offload target(mic) in(rarr1:length(len)) out(rarr1:length(len))    \
                                in(uOMP_Threads) in(uLinesPerThread) in(nrows)  \
                                in(ncols) in(nextralines) in(wcenter) in(wdiag) \
                                in(wcardinal)
    {
        T*  pTmp    = rarr1;
        T*  pCrnt   = (T*)_mm_malloc(len * sizeof(T), sizeof(T));
        T*  pAux    = NULL;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        for (unsigned int cntIterations = 0; cntIterations < nIters; cntIterations ++)
        {
            #pragma omp parallel for firstprivate(pTmp, pCrnt, uLinesPerThread, wdiag, wcardinal, wcenter, ncols)
            for (unsigned int uThreadId = 0; uThreadId < uOMP_Threads; uThreadId++)
            {
                unsigned int uStartLine = 0;
                unsigned int uEndLine   = 0;

                if (uThreadId < nextralines)
                {
                    uStartLine  = uThreadId     * (uLinesPerThread + 1) + 1;
                    uEndLine    = uStartLine    + (uLinesPerThread + 1);
                }
                else
                {
                    uStartLine  = nextralines   + uThreadId * uLinesPerThread + 1;
                    uEndLine    = uStartLine    + uLinesPerThread;
                }

                T   cardinal0   = 0.0;
                T   diagonal0   = 0.0;
                T   center0     = 0.0;

                for (unsigned int cntLine = uStartLine; cntLine < uEndLine; cntLine ++)
                {
                    #pragma ivdep
                    for (unsigned int cntColumn = 1; cntColumn < (ncols - 1); cntColumn ++)
                    {
                        cardinal0   =   pTmp[(cntLine - 1) * ncols + cntColumn] +
                                        pTmp[(cntLine + 1) * ncols + cntColumn] +
                                        pTmp[ cntLine * ncols + cntColumn - 1] +
                                        pTmp[ cntLine * ncols + cntColumn + 1];

                        diagonal0   =   pTmp[(cntLine - 1) * ncols + cntColumn - 1] +
                                        pTmp[(cntLine - 1) * ncols + cntColumn + 1] +
                                        pTmp[(cntLine + 1) * ncols + cntColumn - 1] +
                                        pTmp[(cntLine + 1) * ncols + cntColumn + 1];

                        center0     =   pTmp[cntLine * ncols + cntColumn];


                        pCrnt[cntLine * ncols + cntColumn]  = wcenter * center0 + wdiag * diagonal0 + wcardinal * cardinal0;
                    }
                }
            }

            // Switch pointers
            pAux    = pTmp;
            pTmp    = pCrnt;
            pCrnt   = pAux;
        }

        _mm_free(pCrnt);
    }
}

void
EnsureStencilInstantiation( void )
{
    MICStencil<float> csf( 0, 0, 0, 0 );
    Matrix2D<float> mf( 2, 2 );
    csf( mf, 0 );

    MICStencil<double> csd( 0, 0, 0, 0 );
    Matrix2D<double> md( 2, 2 );
    csd( md, 0 );
}
