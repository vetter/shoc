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
//
// Modified: 7-3-2013 Philip C. Roth to integrate with SHOC infrastructure.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
void
MICStencil<T>::operator()( Matrix2D<T>& mtx, unsigned int nIters )
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    unsigned int uOMP_Threads = omp_get_max_threads_target(TARGET_MIC, device);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    __declspec(target(mic), align(sizeof(T))) T* rarr1 = mtx.GetFlatData();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    __declspec(target(mic), align(4)) unsigned int  nrows           = mtx.GetNumRows();
    __declspec(target(mic), align(4)) unsigned int  ncols           = mtx.GetNumColumns();
    __declspec(target(mic), align(4)) unsigned int  nPaddedCols     = mtx.GetNumPaddedColumns();
    __declspec(target(mic), align(4)) unsigned int  nextralines     = (nrows - 2) % uOMP_Threads;
    __declspec(target(mic), align(4)) unsigned int  uLinesPerThread = (unsigned int)floor((double)(nrows - 2) / uOMP_Threads);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    __declspec(target(mic), align(sizeof(T)))   T wcenter   = this->wCenter;
    __declspec(target(mic), align(sizeof(T)))   T wdiag     = this->wDiagonal;
    __declspec(target(mic), align(sizeof(T)))   T wcardinal = this->wCardinal;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    unsigned int len = nrows * nPaddedCols;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    #pragma offload target(mic:device) \
                                in(rarr1:length(len)) out(rarr1:length(len)) \
                                in(uOMP_Threads) in(uLinesPerThread) in(nrows)  \
                                in(ncols) in(nPaddedCols) in(nextralines) in(wcenter) in(wdiag) \
                                in(wcardinal) \
                                in(nIters) \
                                in(len)
    {
        T*  pTmp    = rarr1;
        T*  pCrnt   = (T*)_mm_malloc(len * sizeof(T), sizeof(T));
        T*  pAux    = NULL;

        // copy the halo from the initial buffer into the second buffer
        // Note: when doing local iterations, these values do not change
        // but they can change in the MPI vesrion after an inter-process
        // halo exchange.
        //
        #pragma omp parallel for firstprivate(pTmp, pCrnt, nrows, ncols, nPaddedCols)
        for( unsigned int c = 0; c < ncols; c++ )
        {
            // logically: pCrnt[0][c] = pTmp[0][c];
            pCrnt[0*nPaddedCols + c] = pTmp[0*nPaddedCols + c];
            // logically: pCrnt[nrows - 1][c] = pTmp[nrows - 1][c];
            pCrnt[(nrows - 1)*nPaddedCols + c] = pTmp[(nrows - 1)*nPaddedCols + c];
        }

        #pragma omp parallel for firstprivate(pTmp, pCrnt, nrows, ncols, nPaddedCols)
        for( unsigned int r = 0; r < nrows; r++ )
        {
            // logically: pCrnt[r][0] = pTmp[r][0];
            pCrnt[r*nPaddedCols + 0] = pTmp[r*nPaddedCols + 0];
            // logically: pCrnt[r][ncols - 1] = pTmp[r][ncols - 1];
            pCrnt[r*nPaddedCols + (ncols - 1)] = pTmp[r*nPaddedCols + (ncols - 1)];
        }



        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        for (unsigned int cntIterations = 0; cntIterations < nIters; cntIterations ++)
        {
            #pragma omp parallel for firstprivate(pTmp, pCrnt, uLinesPerThread, wdiag, wcardinal, wcenter, ncols, nPaddedCols)
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
                        cardinal0   =   pTmp[(cntLine - 1) * nPaddedCols + cntColumn] +
                                        pTmp[(cntLine + 1) * nPaddedCols + cntColumn] +
                                        pTmp[ cntLine * nPaddedCols + cntColumn - 1] +
                                        pTmp[ cntLine * nPaddedCols + cntColumn + 1];

                        diagonal0   =   pTmp[(cntLine - 1) * nPaddedCols + cntColumn - 1] +
                                        pTmp[(cntLine - 1) * nPaddedCols + cntColumn + 1] +
                                        pTmp[(cntLine + 1) * nPaddedCols + cntColumn - 1] +
                                        pTmp[(cntLine + 1) * nPaddedCols + cntColumn + 1];

                        center0     =   pTmp[cntLine * nPaddedCols + cntColumn];


                        pCrnt[cntLine * nPaddedCols + cntColumn]  = wcenter * center0 + wdiag * diagonal0 + wcardinal * cardinal0;
                    }
                }
            }

            // Switch pointers
            pAux    = pTmp;
            pTmp    = pCrnt;
            pCrnt   = pAux;
        }

        // Check for an odd number of iterations.
        // Set pAux to point to the auxiliary array, regardless of
        // the number of iterations.
        if( nIters & 0x01 )
        {
            // We did an odd number of iterations.
            // Right now, pTmp points to the auxiliary buffer we allocated
            // in this function, and pCrnt points to the mtx object's 
            // flat array.
            pAux = pTmp;

            // Copy from the auxiliary buffer back into the mtx 
            // argument's buffer.
            #pragma omp parallel for \
                firstprivate(rarr1, pCrnt, nrows, nPaddedCols)
            for( unsigned int i = 0; i < nrows * nPaddedCols; i++ )
            {
                rarr1[i] = pAux[i];
            }
        }
        else
        {
            // We did an even number of iterations.
            // Right now, pCrnt points to the auxiliary buffer we allocated
            // in this function.
            pAux = pCrnt;
        }

        _mm_free(pAux);
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
