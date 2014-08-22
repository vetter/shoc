// This example from an alpha release of the Scalable HeterOgeneous Computing
// (SHOC) Benchmark Suite Alpha v1.1.4a-mic for Intel MIC architecture
// Contact: Jeffrey Vetter <vetter@ornl.gov>
//          Rezaur Rahman <rezaur.rahman@intel.com>
//
// Copyright (c) 2011-2013, UT-Battelle, LLC
// Copyright (c) 2013, Intel Corporation
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

#include <string.h>

#ifdef __MIC__
#  include "immintrin.h"
#endif // __MIC__

template <class T>
__declspec(target(MIC)) void scanTiling(T *input,  T* output, const size_t n, 
        T* opblocksum, int ThreadCount)
{
    int i;
    int lastElement = n;
    int bufsize = ideal_buffer_size;

    // The operative principle here is to do everything with one portion of 
    // the array before moving to the next
    output[0]=input[0];  // we are doing an inclusive scan

    memset(opblocksum, 0, sizeof(T)*ThreadCount);

    for(int k = 0; k < n; k += ideal_buffer_size*ThreadCount) 
    {
        if (k + ideal_buffer_size*ThreadCount > n)  // if last buffer...
        {
	  bufsize = (int)((((int)n)-k)/ThreadCount);
        }
        
        lastElement = k + bufsize * ThreadCount - 1;

        #pragma omp parallel for shared(input, output) num_threads(ThreadCount)
        for(int i = 0; i < ThreadCount; i++) 
        {
            int offset=k + omp_get_thread_num()*bufsize;
            opblocksum[i+1] = input[offset];  // this is an inclusive scan
            for(int j = offset+1; j < offset+bufsize; j++)
            {
                opblocksum[i+1] += input[j];
            }
        }

        for(int i = 1; i <= ThreadCount; i++)
        {
            opblocksum[i] += opblocksum[i-1];
        }

        #pragma omp parallel for shared(output,opblocksum) num_threads(ThreadCount)
        for(int i = 0; i < ThreadCount; i++) 
        {
            int offset=k + omp_get_thread_num()*bufsize;
            output[offset] = opblocksum[i] + input[offset];
            for(int j = offset+1; j < offset+bufsize;j++) 
            {
                output[j] = output[j-1] + input[j];

#ifdef __MIC__

                // Fixed the Eviction issues but bogs the system WAY down
                if (j & 0xF == 0) 
                {
                    int z = j - 0xF;
                    //Should 0 be _MM_HINT_T0?
                    _mm_clevict (&input[z],_MM_HINT_T0);
                    //Should 1 be _MM_HINT_T1?
                    _mm_clevict (&input[z],_MM_HINT_T1);  
                    _mm_clevict (&output[z],_MM_HINT_T0);
                    _mm_clevict (&output[z],_MM_HINT_T1);
                }
#endif
            }
        } 

        // Element 0 of the sums needs the last value of the previous iteration
        opblocksum[0] = output[lastElement];
    }

    // We handle any remnant here.
    for (i = lastElement; i < n; i++) 
    {
        output[i] = input[i] + output[i-1];
    }
}
