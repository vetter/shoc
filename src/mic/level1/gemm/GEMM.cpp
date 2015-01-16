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

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/sysctl.h>

#include <mkl_service.h>
#include <mkl_blas.h>
#include "omp.h"

#include "offload.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"

using namespace std;

// Forward declarations
template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

template <class T>
inline void devGEMM(char transa, char transb, int m, int n, int k, T alpha,
        const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc);

// ********************************************************
// Function: toString
//
// Purpose:
//   Simple templated function to convert objects into
//   strings using stringstream
//
// Arguments:
//   t: the object to convert to a string
//
// Returns:  a string representation of t
//
// Modifications:
//
// ********************************************************
template<class T> inline std::string toString(const T& t)
{
   stringstream ss;
   ss << t;
   return ss.str();
}

// ********************************************************
// Function: error
//
// Purpose:
//   Simple routine to print an error message and exit
//
// Arguments:
//   message: an error message to print before exiting
//
// ********************************************************
void error(char *message)
{
   cerr << "ERROR: " << message << endl;
   exit(1);
}

// ********************************************************
// Function: fill
//
// Purpose:
//   Simple routine to initialize input array
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//
// ********************************************************
template <class T>
void fill(T *A, const int n, const int maxi)
{
   for (int j = 0; j < n; j++)
      A[j] = T((rand() % (maxi * 2 + 1)) - maxi) / (maxi + 1.);
}

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  The user is allowed to specify
//   the size of the input data in kiB.
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Anthony Danalis
// Creation: September 08, 2009
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
   op.addOption("KiB", OPT_INT, "0", "data size (in Kibibytes)");
   op.addOption("N", OPT_INT, "0", "SQ Matrix Dimension");
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   This benchmark measures the performance of the general
//   matrix multiplication (GEMM) operation in GFLOPS.  Results are
//   reported with and without PCIe transfer time.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// ****************************************************************************

// The following two methods are just a templatized call to GEMM.
template<> 
inline __declspec(target(MIC)) void devGEMM<double>(char transa, char transb, 
        int m, int n, int k, double alpha, const double *A, int lda, 
        const double *B, int ldb, double beta, double *C, int ldc) 
{

    dgemm(&transa, &transb, &m, &n, &k, &alpha,
        A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
inline __declspec(target(MIC)) void devGEMM<float>(char transa, char transb, 
        int m, int n, int k, float alpha, const float *A, int lda, 
        const float *B, int ldb, float beta, float *C, int ldc )
{
    sgemm(&transa, &transb, &m, &n, &k, &alpha,
           A, &lda, B, &ldb, &beta, C, &ldc);
}

void RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{
    cout << "Running single precision test" << endl;
    RunTest<float>("SGEMM", resultDB, op);
    
    cout << "Running double precision test" << endl;
    RunTest<double>("DGEMM", resultDB, op);
}

// Macro for fixing leading dimension
#define FIX_LD(x) (((x) * sizeof(T)) % 1024 == 0 ? (x) + 128 : (x))

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op)
{
    const int micdev = op.getOptionInt("device");
    
    // Repeat the test multiple times
    int passes = op.getOptionInt("passes");
   
    // Dimension of the matrix
    int N;

    // Parse command line options
    // There are basically three options here
    // "-s [1-4]" pick a predefined size
    // "--N [number]" use a number x number square matrix
    // "--KiB [number]" use a large square matrix
    if ((op.getOptionInt("KiB") == 0) && (op.getOptionInt("N") == 0))
    {
        int probSizes[4] = { 1, 4, 8, 16 };
        N = probSizes[op.getOptionInt("size")-1] * 1024 / sizeof(T);
    }
    else if((op.getOptionInt("KiB") == 0)) 
    {
        N = op.getOptionInt("N"); 
        // For double we run half the size matrices
        N = N / (sizeof (T)/sizeof(float));
    }
    else 
    {
        N = op.getOptionInt("KiB") * 1024 / sizeof(T);
    }


    int LDA = FIX_LD(N);

    __declspec(target(MIC)) static T *A;
    __declspec(target(MIC)) static T *B;
    __declspec(target(MIC)) static T *C;

    // Use a square matrix
    size_t matrix_elements = LDA * N;
    size_t matrix_bytes = matrix_elements * sizeof(T);
   
    // Allocate memory for the matrices
    const int alignment = 2 * 1024 * 1024;
    A = (T *)_mm_malloc(matrix_bytes, alignment);
    B = (T *)_mm_malloc(matrix_bytes, alignment);
    C = (T *)_mm_malloc(matrix_bytes, alignment);

    if(!A || !B || !C)
    {
        cerr << "memory allocation failed" << endl;
        return;
    }
   
    // Fill the matrices with some random data
    fill<T>(A, LDA * N, 31);
    fill<T>(B, LDA * N, 31);
    fill<T>(C, LDA * N, 31);

    // Allocate memory on the card and keep it around
    #pragma offload target(MIC:micdev) \
        in(A:length(matrix_elements)  free_if(0) ) \
        in(B:length(matrix_elements)  free_if(0) ) \
        in(C:length(matrix_elements)  free_if(0) ) \
        out(C:length(matrix_elements) free_if(0) )
    {
    }

    // Start the timer for the PCIe transfer
    int txToDevTimerHandle = Timer::Start();

    #pragma offload target(MIC:micdev) \
        in(A:length(matrix_elements)  alloc_if(0) free_if(0)) \
        in(B:length(matrix_elements)  alloc_if(0) free_if(0)) \
        nocopy(C:length(matrix_elements) alloc_if(0) free_if(0))
    {
    }

    double transfer_time = Timer::Stop(txToDevTimerHandle, "tx to dev");

    //Flag for timing first output tranfer
    bool first = true;

    // Begin main test loop
    for (; passes > 0; --passes)
    {
        for (int i = 0; i < 2; i++)
        {
            // Set up all the variables for the GEMM call
            const char transa = 'N';
            const char transb = i ? 'T' : 'N';
            const int nb = 128;
            const int idim = N / nb;
            int dim = idim * nb;
            const int m = dim;
            const int n = dim;
            const int k = dim;
            const int lda = FIX_LD(dim);
            const int ldb = FIX_LD(dim);
            const int ldc = FIX_LD(dim);


            #pragma offload target(MIC:micdev) \
                in(A:length(matrix_elements)  alloc_if(0) free_if(0))  \
                in(B:length(matrix_elements)  alloc_if(0) free_if(0))  \
                in(C:length(matrix_elements)  alloc_if(0) free_if(0))  \
                out(C:length(matrix_elements) alloc_if(0) free_if(0))
            {
                //local declaration
                const T alpha = 1;
                const T beta = -1;

                // Warm up, the reason of this # pragma loop is to load 
                // necessary libraries.
                devGEMM<T>(transa, transb, m, n, k, alpha, A, lda, B, ldb, 
                        beta, C, ldc);
            }

            const T alpha = 1;
            const T beta = 0;
            // Time it takes for the actual gemm call
            int kernelTimerHandle = Timer::Start();

            #pragma offload target(MIC:micdev) \
                            nocopy(A)          \
                            nocopy(B)          \
                            nocopy(C)  
            {
                // Do 4 iterations
                for (int ii = 0; ii < 4; ++ii)
                {
                   devGEMM<T>(transa, transb, m, n, k, alpha, A, lda, B, ldb,
                           beta, C, ldc);
                }
            }
            double blas_time = Timer::Stop(kernelTimerHandle, "gemm") / 4.0;
    // Time transfer out for the first iteration 
    if (first) {
      int txFromDevTimerHandle = Timer::Start();
      #pragma offload target(MIC:micdev) \
        nocopy(A:length(matrix_elements)  alloc_if(0) free_if(0)) \
        nocopy(B:length(matrix_elements)  alloc_if(0) free_if(0)) \
        out(C:length(matrix_elements) alloc_if(0) free_if(0))
      {
      }
      transfer_time += Timer::Stop(txFromDevTimerHandle, "tx from dev");
      first = false;
    }


            // Calculate GFLOPS
            double blas_gflops = 2. * m * n * k / blas_time / 1e9;
            double pcie_gflops = 2. * m * n * k / (blas_time + transfer_time)
                / 1e9;
            resultDB.AddResult(testName+"-"+transb, toString(dim), "GFlops",
                    blas_gflops);
            resultDB.AddResult(testName+"-"+transb+"_PCIe", toString(dim),
                    "GFlops", pcie_gflops);
            resultDB.AddResult(testName+"-"+transb+"_Parity", toString(dim),
                    "N", transfer_time / blas_time);
        }
    }

    // Clean Up MIC storage
    #pragma offload target(MIC:micdev) \
        in(A:length(matrix_elements)  alloc_if(0)) \
        in(B:length(matrix_elements)  alloc_if(0)) \
        in(C:length(matrix_elements)  alloc_if(0)) \
        out(C:length(matrix_elements) alloc_if(0))
    {
    }

    // Clean up Host storage
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
}
