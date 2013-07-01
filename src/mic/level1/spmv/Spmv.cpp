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

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mkl_types.h"
#include "mkl_spblas.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"
#include "Spmv/util.h"

using namespace std; 

enum spmv_target { use_cpu, use_mkl, use_mic, use_mkl_mic };
char *target_str[] = { "CPU", "MKL", "MIC", "MKL_MIC" };

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Lukasz Wesolowski
// Creation: June 21, 2010
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("iterations", OPT_INT, "100", "Number of SpMV iterations "
                 "per pass"); 
    op.addOption("mm_filename", OPT_STRING, "random", "Name of file "
                 "which stores the matrix in Matrix Market format"); 
    op.addOption("maxval", OPT_FLOAT, "10", "Maximum value for random "
                 "matrices");
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose: 
//   Runs sparse matrix vector multiplication on the CPU 
//
// Arguements: 
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element of A
//   rowDelimiters: array of size dim+1 holding indices to rows of A; 
//                  last element is the index one past the last
//                  element of A
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: input - buffer of size dim
//        output - result from the spmv calculation 
// 
// Programmer: Lukasz Wesolowski
// Creation: June 23, 2010
// Returns:
//   nothing directly
//   out indirectly through a pointer
// ****************************************************************************
template <typename floatType>
void spmvCpu(const floatType *val, const int *cols, const int *rowDelimiters, 
         const floatType *vec, int dim, floatType *out) 
{

    for (int i=0; i<dim; i++) 
    {
        floatType t = 0; 
        for (int j=rowDelimiters[i]; j<rowDelimiters[i+1]; j++) 
        {
            int col = cols[j]; 
            t += val[j] * vec[col];
        }    
        out[i] = t; 
    }

}

// *******************************************************************
// Function: spmvMic
//
// Purpose:
//   Runs sparse matrix vector multiplication on the MIC accelerator
// *******************************************************************

template <typename floatType>
__declspec(target(mic)) void spmvMic(const floatType *val, const int *cols,
        const int *rowDelimiters, const floatType *vec, int dim, 
        floatType *out) 
{
    #pragma omp parallel for
    #pragma ivdep
    for (int i=0; i<dim; i++) 
    {
        floatType t = 0; 
        for (int j=rowDelimiters[i]; j<rowDelimiters[i+1]; j++) 
        {
            int col = cols[j]; 
            t += val[j] * vec[col];
        }    
        out[i] = t; 
    }

}

// *******************************************************************
// Function: spmvMkl
//
// Purpose:
//   Runs sparse matrix vector multiplication using MKL
// *******************************************************************

template<typename floatType=float>
__declspec(target(mic)) void spmvMkl(float *val, int *cols, int *rowDelimiters, 
         float *vec, int dim, float *out)
{
    char t='n';
    mkl_cspblas_scsrgemv(&t, &dim, val, rowDelimiters, cols, vec, out);
}
template<typename floatType=double>
__declspec(target(mic)) void spmvMkl(double *val, int *cols, int *rowDelimiters, 
         double *vec, int dim, double *out) 
{
    char t='n';
    mkl_cspblas_dcsrgemv(&t, &dim, val, rowDelimiters, cols, vec, out);
}

// ****************************************************************************
// Function: verifyResults
// 
// Purpose: 
//   Verifies correctness of GPU results by comparing to CPU results
//
// Arguments: 
//   cpuResults: array holding the CPU result vector
//   gpuResults: array hodling the GPU result vector
//   size: number of elements per vector
//   pass: optional iteration number
// 
// Programmer: Lukasz Wesolowski
// Creation: June 23, 2010
// Returns:
//   nothing
//   prints "Passed" if the vectors agree within a relative error of
//   MAX_RELATIVE_ERROR and "FAILED" if they are different
// ****************************************************************************
template <typename floatType>
bool verifyResults(const floatType *cpuResults, const floatType *gpuResults, 
                   const int size, const int pass = -1) 
{

    bool passed = true; 
    for (int i=0; i<size; i++) 
    {
        if (fabs(cpuResults[i] - gpuResults[i]) / cpuResults[i] 
            > MAX_RELATIVE_ERROR) 
        {
#ifdef VERBOSE_OUTPUT
           cout << "Mismatch at i: "<< i << " ref: " << cpuResults[i] << 
                " dev: " << gpuResults[i] << endl;
#endif
            passed = false; 
        }
    }

    if (pass != -1) 
    {
        cout << "Pass "<<pass<<": ";
    }
    if (passed) 
    {
        cout << "Passed" << endl;
    }
    else 
    {
        cout << "---FAILED---" << endl;
    }
    return passed;
}

// ****************************************************************************
// Function: RunTest
//
// Purpose:
//   Executes a run of the sparse matrix - vector multiplication benchmark
//   in either single or double precision
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: stores results from the benchmark
//   op: the options parser / parameter database
//   compileFlags: used to specify either single or double precision floats
//   nRows: number of rows in generated matrix
//
// Returns:  nothing
//
// Modifications:
//
// ****************************************************************************
template <typename floatType> 
void RunTest( ResultDatabase &resultDB, OptionParser &op, enum spmv_target 
        target, int nRows=0) 
{
    // Host data structures
    // array of values in the sparse matrix
    __declspec(target(mic)) static floatType *h_val, *h_valPad;
    // array of column indices for each value in h_val
    __declspec(target(mic)) static int *h_cols, *h_colsPad;       
    // array of indices to the start of each row in h_val/valPad
    __declspec(target(mic)) static int *h_rowDelimiters, *h_rowDelimitersPad;
    // Dense vector of values
    __declspec(target(mic)) static floatType *h_vec;
    // Output vector
    __declspec(target(mic)) static floatType *h_out;
    // Reference solution computed by cpu
    floatType *refOut;

    // Number of non-zero elements in the matrix
    __declspec(target(mic)) static int nItems;
    __declspec(target(mic)) static int nItemsPadded;
    __declspec(target(mic)) static int numRows;

    // This benchmark either reads in a matrix market input file or
    // generates a random matrix
    string inFileName = op.getOptionString("mm_filename");
    if (inFileName == "random")
    {
        // If we're not opening a file, the dimension of the matrix
        // has been passed in as an argument
        numRows = nRows; 
        nItems = numRows * numRows / 100; // 1% of entries will be non-zero
        float maxval = op.getOptionFloat("maxval"); 
        h_val = pmsAllocHostBuffer<floatType>(nItems);
        h_cols = pmsAllocHostBuffer<int>(nItems);
        h_rowDelimiters = pmsAllocHostBuffer<int>(nRows+1); 
        fill(h_val, nItems, maxval); 
        initRandomMatrix(h_cols, h_rowDelimiters, nItems, numRows); 
    }
    else 
    {   char filename[FIELD_LENGTH];
        strcpy(filename, inFileName.c_str());
        readMatrix(filename, &h_val, &h_cols, &h_rowDelimiters,
                &nItems, &numRows);
    }

    // Set up remaining host data
    h_vec = pmsAllocHostBuffer<floatType>(numRows);
    refOut = pmsAllocHostBuffer<floatType>(numRows);
    h_rowDelimitersPad = pmsAllocHostBuffer<int>(numRows+1);
    fill(h_vec, numRows, op.getOptionFloat("maxval")); 

    // Set up the padded data structures
    int paddedSize = numRows + (PAD_FACTOR - numRows % PAD_FACTOR);
    h_out = pmsAllocHostBuffer<floatType>(paddedSize);
    convertToPadded(h_val, h_cols, numRows, h_rowDelimiters, &h_valPad,
            &h_colsPad, h_rowDelimitersPad, &nItemsPadded);
    
    // Compute reference solution
    spmvCpu(h_val, h_cols, h_rowDelimiters, h_vec, numRows, refOut);

    cout << target_str[target] << " Test\n";
    int micdev = op.getOptionInt("target"); 

    int passes = op.getOptionInt("passes");
    int iters  = op.getOptionInt("iterations");

    for (int k = 0; k < passes; k++)
    {
        double iTransferTime, oTransferTime, totalKernelTime;
        int txToDevTimerHandle;
        int txFromDevTimerHandle;
        int kernelTimerHandle;

        switch (target) {
        case use_mic:
            // Warm up MIC device
            #pragma offload target(mic:micdev) in(k)
            { }
            #pragma offload target(mic:micdev) \
                        in(h_cols:length(nItems)             free_if(0)) \
                        in(h_rowDelimiters:length(numRows+1) free_if(0)) \
                        in(h_vec:length(numRows)             free_if(0)) \
                        in(h_val:length(nItems)              free_if(0)) \
                        in(h_out:length(numRows)             free_if(0))
            { }


            txToDevTimerHandle = Timer::Start();
            #pragma offload target(mic:micdev) \
                in(h_cols:length(nItems)              alloc_if(0) free_if(0)) \
                in(h_rowDelimiters:length(numRows+1)  alloc_if(0) free_if(0)) \
                in(h_vec:length(numRows)              alloc_if(0) free_if(0)) \
                in(h_val:length(nItems)               alloc_if(0) free_if(0)) \
                in(h_out:length(numRows)              alloc_if(0) free_if(0))
                { }
            iTransferTime = Timer::Stop(txToDevTimerHandle, "tx to dev");

            kernelTimerHandle = Timer::Start();
            #pragma offload target(mic:micdev) in(numRows, iters) \
            nocopy(h_cols:length(nItems)             alloc_if(0) free_if(0)) \
            nocopy(h_rowDelimiters:length(numRows+1) alloc_if(0) free_if(0)) \
            nocopy(h_vec:length(numRows)             alloc_if(0) free_if(0)) \
            nocopy(h_val:length(nItems)              alloc_if(0) free_if(0)) \
            nocopy(h_out:length(numRows)             alloc_if(0) free_if(0))
            for (int i=0; i<iters; i++) 
            {
                spmvMic(h_val, h_cols, h_rowDelimiters, h_vec, numRows, h_out);
            }
            totalKernelTime = Timer::Stop(kernelTimerHandle, "spmv");

            txFromDevTimerHandle = Timer::Start();
            #pragma offload target(mic:micdev) \
              nocopy(h_cols:length(nItems)             alloc_if(0) free_if(1)) \
              nocopy(h_rowDelimiters:length(numRows+1) alloc_if(0) free_if(1)) \
              nocopy(h_vec:length(numRows)             alloc_if(0) free_if(1)) \
              nocopy(h_val:length(nItems)              alloc_if(0) free_if(1)) \
              out(h_out:length(numRows)                alloc_if(0) free_if(1)) 
            { }
            oTransferTime = Timer::Stop(txFromDevTimerHandle, "tx from dev");
            break;

        case use_cpu:
            kernelTimerHandle = Timer::Start();
            for (int i=0; i<iters; i++) 
            {
                spmvCpu(h_val, h_cols, h_rowDelimiters, h_vec, numRows, h_out);
            }
            totalKernelTime = Timer::Stop(kernelTimerHandle, "spmv");
            iTransferTime = oTransferTime = 0;
        break;

        case use_mkl:
            kernelTimerHandle = Timer::Start();
            for (int i=0; i<iters; i++) 
            {
                    spmvMkl(h_val, h_cols, h_rowDelimiters, h_vec, numRows, h_out);
            }
            totalKernelTime = Timer::Stop(kernelTimerHandle, "spmv");
            iTransferTime = oTransferTime = 0;
        break;

        case use_mkl_mic:
            // Warm up MIC device
            #pragma offload target(mic:micdev) in(k)
            { }

            #pragma offload target(mic:micdev) \
                in(h_cols:length(nItems)             free_if(0)) \
                in(h_rowDelimiters:length(numRows+1) free_if(0)) \
                in(h_vec:length(numRows)             free_if(0)) \
                in(h_val:length(nItems)              free_if(0)) \
                in(h_out:length(numRows)             free_if(0))
            { }

            txToDevTimerHandle = Timer::Start();
            #pragma offload target(mic:micdev) \
            in(h_cols:length(nItems)              alloc_if(0)  free_if(0)) \
            in(h_rowDelimiters:length(numRows+1)  alloc_if(0)  free_if(0)) \
            in(h_vec:length(numRows)              alloc_if(0)  free_if(0)) \
            in(h_val:length(nItems)               alloc_if(0)  free_if(0)) \
            in(h_out:length(numRows)              alloc_if(0)  free_if(0))
            { }
            iTransferTime = Timer::Stop(txToDevTimerHandle, "tx to dev");

            kernelTimerHandle = Timer::Start();
            #pragma offload target(mic:micdev) \
            in(numRows, iters) \
            nocopy(h_cols:length(nItems)             alloc_if(0) free_if(0)) \
            nocopy(h_rowDelimiters:length(numRows+1) alloc_if(0) free_if(0)) \
            nocopy(h_vec:length(numRows)             alloc_if(0) free_if(0)) \
            nocopy(h_val:length(nItems)              alloc_if(0) free_if(0)) \
            nocopy(h_out:length(numRows)             alloc_if(0) free_if(0))
            for (int i=0; i<iters; i++) 
            {
                    spmvMkl(h_val, h_cols, h_rowDelimiters, h_vec, numRows, h_out);
            }
            totalKernelTime = Timer::Stop(kernelTimerHandle, "spmv");

            txFromDevTimerHandle = Timer::Start();
            #pragma offload target(mic:micdev) \
            nocopy(h_cols:length(nItems)             alloc_if(0) free_if(0)) \
            nocopy(h_rowDelimiters:length(numRows+1) alloc_if(0) free_if(0)) \
            nocopy(h_vec:length(numRows)             alloc_if(0) free_if(0)) \
            nocopy(h_val:length(nItems)              alloc_if(0) free_if(0)) \
            out(h_out:length(numRows)                alloc_if(0) free_if(0))
            { }
            oTransferTime = Timer::Stop(txFromDevTimerHandle, "to from dev");

	    #pragma offload target(mic:micdev) \
	    nocopy(h_cols:length(nItems)             alloc_if(0) free_if(1)) \
            nocopy(h_rowDelimiters:length(numRows+1) alloc_if(0) free_if(1)) \
	    nocopy(h_vec:length(numRows)             alloc_if(0) free_if(1)) \
       	    nocopy(h_val:length(nItems)              alloc_if(0) free_if(1)) \
            nocopy(h_out:length(numRows)                alloc_if(0) free_if(1))
            { }
            break;
        }

        verifyResults(refOut, h_out, numRows, k);

        // Store results in the DB
        char atts[TEMP_BUFFER_SIZE];
        char benchName[TEMP_BUFFER_SIZE];
        double avgTime = totalKernelTime / (double)iters;
        sprintf(atts, "%d_elements_%d_rows", nItems, numRows);
        double gflop = 2 * (double) nItems / 1e9;
        bool dpTest = (sizeof(floatType) == sizeof(double));
        sprintf(benchName, "%s-%s", target_str[target], dpTest ? "DP":"SP");
        resultDB.AddResult(benchName, atts, "Gflop/s", gflop/avgTime);
        sprintf(benchName, "%s_PCIe", benchName);
        resultDB.AddResult(benchName, atts, "Gflop/s", gflop / 
            (avgTime + iTransferTime + oTransferTime));
    }

    pmsFreeHostBuffer(h_val);
    pmsFreeHostBuffer(h_cols);
    pmsFreeHostBuffer(h_rowDelimiters);
    pmsFreeHostBuffer(h_vec);
    pmsFreeHostBuffer(h_out);
    pmsFreeHostBuffer(h_valPad);
    pmsFreeHostBuffer(h_colsPad);
    pmsFreeHostBuffer(h_rowDelimitersPad);
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the sparse matrix - vector multiplication benchmark
//
// Arguments:
//   resultDB: stores results from the benchmark
//   op: the options parser / parameter database
//
// Returns:  nothing
// Programmer: Lukasz Wesolowski
// Creation: August 13, 2010
//
// Modifications:
//
// ****************************************************************************
void
RunBenchmark( OptionParser &op, ResultDatabase &resultDB)
{
    // Create list of problem sizes
    int probSizes[4] = {1024, 8192, 12288, 16384};
    int sizeClass = op.getOptionInt("size") - 1; 

    cout << "Single precision tests:\n"; 

    RunTest<float> (resultDB, op, use_mkl, probSizes[sizeClass]);
    RunTest<float> (resultDB, op, use_mkl_mic, probSizes[sizeClass]);

    cout << "Double precision tests:\n"; 
    RunTest<double> (resultDB, op, use_mkl, probSizes[sizeClass]);
    RunTest<double> (resultDB, op, use_mkl_mic, probSizes[sizeClass]);
}
