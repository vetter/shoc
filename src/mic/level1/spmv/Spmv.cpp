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
#include <cassert>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mkl_types.h"
#include "mkl_spblas.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"
#include "Spmv/util.h"

#include "offload.h"

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
    op.addOption("seed", OPT_INT, "24115438", "Seed for PRNG");
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose: 
//   Compute Ma = b on a single core of the CPU.
//   M is assumed to be square.
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
void
spmvCpu(const floatType* val,
        const int* cols,
        const int* rowDelimiters, 
        const floatType* vec,
        int dim,
        floatType* out) 
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
__declspec(target(mic))
void
spmvMic(const floatType* val,
        const int* cols,
        const int* rowDelimiters,
        const floatType* vec,
        int dim, 
        floatType* out) 
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
//
// Compute u = Av on the MIC accelerator, where A is square,
// sparse, and expressed in ELLPACK-R format.
// ELLPACK-R is described in Vazquez, Ortega, Fernandez, Garzon,
// "Improving the performance of the sparse matrix vector product with 
// GPUs," 10th International IEEE Conference on Computer and Information
// Technology (CIT 2010).
//
// Two major differences between ELLPACK-R and CSR is that the array vals
// of non-zeros is in column-major order, and is padded so that it is
// of size nRows x maxRowLength, where maxRowLength is the maximum
// of the number of non-zeros per row from the original matrix A.
// *******************************************************************
template<typename floatType>
__declspec(target(mic))
void
spmvMicELLPACKR( const floatType* vals,     // non-zeros from A (ELLPACK-R fmt)
                    const int* rowLengths,  // number of non-zeros in each A row
                    const int* colIndices,  // column index of each non-zero
                    const floatType* vec,   // vector v to be multiplied with A
                    int dim,                // number of rows (and columns) in A
                    floatType* result )     // vector u, result of Av
{
    #pragma omp parallel for
    #pragma ivdep
    for( unsigned int r = 0; r < dim; r++ )
    {
        floatType rowDotProduct = 0.0;
        int currRowLength = rowLengths[r];

        // for each non-zero value in the row
        for( unsigned int c = 0; c < currRowLength; c++ )
        {
            // add its contribution to the row's dot product
            floatType currAValue = vals[r + c*dim];
            unsigned int currColIdx = colIndices[r + c*dim];
            rowDotProduct += currAValue * vec[currColIdx];
        }

        result[r] = rowDotProduct;
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



//----------------------------------------------------------------------------
// Perform sparse matrix-vector multiplication on MIC.
// Inner loop is executed in parallel, in contrast to spmvMic where inner loop
// is executed sequentially by each thread.
//----------------------------------------------------------------------------

template <typename floatType>
__declspec(target(mic)) 
void
spmvMicVector(const floatType* mtxVals, 
                const int* cols,
                const int* rowDelimiters, 
                const floatType* vec,
                int nNonZeros, 
                floatType* res,
                int micdev ) 
{
//    int maxThreads = omp_get_max_threads_target( TARGET_MIC, micdev );
    int maxThreads = omp_get_max_threads();
    int redThreads = 4; // TODO change to not be hardcoded
    int outerThreads = maxThreads / redThreads;

    if( (outerThreads == 0) || (redThreads > maxThreads) )
    {
        // We have a small number of threads to work with.
        // Fall back to the "serial" case where inner loop is computed
        // by one thread.
        outerThreads = maxThreads;
        redThreads = 1;
    }

    if( omp_get_thread_num() == 0 )
    {
        printf( "vector kernel using %d outer threads, %d inner threads\n",
            outerThreads, redThreads );
    }

    // set the OpenMP runtime to *not* use dynamic thread provisioning
    int dynSaved = omp_get_dynamic();
    omp_set_dynamic(0);

    #pragma omp parallel num_threads(outerThreads)
    for( int i = 0; i < nNonZeros; i++ )
    {
        floatType rowRes = 0; 
        #pragma omp parallel for \
            num_threads(redThreads) \
            reduction(+:rowRes)
        // #pragma ivdep
        for( int j = rowDelimiters[i]; j < rowDelimiters[i+1]; j++ )
        {
            int col = cols[j]; 
            rowRes = rowRes + (mtxVals[j] * vec[col]);
        }    
        res[i] = rowRes; 
    }

    // restore the OpenMP runtime's setting for dynamic thread provisioning
    omp_set_dynamic(dynSaved);    
}


//----------------------------------------------------------------------------
// Zero the output array (on the device,
// so we can be sure it is computing 
// something on each pass)
//----------------------------------------------------------------------------
template<typename floatType>
__declspec(target(mic))
void
zero( floatType* h_out, int numRows )
{
    #pragma omp parallel for
    for( unsigned int i = 0; i < numRows; i++ )
    {
        h_out[i] = 0;
    }
}


//----------------------------------------------------------------------------
// Measure performance of sparse matrix-vector multiply,
// with matrix stored in CSR form (possibly padded).
//----------------------------------------------------------------------------
template<typename floatType>
void
csrTest( ResultDatabase& resultDB,
            OptionParser& op,
            floatType* h_val,
            int* h_cols,
            int* h_rowDelimiters,
            floatType* h_vec,
            floatType* h_out,
            int numRows,
            int numNonZeros,
            floatType* refOut,
            bool padded )
{
    int nPasses = op.getOptionInt( "passes" );
    int nIters = op.getOptionInt( "iterations" );
    int micdev = op.getOptionInt( "device" );

    // Results description
    std::ostringstream attstr;
    attstr << numNonZeros << "_elements_" << numRows << "_rows";
    double gflop = 2 * (double)numNonZeros / 1.0e9;
    std::string prefix = (padded ? "Padded_" : "");
    std::string suffix = (sizeof(floatType) == sizeof(float)) ? "-SP" : "-DP";

    // transfer data to device
    int txToDevTimerHandle = Timer::Start();
    #pragma offload_transfer \
        target(mic:micdev) \
        in( h_val:length(numNonZeros) alloc_if(1) free_if(0) ) \
        in( h_cols:length(numNonZeros) alloc_if(1) free_if(0) ) \
        in( h_rowDelimiters:length(numRows+1) alloc_if(1) free_if(0) ) \
        in( h_vec:length(numRows) alloc_if(1) free_if(0) ) \
        nocopy( h_out:length(numRows) alloc_if(1) free_if(0) )
    double iTransferTime = Timer::Stop( txToDevTimerHandle, "tx to dev" );

    // Do as many passes as desired
    for( int p = 0; p < nPasses; p++ )
    {
        // run the scalar kernel
        int kernelTimerHandle = Timer::Start();
        #pragma offload \
            target(mic:micdev) \
            nocopy( h_val:length(numNonZeros) alloc_if(0) free_if(0) ) \
            nocopy( h_cols:length(numNonZeros) alloc_if(0) free_if(0) ) \
            nocopy( h_rowDelimiters:length(numRows+1) alloc_if(0) free_if(0) ) \
            nocopy( h_vec:length(numRows) alloc_if(0) free_if(0) ) \
            nocopy( h_out:length(numRows) alloc_if(0) free_if(0) )
        for( int i = 0; i < nIters; i++ )
        {
            spmvMic( h_val, h_cols, h_rowDelimiters, h_vec, numRows, h_out );
        }
        double scalarKernelTime = Timer::Stop( kernelTimerHandle, "kernel timer" );

        // Transfer data back to the host.
        int txFromDevTimerHandle = Timer::Start();
        #pragma offload_transfer \
            target(mic:micdev) \
            out( h_out:length(numRows) alloc_if(0) free_if(0) )
        double oTransferTime = Timer::Stop( txFromDevTimerHandle, "tx from dev" );
        
        // Compare the device result to the reference result.
        if( verifyResults( refOut, h_out, numRows, p ) )
        {
            // Results match - the device computed a result equivalent to
            // the host.
            //
            // Record the average performance of for one iteration.
            scalarKernelTime = (scalarKernelTime / (double)nIters) * 1.e-3;
            std::string testName = prefix+"CSR-Scalar"+suffix;
            double totalTransfer = iTransferTime + oTransferTime;

            resultDB.AddResult( testName,
                                attstr.str().c_str(),
                                "Gflop/s",
                                gflop/(scalarKernelTime) );
            resultDB.AddResult( testName+"_PCIe",
                                attstr.str().c_str(),
                                "Gflop/s",
                                gflop / (scalarKernelTime + totalTransfer) );
        }
        else
        {
            // Results do not match.
            // Don't report performance, and don't continue to run tests.
            return;
        }
    }
    #pragma offload \
        target(mic:micdev) \
        nocopy( h_out:length(numRows) alloc_if(0) free_if(0) )
    zero<floatType>( h_out, numRows );

#if READY
    std::cout << "CSR Vector Kernel\n";
    for( int p = 0; p < nPasses; p++ )
    {
        // run the vector kernel
        int kernelTimerHandle = Timer::Start();
        #pragma offload \
            target(mic:micdev) \
            nocopy( h_val:length(numNonZeros) alloc_if(0) free_if(0) ) \
            nocopy( h_cols:length(numNonZeros) alloc_if(0) free_if(0) ) \
            nocopy( h_rowDelimiters:length(numRows+1) alloc_if(0) free_if(0) ) \
            nocopy( h_vec:length(numRows) alloc_if(0) free_if(0) ) \
            nocopy( h_out:length(numRows) alloc_if(0) free_if(0) )
        for( int i = 0; i < nIters; i++ )
        {
            spmvMicVector( h_val, h_cols, h_rowDelimiters, h_vec, numRows, h_out, micdev );
        }
        double vectorKernelTime = Timer::Stop( kernelTimerHandle, "kernel timer" );

        // Transfer data back to the host.
        int txFromDevTimerHandle = Timer::Start();
        #pragma offload_transfer \
            target(mic:micdev) \
            out( h_out:length(numRows) alloc_if(0) free_if(0) )
        double oTransferTime = Timer::Stop( txFromDevTimerHandle, "tx from dev" );
        
        // Compare the device result to the reference result.
        if( verifyResults( refOut, h_out, numRows, p ) )
        {
            // Results match - the device computed a result equivalent to
            // the host.
            //
            // Record the average performance of for one iteration.
            vectorKernelTime = (vectorKernelTime / (double)nIters) * 1.e-3;
            std::string testName = prefix+"CSR-Vector"+suffix;
            double totalTransfer = iTransferTime + oTransferTime;

            resultDB.AddResult( testName,
                                attstr.str().c_str(),
                                "Gflop/s",
                                gflop/(vectorKernelTime) );
            resultDB.AddResult( testName+"_PCIe",
                                attstr.str().c_str(),
                                "Gflop/s",
                                gflop / (vectorKernelTime + totalTransfer) );
        }
        else
        {
            // Results do not match.
            // Don't report performance, and don't continue to run tests.
            return;
        }
    }
#endif // READY

    // release the data from the device
    #pragma offload_transfer \
        target(mic:micdev) \
        nocopy( h_val:length(numNonZeros) alloc_if(0) free_if(1) ) \
        nocopy( h_cols:length(numNonZeros) alloc_if(0) free_if(1) ) \
        nocopy( h_rowDelimiters:length(numRows+1) alloc_if(0) free_if(1) ) \
        nocopy( h_vec:length(numRows) alloc_if(0) free_if(1) ) \
        nocopy( h_out:length(numRows) alloc_if(0) free_if(1) )
}


//----------------------------------------------------------------------------
// Measure performance of sparse matrix vector multiply (u = Av).
// with matrix A stored in ELLPACK-R form.
// See header comment to the spmvMicELLPACKR function for more
// information about ELLPACKR.
//----------------------------------------------------------------------------
template<typename floatType>
void
ellpackrTest( ResultDatabase& resultDB,
            OptionParser& op,
            floatType* h_vals,
            int* h_colIndices,
            int* h_rowLengths,
            int maxRowLength,
            floatType* h_vec,
            floatType* h_out,
            int numRows,
            int numNonZeros,
            floatType* refOut )
{
    int nPasses = op.getOptionInt( "passes" );
    int nIters = op.getOptionInt( "iterations" );
    int micdev = op.getOptionInt( "device" );

    // Results description
    std::ostringstream attstr;
    attstr << numNonZeros << "_elements_" << numRows << "_rows";
    double gflop = 2 * (double)numNonZeros / 1.0e9;
    std::string suffix = (sizeof(floatType) == sizeof(float)) ? "-SP" : "-DP";

    // transfer data to device
    int txToDevTimerHandle = Timer::Start();
    #pragma offload_transfer \
        target(mic:micdev) \
        in( h_vals:length(numRows*maxRowLength) alloc_if(1) free_if(0) ) \
        in( h_colIndices:length(numRows*maxRowLength) alloc_if(1) free_if(0) ) \
        in( h_rowLengths:length(numRows) alloc_if(1) free_if(0) ) \
        in( h_vec:length(numRows) alloc_if(1) free_if(0) ) \
        nocopy( h_out:length(numRows) alloc_if(1) free_if(0) )
    double iTransferTime = Timer::Stop( txToDevTimerHandle, "tx to dev" );

    // Do as many passes as desired
    for( int p = 0; p < nPasses; p++ )
    {
        // run the SpMV kernel using the matrix in ELLPACK-R form
        int kernelTimerHandle = Timer::Start();
        #pragma offload \
            target(mic:micdev) \
            nocopy( h_vals:length(numRows*maxRowLength) alloc_if(0) free_if(0) ) \
            nocopy( h_colIndices:length(numRows*maxRowLength) alloc_if(0) free_if(0) ) \
            nocopy( h_rowLengths:length(numRows) alloc_if(0) free_if(0) ) \
            nocopy( h_vec:length(numRows) alloc_if(0) free_if(0) ) \
            nocopy( h_out:length(numRows) alloc_if(0) free_if(0) )
        for( int i = 0; i < nIters; i++ )
        {
            spmvMicELLPACKR( h_vals,
                                h_rowLengths,
                                h_colIndices,
                                h_vec,
                                numRows,
                                h_out );
        }
        double scalarKernelTime = Timer::Stop( kernelTimerHandle, "kernel timer" );

        // Transfer data back to the host.
        int txFromDevTimerHandle = Timer::Start();
        #pragma offload_transfer \
            target(mic:micdev) \
            out( h_out:length(numRows) alloc_if(0) free_if(0) )
        double oTransferTime = Timer::Stop( txFromDevTimerHandle, "tx from dev" );
        
        // Compare the device result to the reference result.
        if( verifyResults( refOut, h_out, numRows, p ) )
        {
            // Results match - the device computed a result equivalent to
            // the host.
            //
            // Record the average performance of for one iteration.
            scalarKernelTime = (scalarKernelTime / (double)nIters) * 1.e-3;
            std::string testName = "ELLPACKR"+suffix;
            double totalTransfer = iTransferTime + oTransferTime;

            resultDB.AddResult( testName,
                                attstr.str().c_str(),
                                "Gflop/s",
                                gflop/(scalarKernelTime) );
            resultDB.AddResult( testName+"_PCIe",
                                attstr.str().c_str(),
                                "Gflop/s",
                                gflop / (scalarKernelTime + totalTransfer) );
        }
        else
        {
            // Results do not match.
            // Don't report performance, and don't continue to run tests.
            return;
        }
    }

    // release the data from the device
    #pragma offload_transfer \
        target(mic:micdev) \
        nocopy( h_vals:length(numRows*maxRowLength) alloc_if(0) free_if(1) ) \
        nocopy( h_colIndices:length(numRows*maxRowLength) alloc_if(0) free_if(1) ) \
        nocopy( h_rowLengths:length(numRows) alloc_if(0) free_if(1) ) \
        nocopy( h_vec:length(numRows) alloc_if(0) free_if(1) ) \
        nocopy( h_out:length(numRows) alloc_if(0) free_if(1) )
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
// Function: RunTestMICMKL
//
// Purpose:
//   Computes Ma = b, where M is a sparse matrix, and a and b are vectors.
//   M is assumed to be square.
//   Templated to support both single and double-precision floating point
//   values, and can compute on CPU or MIC, with MKL or "by hand."
//
// Arguments:
//   resultDB: stores results from the benchmark
//   micdev: the MIC device id to use for the benchmark
//   h_val: non-zero values from M
//   h_cols: column indices for non-zero values
//   h_rowDelimiters: indices within h_val where each row starts
//   h_vec: the vector a
//   h_out: the vector b (output)
//   nItems: number non zero elements in h_val.
//   numRows: number of rows/colums in M; number of items in a and b.
//   refOut: the reference output (the result that device should compute)
//   target: approach to use to do the multiplication (on MIC, with MKL, etc.)
//
// Returns:  nothing
//
// Modifications:
//
// ****************************************************************************
template<class floatType>
void
RunTestMICMKL( ResultDatabase& resultDB,
            OptionParser& op,
            floatType* h_val,
            int* h_cols,
            int* h_rowDelimiters,
            floatType* h_vec,
            floatType* h_out,
            int numRows,
            int nItems,
            floatType* refOut,
            enum spmv_target target )
{
    int nPasses = op.getOptionInt( "passes" );
    int iters = op.getOptionInt( "iterations" );
    int micdev = op.getOptionInt( "device" );

    cout << target_str[target] << " Test\n";
    double iTransferTime, oTransferTime, totalKernelTime;
    int txToDevTimerHandle;
    int txFromDevTimerHandle;
    int kernelTimerHandle;

    for( int pass = 0; pass < nPasses; pass++ )
    {
        switch (target) {
        case use_mic:
            // Warm up MIC device
            #pragma offload target(mic:micdev) in(pass)
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
            #pragma offload target(mic:micdev) in(pass)
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

        verifyResults(refOut, h_out, numRows, pass);

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
template<typename floatType> 
void
RunTest( ResultDatabase &resultDB,
            OptionParser &op, 
            int nRows=0) 
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

    // Seed the random number generator used to initialize the 
    // values in matrix A and vector v (we are computing u = Av).
    unsigned int rngSeed = (unsigned int)op.getOptionInt("seed");
    InitRNG( rngSeed );

    // Obtain a square, sparse matrix A in CSR format.
    // We can read the matrix from a MatrixMarket input file,
    // or generate a random matrix.
    std::string inFileName = op.getOptionString( "mm_filename" );
    if( inFileName != "random" )
    {
        int numCols = -1;

        // We have been asked to read the matrix A from a file.
        readMatrix( inFileName.c_str(),
                    &h_val,
                    &h_cols,
                    &h_rowDelimiters,
                    &nItems,
                    &numRows,
                    &numCols );
        if( numRows != numCols )
        {
            // We read a matrix that was not square, but we can only
            // work with square matrices.
            std::cerr << "This benchmark can only work with square matrices,\nbut file "
                << inFileName << " contains a non-square matrix "
                << "(nRows=" << numRows << ", nCols=" << numCols << ")."
                << std::endl;
            exit( 1 );
        }
        nRows = numRows;
    }
    else
    {
        // We are not using a matrix from a file.
        // Use the number of rows provided as an argument to this function,
        // and construct a square matrix A in CSR form with random values.
        numRows = nRows; 

        
        // determine the number of non-zeros in the matrix
        // Our target is 1% of the entries (with a minimum of 1) will be
        // non-zero.
        nItems = numRows * numRows / 100;
        if( nItems == 0 )
        {
            nItems = 1;
        }

        float maxval = op.getOptionFloat("maxval"); 
        h_val = pmsAllocHostBuffer<floatType>(nItems);
        h_cols = pmsAllocHostBuffer<int>(nItems);
        h_rowDelimiters = pmsAllocHostBuffer<int>(nRows+1); 
        initRandomVector(h_val, nItems, maxval); 
        initRandomMatrix(h_cols, h_rowDelimiters, nItems, numRows); 
    }

    // Build the matrix A in ELLPACK-R format.
    // See header comment at spmvMicELLPACKR function for more
    // information about ELLPACK-R format.
    //
    // First, we build the ELLPACK-R format's array of row lengths,
    // keeping track of the maximum row length as we do so.
    int* h_rowLengths = new int[numRows];
    int maxRowLength = 0;
    for( int r = 0; r < numRows; r++ )
    {
        h_rowLengths[r] = h_rowDelimiters[r+1] - h_rowDelimiters[r];
        if( h_rowLengths[r] > maxRowLength )
        {
            maxRowLength = h_rowLengths[r];
        }
    }
    assert( maxRowLength > 0 );

    // Next, construct the ELLPACK-R array of non-zeros.
    // This array is column-major and padded so that each row
    // has maxRowLength values.
    floatType* h_vals_ellpackr = new floatType[numRows * maxRowLength];
    int* h_col_indices_ellpackr = new int[numRows * maxRowLength];
    convertToColMajor( h_val,               // input: matrix in CSR format
                        h_cols,             // ""
                        numRows,            // ""
                        h_rowDelimiters,    // ""
                        h_vals_ellpackr,    // output: matrix in ELLPACK-R format
                        h_col_indices_ellpackr, // ""
                        h_rowLengths,       // ELLPACK-R row length array (already computed)
                        maxRowLength,       // max value from ELLPACK-R row length array
                        0 );                // CSR format is not padded
    // We now have the matrix A in ELLPACK-R format
    // h_vals_ellpackr is the numRows x maxRowLength array of non-zeros
    // h_col_indices_ellpackr is the numRows x maxRowLength array of column 
    //    indices associated with the items in h_vals_ellpackr
    // h_rowLengths is the array holding the number of non-zeros in each row


    // Set up remaining host data
    h_vec = pmsAllocHostBuffer<floatType>(numRows);
    refOut = pmsAllocHostBuffer<floatType>(numRows);
    h_rowDelimitersPad = pmsAllocHostBuffer<int>(numRows+1);
    initRandomVector(h_vec, numRows, op.getOptionFloat("maxval")); 

    // Set up the padded data structures
    int paddedSize = numRows + (PAD_FACTOR - numRows % PAD_FACTOR);
    h_out = pmsAllocHostBuffer<floatType>(paddedSize);
    convertToPadded(h_val, h_cols, numRows, h_rowDelimiters, &h_valPad,
            &h_colsPad, h_rowDelimitersPad, &nItemsPadded);
    
    // Compute reference solution
    spmvCpu(h_val, h_cols, h_rowDelimiters, h_vec, numRows, refOut);

    int micdev = op.getOptionInt("device"); 
    int passes = op.getOptionInt("passes");
    int iters  = op.getOptionInt("iterations");


    // The CSR "vector" implementation uses nested OpenMP constructs.
    // Tell the OpenMP implementation we want to use nested parallel regions.
    // NOTE: this doesn't necessarily mean that our "inner loops" will
    // be parallelized, since some OpenMP implementations that say they
    // support nesting will only use one thread for the inner loop.
    omp_set_nested_target( TARGET_MIC, micdev, 1 );

    // tests implemented in a way comparable to the Spmv benchmark
    // implementations for the other supported programming models
    std::cout << "CSR Test\n";
    csrTest<floatType>( resultDB,
                            op,
                            h_val,
                            h_cols,
                            h_rowDelimiters,
                            h_vec,
                            h_out,
                            numRows,
                            nItems,
                            refOut,
                            false );

    std::cout << "CSR Test -- Padded Data\n";
    csrTest<floatType>( resultDB,
                            op,
                            h_valPad,
                            h_colsPad,
                            h_rowDelimitersPad,
                            h_vec,
                            h_out,
                            numRows,
                            nItemsPadded,
                            refOut,
                            true );

    std::cout << "ELLPACK-R Test\n";
    ellpackrTest<floatType>( resultDB,
                            op,
                            h_vals_ellpackr,
                            h_col_indices_ellpackr,
                            h_rowLengths,
                            maxRowLength,
                            h_vec,
                            h_out,
                            numRows,
                            nItems,     // number of non-zeros
                            refOut );


    // Tests implemented to use MKL.  Not directly comparable
    // to implementations for other programming models, but
    // interesting to compare against those other implementations.
    // The 'use_mic' should give similar performance to
    // csrTest with unpadded data.
    RunTestMICMKL<floatType>( resultDB,
                            op,
                            h_val,
                            h_cols,
                            h_rowDelimiters,
                            h_vec,
                            h_out,
                            numRows,
                            nItems,
                            refOut,
                            use_mic );
    RunTestMICMKL<floatType>( resultDB,
                            op,
                            h_val,
                            h_cols,
                            h_rowDelimiters,
                            h_vec,
                            h_out,
                            numRows,
                            nItems,
                            refOut,
                            use_mkl_mic );
    RunTestMICMKL<floatType>( resultDB,
                            op,
                            h_val,
                            h_cols,
                            h_rowDelimiters,
                            h_vec,
                            h_out,
                            numRows,
                            nItems,
                            refOut,
                            use_cpu );
    RunTestMICMKL<floatType>( resultDB,
                            op,
                            h_val,
                            h_cols,
                            h_rowDelimiters,
                            h_vec,
                            h_out,
                            numRows,
                            nItems,
                            refOut,
                            use_mkl );

    // clean up
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
    RunTest<float> (resultDB, op, probSizes[sizeClass]);

    cout << "Double precision tests:\n"; 
    RunTest<double> (resultDB, op, probSizes[sizeClass]);
}

