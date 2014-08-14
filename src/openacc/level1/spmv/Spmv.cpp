#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"
#include "util.h"
#include "Spmv.h"

template <typename floatType>
void RunTest(ResultDatabase &resultDB, OptionParser &op, int nRows=0);

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
    for (int i = 0; i < size; i++)
    {
        if (fabs(cpuResults[i] - gpuResults[i]) / cpuResults[i] 
            > MAX_RELATIVE_ERROR) 
        {
//            cout << "Mismatch at i: "<< i << " ref: " << cpuResults[i] <<
//                " dev: " << gpuResults[i] << endl;
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

extern "C" void spmv_csr_scalar_kernel_float (const void * val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, const int nItems, void * vec, void * out,
                       const int niters, double * iTransferTime,
                       double * oTransferTime, double * kernelTime);

extern "C" void spmv_csr_scalar_kernel_double (const void * val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, const int nItems, void * vec, void * out,
                       const int niters, double * iTransferTime,
                       double * oTransferTime, double * kernelTime);

extern "C" void spmv_csr_vector_kernel_float (const void * val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, const int nItems, void * vec, void * out,
                       const int niters, double * iTransferTime,
                       double * oTransferTime, double * kernelTime);

extern "C" void spmv_csr_vector_kernel_double (const void * val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, const int nItems, void * vec, void * out,
                       const int niters, double * iTransferTime,
                       double * oTransferTime, double * kernelTime);

extern "C" void spmv_ellpackr_kernel_float (const void * val,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     const int dim, const int nItems, void * vec, void * out,
                     const int niters, double * iTransferTime,
                     double * oTransferTime, double * kernelTime);

extern "C" void spmv_ellpackr_kernel_double (const void * val,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     const int dim, const int nItems, void * vec, void * out,
                     const int niters, double * iTransferTime,
                     double * oTransferTime, double * kernelTime);

extern "C" void zero_float (void * a, const int size);

extern "C" void zero_double (void * a, const int size);


template <typename floatType>
void csrTest(ResultDatabase& resultDB, OptionParser& op, floatType* h_val,
        int* h_cols, int* h_rowDelimiters, floatType* h_vec, floatType* h_out,
        int numRows, int numNonZeroes, floatType* refOut, bool padded)
{
	void (*spmv_csr_scalar_kernel)(const void * ,  
                       const int    * __restrict__,
                       const int    * __restrict__,
                       const int, const int, void *, void *,
                       const int, double *,
                       double *, double *);

	void (*spmv_csr_vector_kernel)(const void * ,  
                       const int    * __restrict__,
                       const int    * __restrict__,
                       const int , const int, void *, void * ,
                       const int, double *,
                       double *, double *);

	//void (*zero)(void * , const int);


      string suffix;
      if (sizeof(floatType) == sizeof(float))
      {
          spmv_csr_scalar_kernel = spmv_csr_scalar_kernel_float;
          spmv_csr_vector_kernel = spmv_csr_vector_kernel_float;
          //zero = zero_float;
          suffix = "-SP";
      }
      else {
          spmv_csr_scalar_kernel = spmv_csr_scalar_kernel_double;
          spmv_csr_vector_kernel = spmv_csr_vector_kernel_double;
          //zero = zero_double;
          suffix = "-DP";
      }

      // Setup thread configuration
      int nBlocksScalar = (int) ceil((floatType) numRows / BLOCK_SIZE);
      int nBlocksVector = (int) ceil(numRows /
                  (floatType)(BLOCK_SIZE / WARP_SIZE));
      int passes = op.getOptionInt("passes");
      int iters  = op.getOptionInt("iterations");
      double iTransferTime, oTransferTime;
      double scalarKernelTime, vectorKernelTime;

      // Results description info
      char atts[TEMP_BUFFER_SIZE];
      sprintf(atts, "%d_elements_%d_rows", numNonZeroes, numRows);
      string prefix = "";
      prefix += (padded) ? "Padded_" : "";
      double gflop = 2 * (double) numNonZeroes / 1e9;
      cout << "CSR Scalar Kernel\n";
      for (int k=0; k<passes; k++)
      {
          // Run Scalar Kernel
          spmv_csr_scalar_kernel(h_val, h_cols, h_rowDelimiters, numRows, numNonZeroes, h_vec,
          h_out, iters, &iTransferTime, &oTransferTime, &scalarKernelTime);
          // Compare reference solution to GPU result
          if (! verifyResults(refOut, h_out, numRows, k))
          {
              return;  // If results don't match, don't report performance
          }
          scalarKernelTime = (scalarKernelTime / (double)iters);
          string testName = prefix+"CSR-Scalar"+suffix;
          double totalTransfer = iTransferTime + oTransferTime;
          resultDB.AddResult(testName, atts, "Gflop/s",
                  gflop/(scalarKernelTime));
          resultDB.AddResult(testName+"_PCIe", atts, "Gflop/s",
                            gflop / (scalarKernelTime+totalTransfer));
      }
      //DEBUG: We don't need this anymore.
      //zero(h_out, numRows);

      cout << "CSR Vector Kernel\n";
      for (int k=0; k<passes; k++)
      {
          // Run Vector Kernel
          spmv_csr_vector_kernel(h_val, h_cols, h_rowDelimiters, numRows, numNonZeroes, h_vec,
          h_out, iters, &iTransferTime, &oTransferTime, &vectorKernelTime);
          // Compare reference solution to GPU result
          if (! verifyResults(refOut, h_out, numRows, k))
          {
              return;  // If results don't match, don't report performance
          }
          vectorKernelTime = (vectorKernelTime / (double)iters);
          string testName = prefix+"CSR-Vector"+suffix;
          double totalTransfer = iTransferTime + oTransferTime;
          resultDB.AddResult(testName, atts, "Gflop/s", gflop/vectorKernelTime);
          resultDB.AddResult(testName+"_PCIe", atts, "Gflop/s",
                            gflop/(vectorKernelTime+totalTransfer));
      }
}

template <typename floatType>
void ellPackTest(ResultDatabase& resultDB, OptionParser& op, floatType* h_val,
        int* h_cols, int* h_rowDelimiters, floatType* h_vec, floatType* h_out,
        int numRows, int numNonZeroes, floatType* refOut, bool padded,
        int paddedSize)
{
	void (*spmv_ellpackr_kernel)(const void * __restrict__,
                     const int    * __restrict__,
                     const int    * __restrict__,
                     const int, void * __restrict__,
                     const int, double *,
                     double *, double *);
	void (*zero)(float * __restrict__, const int);

    int *h_rowLengths = (int *)malloc(paddedSize*sizeof(int)); 
    int maxrl = 0;
    for (int k=0; k<numRows; k++)
    {
        h_rowLengths[k] = h_rowDelimiters[k+1] - h_rowDelimiters[k];
        if (h_rowLengths[k] > maxrl)
        {
            maxrl = h_rowLengths[k];
        }
    }
    for (int p=numRows; p < paddedSize; p++)
    {
        h_rowLengths[p] = 0;
    }

    // Column major format host data structures
    int cmSize = padded ? paddedSize : numRows;
    floatType *h_valcm = (floatType *)malloc(maxrl*cmSize*sizeof(floatType));
    int *h_colscm = (floatType *)malloc(maxrl*cmSize*sizeof(int));
    convertToColMajor(h_val, h_cols, numRows, h_rowDelimiters, h_valcm,
                              h_colscm, h_rowLengths, maxrl, padded);

    // Transfer data to device
    // DEBUG: d_rowLengths may be different from h_rowLengths.
    //CUDA_SAFE_CALL(cudaMemcpy(d_rowLengths, h_rowLengths,
    //        cmSize * sizeof(int), cudaMemcpyHostToDevice));

    // Bind texture for position
    if (sizeof(floatType) == sizeof(float))
    {
        spmv_ellpackr_kernel = spmv_ellpackr_kernel_float;
    }
    else
    {
        spmv_ellpackr_kernel = spmv_ellpackr_kernel_double;
    }
    int nBlocks = (int) ceil((floatType) cmSize / BLOCK_SIZE);
    int passes = op.getOptionInt("passes");
    int iters  = op.getOptionInt("iterations");
    double iTransferTime, oTransferTime;
    double totalKernelTime;
    for (int k=0; k<passes; k++)
    {
        spmv_ellpackr_kernel(h_valcm, h_colscm, h_rowLengths, cmSize, numNonZeros, h_vec,
        h_out, iters, &iTransferTime, &oTransferTime, &totalKernelTime);

        // Compare reference solution to GPU result
        if (! verifyResults(refOut, h_out, numRows, k)) {
            return;
        }
        char atts[TEMP_BUFFER_SIZE];
        char benchName[TEMP_BUFFER_SIZE];
        double avgTime = totalKernelTime / (float)iters;
        sprintf(atts, "%d_elements_%d_rows", numNonZeroes, cmSize);
        double gflop = 2 * (double) numNonZeroes / 1e9;
        bool dpTest = (sizeof(floatType) == sizeof(double)) ? true : false;
        sprintf(benchName, "%sELLPACKR-%s", padded ? "Padded_":"",
                dpTest ? "DP":"SP");
        resultDB.AddResult(benchName, atts, "Gflop/s", gflop/avgTime);
    }
}

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
        for (int j = rowDelimiters[i]; j < rowDelimiters[i + 1]; j++)
        {
            int col = cols[j]; 
            t += val[j] * vec[col];
        }    
        out[i] = t; 
    }
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
//
// Programmer: Seyong Lee
// Creation: Jan. 30, 2013
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{

    int probSizes[4] = {1024, 8192, 12288, 16384};
    int sizeClass = op.getOptionInt("size") - 1;

    cout <<"Single precision tests:\n";
    RunTest<float>(resultDB, op, probSizes[sizeClass]);

    cout <<"Double precision tests:\n";
    RunTest<double>(resultDB, op, probSizes[sizeClass]);

/*
    std::cout << "Double precision not supported by chosen device, skipping" << std::endl;
    // driver script still needs entries for all tests, even if not run
    int nPasses = (int)op.getOptionInt( "passes" );
    for( unsigned int p = 0; p < nPasses; p++ )
    {
            resultDB.AddResult( (const char*)"CSR-Scalar-DP",
                                "N/A",
                                "Gflop/s",
                                FLT_MAX );
            resultDB.AddResult( (const char*)"CSR-Scalar-DP_PCIe",
                                "N/A",
                                "Gflop/s",
                                FLT_MAX );
            resultDB.AddResult( (const char*)"CSR-Vector-DP",
                                "N/A",
                                "Gflop/s",
                                FLT_MAX );
            resultDB.AddResult( (const char*)"CSR-Vector-DP_PCIe",
                                "N/A",
                                "Gflop/s",
                                FLT_MAX );
            resultDB.AddResult( (const char*)"ELLPACKR-DP",
                                "N/A",
                                "Gflop/s",
                                FLT_MAX );
            resultDB.AddResult( (const char*)"ELLPACKR-DP_PCIe",
                                "N/A",
                                "Gflop/s",
                                FLT_MAX );
            resultDB.AddResult( (const char*)"Padded_CSR-Scalar-DP",
                                "N/A",
                                "Gflop/s",
                                FLT_MAX );
            resultDB.AddResult( (const char*)"Padded_CSR-Scalar-DP_PCIe",
                                "N/A",
                                "Gflop/s",
                                FLT_MAX );
            resultDB.AddResult( (const char*)"Padded_CSR-Vector-DP",
                                "N/A",
                                "Gflop/s",
                                FLT_MAX );
            resultDB.AddResult( (const char*)"Padded_CSR-Vector-DP_PCIe",
                                "N/A",
                                "Gflop/s",
                                FLT_MAX );
    }
*/
}


// ****************************************************************************
// Function: RunTest
//
// Purpose:
//   Executes a run of the sparse matrix - vector multiplication benchmark
//   in either single or double precision
//
// Arguments:
//   resultDB: stores results from the benchmark
//   op: the options parser / parameter database
//   nRows: number of rows in generated matrix
//
// Returns:  nothing
//
// Programmer: Seyong Lee
// Creation: Jan. 30, 2013
//
// Modifications:
//
// ****************************************************************************
template <typename floatType>
void RunTest(ResultDatabase &resultDB, OptionParser &op, int nRows) 
{
    // Host data structures
    // Array of values in the sparse matrix
    floatType *h_val, *h_valPad;
    // Array of column indices for each value in h_val
    int *h_cols, *h_colsPad;
    // Array of indices to the start of each row in h_Val
    int *h_rowDelimiters, *h_rowDelimitersPad;
    // Dense vector and space for dev/cpu reference solution
    floatType *h_vec, *h_out, *refOut;
    // nItems = number of non zero elems
    int nItems, nItemsPadded, numRows;

    // This benchmark either reads in a matrix market input file or
    // generates a random matrix
    string inFileName = op.getOptionString("mm_filename");
    if (inFileName == "random")
    {
        numRows = nRows;
        nItems = numRows * numRows / 100; // 1% of entries will be non-zero
        float maxval = op.getOptionFloat("maxval"); 
		h_val = new floatType[nItems];
		h_cols = new int[nItems];
		h_rowDelimiters = new int[(numRows+1)];
        fill(h_val, nItems, maxval); 
        initRandomMatrix(h_cols, h_rowDelimiters, nItems, numRows);
    }
    else 
    {
        char filename[FIELD_LENGTH];
        strcpy(filename, inFileName.c_str());
        readMatrix(filename, &h_val, &h_cols, &h_rowDelimiters,
                &nItems, &numRows);
    }

    // Set up remaining host data
	h_vec = new floatType[numRows];
    refOut = new floatType[numRows];
	h_rowDelimitersPad = new int[(numRows+1)];
    fill(h_vec, numRows, op.getOptionFloat("maxval"));

    // Set up the padded data structures
    int paddedSize = numRows + (PAD_FACTOR - numRows % PAD_FACTOR);
	h_out = new floatType[paddedSize];
    convertToPadded(h_val, h_cols, numRows, h_rowDelimiters, &h_valPad,
            &h_colsPad, h_rowDelimitersPad, &nItemsPadded);

    // Compute reference solution
    spmvCpu(h_val, h_cols, h_rowDelimiters, h_vec, numRows, refOut);

    // Test CSR kernels on normal data
    cout << "CSR Test\n";
    csrTest<floatType>(resultDB, op, h_val, h_cols,
            h_rowDelimiters, h_vec, h_out, numRows, nItems, refOut, false);

#if READY
//TODO: below codes need to be implemented.
/*
    // Test CSR kernels on padded data
    cout << "CSR Test -- Padded Data\n";
    csrTest<floatType>(resultDB, op, h_valPad, h_colsPad,
            h_rowDelimitersPad, h_vec, h_out, numRows, nItemsPadded, refOut, true);

    // Test ELLPACKR kernel
    cout << "ELLPACKR Test\n";
    ellPackTest<floatType>(resultDB, op, h_val, h_cols,
            h_rowDelimiters, h_vec, h_out, numRows, nItems, refOut, false,
            paddedSize);
*/
#else
    // add "no result" results for the tests that aren't implemented
    std::string suffix = (sizeof(floatType) == sizeof(float)) ? "SP" : "DP";

    resultDB.AddResult(std::string("Padded_CSR-Scalar-") + suffix, "N/A", "Gflop/s", FLT_MAX);
    resultDB.AddResult(std::string("Padded_CSR-Scalar-") + suffix + "_PCIe", "N/A", "Gflop/s", FLT_MAX);

    resultDB.AddResult(std::string("Padded_CSR-Vector-") + suffix, "N/A", "Gflop/s", FLT_MAX);
    resultDB.AddResult(std::string("Padded_CSR-Vector-") + suffix + "_PCIe", "N/A", "Gflop/s", FLT_MAX);

    resultDB.AddResult(std::string("ELLPACKR-") + suffix, "N/A", "Gflop/s", FLT_MAX);
#endif // READY

    delete[] refOut; 
    delete[] h_val; 
    delete[] h_cols; 
    delete[] h_rowDelimiters; 
    delete[] h_vec; 
    delete[] h_out; 
    delete[] h_valPad; 
    delete[] h_colsPad; 
    delete[] h_rowDelimitersPad; 
}
