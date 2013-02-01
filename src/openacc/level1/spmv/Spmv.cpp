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


template <class T>
void RunTest(const std::string& testName, 
                ResultDatabase& resultDB,
                OptionParser& op);

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

extern "C" void spmv_csr_scalar_kernel_float (const float * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * __restrict__ out);

extern "C" void spmv_csr_scalar_kernel_double (const double * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, double * __restrict__ out);

extern "C" void spmv_csr_vector_kernel_float (const float * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * __restrict__ out);

extern "C" void spmv_csr_vector_kernel_double (const double * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, double * __restrict__ out);

extern "C" void spmv_ellpackr_kernel_float (const float * __restrict__ val,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     const int dim, float * __restrict__ out);

extern "C" void spmv_ellpackr_kernel_double (const double * __restrict__ val,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     const int dim, double * __restrict__ out);

extern "C" void zero_float (float * __restrict__ a, const int size);

extern "C" void zero_double (double * __restrict__ a, const int size);


template <typename floatType, typename texReader>
void csrTest(ResultDatabase& resultDB, OptionParser& op, floatType* h_val,
        int* h_cols, int* h_rowDelimiters, floatType* h_vec, floatType* h_out,
        int numRows, int numNonZeroes, floatType* refOut, bool padded)
{
	void (*spmv_csr_scalar_kernel)(const void *,  
                       const int    *,
                       const int    *,
                       const int , void *);

	void (*spmv_csr_vector_kernel)(const void *,  
                       const int    *,
                       const int    *,
                       const int , void *);

	void (*zero)(float *, const int);

      // Device data structures
      floatType *d_val, *d_vec, *d_out;
      int *d_cols, *d_rowDelimiters;

      // Allocate device memory
      CUDA_SAFE_CALL(cudaMalloc(&d_val,  numNonZeroes * sizeof(floatType)));
      CUDA_SAFE_CALL(cudaMalloc(&d_cols, numNonZeroes * sizeof(int)));
      CUDA_SAFE_CALL(cudaMalloc(&d_vec,  numRows * sizeof(floatType)));
      CUDA_SAFE_CALL(cudaMalloc(&d_out,  numRows * sizeof(floatType)));
      CUDA_SAFE_CALL(cudaMalloc(&d_rowDelimiters, (numRows+1) * sizeof(int)));

      // Setup events for timing
      cudaEvent_t start, stop;
      CUDA_SAFE_CALL(cudaEventCreate(&start));
      CUDA_SAFE_CALL(cudaEventCreate(&stop));

      // Transfer data to device
      CUDA_SAFE_CALL(cudaEventRecord(start, 0));
      CUDA_SAFE_CALL(cudaMemcpy(d_val, h_val,   numNonZeroes * sizeof(floatType),
              cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_cols, h_cols, numNonZeroes * sizeof(int),
              cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_vec, h_vec, numRows * sizeof(floatType),
                    cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_rowDelimiters, h_rowDelimiters,
              (numRows+1) * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
      CUDA_SAFE_CALL(cudaEventSynchronize(stop));

      float iTransferTime, oTransferTime;
      CUDA_SAFE_CALL(cudaEventElapsedTime(&iTransferTime, start, stop));
      iTransferTime *= 1.e-3;

      string suffix;
      if (sizeof(floatType) == sizeof(float))
      {
          spmv_csr_scalar_kernel = spmv_csr_scalar_kernel_float;
          spmv_csr_vector_kernel = spmv_csr_vector_kernel_float;
          spmv_ellpackr_kernel = spmv_ellpackr_kernel_float;
          zero = zero_float;
          suffix = "-SP";
      }
      else {
          spmv_csr_scalar_kernel = spmv_csr_scalar_kernel_double;
          spmv_csr_vector_kernel = spmv_csr_vector_kernel_double;
          spmv_ellpackr_kernel = spmv_ellpackr_kernel_double;
          zero = zero_double;
          suffix = "-DP";
      }

      // Setup thread configuration
      int nBlocksScalar = (int) ceil((floatType) numRows / BLOCK_SIZE);
      int nBlocksVector = (int) ceil(numRows /
                  (floatType)(BLOCK_SIZE / WARP_SIZE));
      int passes = op.getOptionInt("passes");
      int iters  = op.getOptionInt("iterations");

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
          CUDA_SAFE_CALL(cudaEventRecord(start, 0));
          for (int j = 0; j < iters; j++)
          {
              spmv_csr_scalar_kernel<floatType, texReader>
              <<<nBlocksScalar, BLOCK_SIZE>>>
              (d_val, d_cols, d_rowDelimiters, numRows, d_out);
          }
          CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
          CUDA_SAFE_CALL(cudaEventSynchronize(stop));
          float scalarKernelTime;
          CUDA_SAFE_CALL(cudaEventElapsedTime(&scalarKernelTime, start, stop));
          // Transfer data back to host
          CUDA_SAFE_CALL(cudaEventRecord(start, 0));
          CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, numRows * sizeof(floatType),
                  cudaMemcpyDeviceToHost));
          CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
          CUDA_SAFE_CALL(cudaEventSynchronize(stop));
          CUDA_SAFE_CALL(cudaEventElapsedTime(&oTransferTime, start, stop));
          oTransferTime *= 1.e-3;
          // Compare reference solution to GPU result
          if (! verifyResults(refOut, h_out, numRows, k))
          {
              return;  // If results don't match, don't report performance
          }
          scalarKernelTime = (scalarKernelTime / (float)iters) * 1.e-3;
          string testName = prefix+"CSR-Scalar"+suffix;
          double totalTransfer = iTransferTime + oTransferTime;
          resultDB.AddResult(testName, atts, "Gflop/s",
                  gflop/(scalarKernelTime));
          resultDB.AddResult(testName+"_PCIe", atts, "Gflop/s",
                            gflop / (scalarKernelTime+totalTransfer));
      }
      zero<floatType><<<nBlocksScalar, BLOCK_SIZE>>>(d_out, numRows);
      cudaThreadSynchronize();

      cout << "CSR Vector Kernel\n";
      for (int k=0; k<passes; k++)
      {
          // Run Vector Kernel
          CUDA_SAFE_CALL(cudaEventRecord(start, 0));
          for (int j = 0; j < iters; j++)
          {
              spmv_csr_vector_kernel<floatType, texReader>
              <<<nBlocksVector, BLOCK_SIZE>>>
              (d_val, d_cols, d_rowDelimiters, numRows, d_out);
          }
          CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
          CUDA_SAFE_CALL(cudaEventSynchronize(stop));
          float vectorKernelTime;
          CUDA_SAFE_CALL(cudaEventElapsedTime(&vectorKernelTime, start, stop));
          CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, numRows * sizeof(floatType),
                  cudaMemcpyDeviceToHost));
          cudaThreadSynchronize();
          // Compare reference solution to GPU result
          if (! verifyResults(refOut, h_out, numRows, k))
          {
              return;  // If results don't match, don't report performance
          }
          vectorKernelTime = (vectorKernelTime / (float)iters) * 1.e-3;
          string testName = prefix+"CSR-Vector"+suffix;
          double totalTransfer = iTransferTime + oTransferTime;
          resultDB.AddResult(testName, atts, "Gflop/s", gflop/vectorKernelTime);
          resultDB.AddResult(testName+"_PCIe", atts, "Gflop/s",
                            gflop/(vectorKernelTime+totalTransfer));
      }
      // Free device memory
      CUDA_SAFE_CALL(cudaFree(d_rowDelimiters));
      CUDA_SAFE_CALL(cudaFree(d_vec));
      CUDA_SAFE_CALL(cudaFree(d_out));
      CUDA_SAFE_CALL(cudaFree(d_val));
      CUDA_SAFE_CALL(cudaFree(d_cols));
      CUDA_SAFE_CALL(cudaUnbindTexture(vecTexD));
      CUDA_SAFE_CALL(cudaUnbindTexture(vecTex));
      CUDA_SAFE_CALL(cudaEventDestroy(start));
      CUDA_SAFE_CALL(cudaEventDestroy(stop));
}

template <typename floatType, typename texReader>
void ellPackTest(ResultDatabase& resultDB, OptionParser& op, floatType* h_val,
        int* h_cols, int* h_rowDelimiters, floatType* h_vec, floatType* h_out,
        int numRows, int numNonZeroes, floatType* refOut, bool padded,
        int paddedSize)
{
	void (*spmv_ellpackr_kernel)(const void *,
                     const int    *,
                     const int    *,
                     const int, void *);
	void (*zero)(float *, const int);

    int *h_rowLengths; 
    CUDA_SAFE_CALL(cudaMallocHost(&h_rowLengths, paddedSize * sizeof(int))); 
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
    floatType *h_valcm;
    CUDA_SAFE_CALL(cudaMallocHost(&h_valcm, maxrl * cmSize * sizeof(floatType))); 
    int *h_colscm;
    CUDA_SAFE_CALL(cudaMallocHost(&h_colscm, maxrl * cmSize * sizeof(int))); 
    convertToColMajor(h_val, h_cols, numRows, h_rowDelimiters, h_valcm,
                              h_colscm, h_rowLengths, maxrl, padded);

    // Device data structures
    floatType *d_val, *d_vec, *d_out;
    int *d_cols, *d_rowLengths;

    // Allocate device memory
    CUDA_SAFE_CALL(cudaMalloc(&d_val,  maxrl*cmSize * sizeof(floatType)));
    CUDA_SAFE_CALL(cudaMalloc(&d_cols, maxrl*cmSize * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_vec,  numRows * sizeof(floatType)));
    CUDA_SAFE_CALL(cudaMalloc(&d_out,  paddedSize * sizeof(floatType)));
    CUDA_SAFE_CALL(cudaMalloc(&d_rowLengths, cmSize * sizeof(int)));

    // Transfer data to device
    CUDA_SAFE_CALL(cudaMemcpy(d_val, h_valcm, maxrl*cmSize * sizeof(floatType),
            cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_cols, h_colscm, maxrl*cmSize * sizeof(int),
            cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_vec, h_vec, numRows * sizeof(floatType),
            cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_rowLengths, h_rowLengths,
            cmSize * sizeof(int), cudaMemcpyHostToDevice));

    // Bind texture for position
    if (sizeof(floatType) == sizeof(float))
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        CUDA_SAFE_CALL(cudaBindTexture(0, vecTex, d_vec, channelDesc,
                numRows * sizeof(float)));
    }
    else
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int2>();
        CUDA_SAFE_CALL(cudaBindTexture(0, vecTexD, d_vec, channelDesc,
                numRows * sizeof(int2)));
    }
    int nBlocks = (int) ceil((floatType) cmSize / BLOCK_SIZE);
    int passes = op.getOptionInt("passes");
    int iters  = op.getOptionInt("iterations");
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    for (int k=0; k<passes; k++)
    {
        CUDA_SAFE_CALL(cudaEventRecord(start, 0));
        for (int j = 0; j < iters; j++)
        {
            spmv_ellpackr_kernel<floatType, texReader><<<nBlocks, BLOCK_SIZE>>>
                    (d_val, d_cols, d_rowLengths, cmSize, d_out);
        }
        CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(stop));
        float totalKernelTime;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&totalKernelTime, start, stop));
        totalKernelTime *= 1.e-3;

        CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, cmSize * sizeof(floatType),
                cudaMemcpyDeviceToHost));

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

    // Free device memory
    CUDA_SAFE_CALL(cudaFree(d_rowLengths));
    CUDA_SAFE_CALL(cudaFree(d_vec));
    CUDA_SAFE_CALL(cudaFree(d_out));
    CUDA_SAFE_CALL(cudaFree(d_val));
    CUDA_SAFE_CALL(cudaFree(d_cols));
    if (sizeof(floatType) == sizeof(double))
    {
        CUDA_SAFE_CALL(cudaUnbindTexture(vecTexD));
    }
    else
    {
        CUDA_SAFE_CALL(cudaUnbindTexture(vecTex));
    }
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    CUDA_SAFE_CALL(cudaFreeHost(h_rowLengths));
    CUDA_SAFE_CALL(cudaFreeHost(h_valcm));
    CUDA_SAFE_CALL(cudaFreeHost(h_colscm));
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
    RunTest<float, texReaderSP>(resultDB, op, probSizes[sizeClass]);

    cout <<"Double precision tests:\n";
    RunTest<double, texReaderDP>(resultDB, op, probSizes[sizeClass]);

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
template <typename floatType, typename texReader>
void RunTest(ResultDatabase &resultDB, OptionParser &op, int nRows=0) 
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
    csrTest<floatType, texReader>(resultDB, op, h_val, h_cols,
            h_rowDelimiters, h_vec, h_out, numRows, nItems, refOut, false);

    // Test CSR kernels on padded data
    cout << "CSR Test -- Padded Data\n";
    csrTest<floatType, texReader>(resultDB, op, h_valPad, h_colsPad,
            h_rowDelimitersPad, h_vec, h_out, numRows, nItemsPadded, refOut, true);

    // Test ELLPACKR kernel
    cout << "ELLPACKR Test\n";
    ellPackTest<floatType, texReader>(resultDB, op, h_val, h_cols,
            h_rowDelimiters, h_vec, h_out, numRows, nItems, refOut, false,
            paddedSize);

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
