#include <iostream>
#include <sstream>
#include <string>
#include "cuda.h"
#include "cudacommon.h"
#include "cublas.h"
#include "cuda_runtime.h"
#include "Timer.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

#ifndef _WIN32
#include <sys/time.h>
#endif

using namespace std;

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
void fill(T *A, int n, int maxi)
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
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   This benchmark measures the performance of the single precision general
//   matrix multiplication (SGEMM) operation in GFLOPS.  Data transfer time
//   over the PCIe bus is not included in this measurement.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: September 08, 2009
//
// Modifications:
//
// ****************************************************************************
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    cout << "Running single precision test" << endl;
    RunTest<float>("SGEMM", resultDB, op);

    // Test to see if this device supports double precision
    if ((deviceProp.major == 1 && deviceProp.minor >= 3) ||
               (deviceProp.major >= 2))
    {
        cout << "Running double precision test" << endl;
        RunTest<double>("DGEMM", resultDB, op);
    } else {
        cout << "Skipping double precision test" << endl;
        char atts[1024] = "DP_Not_Supported";
        // resultDB requires neg entry for every possible result
        int passes = op.getOptionInt("passes");
        for (; passes > 0; --passes) {
            for (int i = 0; i < 2; i++) {
                const char transb = i ? 'T' : 'N';
                string testName="DGEMM";
                resultDB.AddResult(testName+"-"+transb, atts, "GFlops", FLT_MAX);
                resultDB.AddResult(testName+"-"+transb+"_PCIe", atts, "GFlops", FLT_MAX);
                resultDB.AddResult(testName+"-"+transb+"_Parity", atts, "N", FLT_MAX);
            }
        }
    }
}

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op)
{
    int passes = op.getOptionInt("passes");
    int N;
    if (op.getOptionInt("KiB") == 0)
    {
        int probSizes[4] = { 1, 4, 8, 16 };
        N = probSizes[op.getOptionInt("size")-1] * 1024 / sizeof(T);
    } else {
        N = op.getOptionInt("KiB") * 1024 / sizeof(T);
    }

    // Initialize the cublas library
    cublasInit();

    // Allocate GPU memory
    T *dA, *dB, *dC;
    CUDA_SAFE_CALL(cudaMalloc(&dA, N * N * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc(&dB, N * N * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc(&dC, N * N * sizeof(T)));

    // Initialize host memory
    T *A;
    T *B;
    T *C;

    CUDA_SAFE_CALL(cudaMallocHost(&A, N * N * sizeof(T)));
    CUDA_SAFE_CALL(cudaMallocHost(&B, N * N * sizeof(T)));
    CUDA_SAFE_CALL(cudaMallocHost(&C, N * N * sizeof(T)));

    fill<T>(A, N * N, 31);
    fill<T>(B, N * N, 31);
    fill<T>(C, N * N, 31);

    // Copy input to GPU
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));
    CUDA_SAFE_CALL(cudaMemcpy(dA, A, N * N * sizeof(T),
            cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dB, B, N * N * sizeof(T),
            cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    // Get elapsed time
    float transferTime = 0.0f;
    cudaEventElapsedTime(&transferTime, start, stop);
    transferTime *= 1.e-3;

    bool first = true;
    for (; passes > 0; --passes)
    {
        for (int i = 0; i < 2; i++)
        {
            const char transa = 'N';
            const char transb = i ? 'T' : 'N';
            const int nb = 128;
            const int idim = N / nb;

            int dim = idim * nb;

            const int m = dim;
            const int n = dim;
            const int k = dim;
            const int lda = dim;
            const int ldb = dim;
            const int ldc = dim;
            const float alpha = 1;
            const float beta = 0;//-1;

            // Warm Up
            devGEMM<T>(transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta,
                    dC, ldc);
            CUDA_SAFE_CALL(cudaThreadSynchronize());

            double cublas_time;
            float kernel_time = 0.0f;
            for (int ii = 0; ii < 4; ++ii)
            {
                CUDA_SAFE_CALL(cudaEventRecord(start, 0));
                devGEMM<T>(transa, transb, m, n, k, alpha, dA, lda, dB, ldb,
                        beta, dC, ldc);
                CHECK_CUDA_ERROR();
                cudaEventRecord(stop, 0);
                CUDA_SAFE_CALL(cudaEventSynchronize(stop));
                float currTime = 0.0f;
                cudaEventElapsedTime(&currTime, start, stop);
                kernel_time += currTime;
            }
            cublas_time = (kernel_time / 4.0) * 1.e-3;

            CUDA_SAFE_CALL(cudaEventRecord(start, 0));
            CUDA_SAFE_CALL(cudaMemcpy(C, dC, N * N * sizeof(float),
                    cudaMemcpyDeviceToHost));
            cudaEventRecord(stop, 0);
            CUDA_SAFE_CALL(cudaEventSynchronize(stop));

            float oTransferTime = 0.0f;
            cudaEventElapsedTime(&oTransferTime, start, stop);
            oTransferTime *= 1.e-3;

            // Add the PCIe transfer time to total transfer time only once
            if (first)
            {
                transferTime += oTransferTime;
                first = false;
            }

            double cublas_gflops = 2. * m * n * k / cublas_time / 1e9;
            double pcie_gflops = 2. * m * n * k / (cublas_time + transferTime)
                    / 1e9;
            resultDB.AddResult(testName+"-"+transb, toString(dim), "GFlops",
                    cublas_gflops);
            resultDB.AddResult(testName+"-"+transb+"_PCIe", toString(dim),
                    "GFlops", pcie_gflops);
            resultDB.AddResult(testName+"-"+transb+"_Parity", toString(dim),
                    "N", transferTime / cublas_time);
        }
    }

    // Clean Up
    CUDA_SAFE_CALL(cudaFree(dA));
    CUDA_SAFE_CALL(cudaFree(dB));
    CUDA_SAFE_CALL(cudaFree(dC));
    CUDA_SAFE_CALL(cudaFreeHost(A));
    CUDA_SAFE_CALL(cudaFreeHost(B));
    CUDA_SAFE_CALL(cudaFreeHost(C));
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    cublasShutdown();
}

template<>
inline void devGEMM<double>(char transa, char transb, int m, int n, int k,
        double alpha, const double *A, int lda, const double *B, int ldb,
        double beta, double *C, int ldc) {
    cublasDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline void devGEMM<float>(char transa, char transb, int m, int n, int k,
        float alpha, const float *A, int lda, const float *B, int ldb,
        float beta, float *C, int ldc) {
    cublasSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
