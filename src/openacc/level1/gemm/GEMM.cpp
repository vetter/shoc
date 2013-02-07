#include <iostream>
#include <sstream>
#include <string>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"

#ifndef _WIN32
#include <sys/time.h>
#endif

using namespace std;

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

template <class T>
inline void devGEMM(char transa, char transb, int m, int n, int k, T alpha,
        const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc,
        double* kTime, double* tTime);

extern "C" void sgemm(char transa, char transb, int m, int n, int k, 
        float alpha, const float *A, int lda, const float *B, int ldb, 
        float beta, float *C, int ldc, double* kTime, double* tTime);

extern "C" void dgemm(char transa, char transb, int m, int n, int k, 
        double alpha, const double *A, int lda, const double *B, int ldb, 
        double beta, double *C, int ldc, double* kTime, double* tTime);

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
    {
        A[j] = T((rand() % (maxi * 2 + 1)) - maxi) / (maxi + 1.);
    }
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
//  KS (2/5/13): Modified CUDA version to drive OpenACC GEMM kernel (M. Horton)
//
// ****************************************************************************
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{
    cout << "Running single precision test" << endl;
    RunTest<float>("SGEMM", resultDB, op);

    cout << "Running double precision test" << endl;
    RunTest<double>("DGEMM", resultDB, op);
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

    // Initialize host memory
    T *A = new T[N * N];
    T *B = new T[N * N];
    T *C = new T[N * N];
    
    fill<T>(A, N * N, 31);
    fill<T>(B, N * N, 31);
    fill<T>(C, N * N, 31);

    double transferTime = 1.0;
    double kernelTime   = 1.0;

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

            
            devGEMM<T>(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,
                    C, ldc, &kernelTime, &transferTime);
            
            double num_gflops = 2. * m * n * k / 1e9;
            double gflops_ach = num_gflops / kernelTime;
            double pcie_gflops = num_gflops / (kernelTime + transferTime);
            resultDB.AddResult(testName+"-"+transb, toString(dim), "GFlops",
                    gflops_ach);
            resultDB.AddResult(testName+"-"+transb+"_PCIe", toString(dim),
                    "GFlops", pcie_gflops);
            resultDB.AddResult(testName+"-"+transb+"_Parity", toString(dim),
                    "N", transferTime / kernelTime);
        }
    }
    delete[] A;
    delete[] B;
    delete[] C;
}

template<>
inline void devGEMM<double>(char transa, char transb, int m, int n, int k,
        double alpha, const double *A, int lda, const double *B, int ldb,
        double beta, double *C, int ldc, double* kTime, double* tTime) 
{
    dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 
            kTime, tTime);
}

template <>
inline void devGEMM<float>(char transa, char transb, int m, int n, int k,
        float alpha, const float *A, int lda, const float *B, int ldb,
        float beta, float *C, int ldc, double* kTime, double* tTime) 
{
    sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
            kTime, tTime);
}
