#include <cassert>
#include <hip_runtime_api.h>
#include <string>
#include <sstream>
#include "cudacommon.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "S3D.h"
#include "Timer.h"
#include "gr_base.h"
#include "ratt.h"
#include "ratt2.h"
#include "ratx.h"
#include "qssa.h"
#include "qssa2.h"
#include "rdwdot.h"

using namespace std;

// Forward declaration
template <class real>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

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
// Returns:  a string representation
//
// Modifications:
//
// ********************************************************
template<class T> inline string toString(const T& t)
{
    stringstream ss;
    ss << t;
    return ss.str();
}

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: March 13, 2010
//
// Modifications:
//
// ****************************************************************************
void
addBenchmarkSpecOptions(OptionParser &op)
{
    ; // No S3D specific options
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the S3D benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: March 13, 2010
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{
    // Always run the single precision test
    RunTest<float>("S3D-SP", resultDB, op);

    // Check to see if the device supports double precision
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, device);
    if ((deviceProp.major == 1 && deviceProp.minor >= 3) ||
                   (deviceProp.major >= 2))
    {
        cout << "Running double precision test" << endl;
        RunTest<double>("S3D-DP", resultDB, op);
    } else {
        cout << "Skipping double precision test" << endl;
        char atts[1024] = "DP_Not_Supported";
        // resultDB requires neg entry for every possible result
        unsigned int passes = op.getOptionInt("passes");
        for (unsigned int i = 0; i < passes; i++) {
            resultDB.AddResult("S3D-DP" , atts, "GFLOPS/s", FLT_MAX);
            resultDB.AddResult("S3D-DP_PCIe" , atts, "GFLOPS/s", FLT_MAX);
            resultDB.AddResult("S3D-DP_Parity" , atts, "GFLOPS/s", FLT_MAX);
        }
    }
}

template <class real>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op)
{
    // Number of grid points (specified in header file)
    int probSizes_SP[4] = { 24, 32, 40, 48};
    int probSizes_DP[4] = { 16, 24, 32, 40};
    int *probSizes = (sizeof(real) == sizeof(double)) ? probSizes_DP : probSizes_SP;
    int sizeClass = op.getOptionInt("size") - 1;
    assert(sizeClass >= 0 && sizeClass < 4);
    sizeClass = probSizes[sizeClass];
    int n = sizeClass * sizeClass * sizeClass;

    // Host variables
    real* host_t;
    real* host_p;
    real* host_y;
    real* host_wdot;
    real* host_molwt;

    // GPU Variables
    real* gpu_t; //Temperatures array
    real* gpu_p; //Pressures array
    real* gpu_y; //Mass fractions
    real* gpu_wdot; //Output variables

    // GPU Intermediate Variables
    real* gpu_rf, *gpu_rb;
    real* gpu_rklow;
    real* gpu_c;
    real* gpu_a;
    real* gpu_eg;
    real* gpu_molwt;

    // CUDA streams
    hipStream_t s1, s2;

   // configure kernels for large L1 cache, as we don't need shared memory
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratt_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(rdsmh_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(gr_base, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratt2_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratt3_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratt4_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratt5_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratt6_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratt7_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratt8_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratt9_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratx_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratxb_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratx2_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(ratx4_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(qssa_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(qssab_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(qssa2_kernel, cudaFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(rdwdot_kernel,
//            hipFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(rdwdot2_kernel,
//            hipFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(rdwdot3_kernel,
//            hipFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(rdwdot6_kernel,
//            hipFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(rdwdot7_kernel,
//            hipFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(rdwdot8_kernel,
//            hipFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(rdwdot9_kernel,
//            hipFuncCachePreferL1));
//    CUDA_SAFE_CALL(hipFuncSetCacheConfig(rdwdot10_kernel,
//            hipFuncCachePreferL1));

    // Malloc host memory
    CUDA_SAFE_CALL(hipHostMalloc((void**)&host_t,        n*sizeof(real)));
    CUDA_SAFE_CALL(hipHostMalloc((void**)&host_p,        n*sizeof(real)));
    CUDA_SAFE_CALL(hipHostMalloc((void**)&host_y, Y_SIZE*n*sizeof(real)));
    CUDA_SAFE_CALL(hipHostMalloc((void**)&host_wdot,WDOT_SIZE*n*sizeof(real)));
    CUDA_SAFE_CALL(hipHostMalloc((void**)&host_molwt,WDOT_SIZE*sizeof(real)));

    // Create streams
    CUDA_SAFE_CALL(hipStreamCreate(&s1));
    CUDA_SAFE_CALL(hipStreamCreate(&s2));

    // Initialize Test Problem

    // For now these are just 1, to compare results between cpu & gpu
    real rateconv = 1.0;
    real tconv = 1.0;
    real pconv = 1.0;

    // Initialize temp and pressure
    for (int i=0; i<n; i++)
    {
        host_p[i] = 1.0132e6;
        host_t[i] = 1000.0;
    }

    // Init molwt: for now these are just 1, to compare results betw. cpu & gpu
    for (int i=0; i<WDOT_SIZE; i++)
    {
        host_molwt[i] = 1;
    }

    // Initialize mass fractions
    for (int j=0; j<Y_SIZE; j++)
    {
        for (int i=0; i<n; i++)
        {
            host_y[(j*n)+i]= 0.0;
            if (j==14)
                host_y[(j*n)+i] = 0.064;
            if (j==3)
                host_y[(j*n)+i] = 0.218;
            if (j==21)
                host_y[(j*n)+i] = 0.718;
        }
    }

    // Malloc GPU memory
    CUDA_SAFE_CALL(hipMalloc((void**)&gpu_t, n*sizeof(real)));
    CUDA_SAFE_CALL(hipMalloc((void**)&gpu_p, n*sizeof(real)));
    CUDA_SAFE_CALL(hipMalloc((void**)&gpu_y, Y_SIZE*n*sizeof(real)));
    CUDA_SAFE_CALL(hipMalloc((void**)&gpu_wdot, WDOT_SIZE*n*sizeof(real)));
    CUDA_SAFE_CALL(hipMalloc((void**)&gpu_rf, RF_SIZE*n*sizeof(real)));
    CUDA_SAFE_CALL(hipMalloc((void**)&gpu_rb, RB_SIZE*n*sizeof(real)));
    CUDA_SAFE_CALL(hipMalloc((void**)&gpu_rklow, RKLOW_SIZE*n*sizeof(real)));
    CUDA_SAFE_CALL(hipMalloc((void**)&gpu_c, C_SIZE*n*sizeof(real)));
    CUDA_SAFE_CALL(hipMalloc((void**)&gpu_a, A_SIZE*n*sizeof(real)));
    CUDA_SAFE_CALL(hipMalloc((void**)&gpu_eg, EG_SIZE*n*sizeof(real)));
    CUDA_SAFE_CALL(hipMalloc((void**)&gpu_molwt, WDOT_SIZE*sizeof(real)));

    // Get kernel launch config, assuming n is divisible by block size
    dim3 thrds(BLOCK_SIZE,1,1);
    dim3 blks(n / BLOCK_SIZE,1,1);
    dim3 thrds2(BLOCK_SIZE2,1,1);
    dim3 blks2(n / BLOCK_SIZE2,1,1);

    hipEvent_t start, stop;
    CUDA_SAFE_CALL(hipEventCreate(&start));
    CUDA_SAFE_CALL(hipEventCreate(&stop));

    // Download of gpu_t, gpu_p, gpu_y, gpu_molwt
    CUDA_SAFE_CALL(hipEventRecord(start, 0));
    CUDA_SAFE_CALL(hipMemcpyAsync(gpu_t, host_t, n*sizeof(real),
            hipMemcpyHostToDevice, s1));
    CUDA_SAFE_CALL(hipMemcpyAsync(gpu_p, host_p, n*sizeof(real),
            hipMemcpyHostToDevice, s2));
    CUDA_SAFE_CALL(hipMemcpyAsync(gpu_y, host_y, Y_SIZE*n*sizeof(real),
            hipMemcpyHostToDevice, s2));
    CUDA_SAFE_CALL(hipMemcpyAsync(gpu_molwt,host_molwt,WDOT_SIZE*sizeof(real),
            hipMemcpyHostToDevice, s2));
    CUDA_SAFE_CALL(hipEventRecord(stop, 0));
    CUDA_SAFE_CALL(hipEventSynchronize(stop));

    // Get elapsed transfer time
    float iTransferTime = 0.0f;
    hipEventElapsedTime(&iTransferTime, start, stop);
    iTransferTime *= 1.e-3;

    unsigned int passes = op.getOptionInt("passes");
    for (unsigned int i = 0; i < passes; i++)
    {
        CUDA_SAFE_CALL(hipEventRecord(start, 0));
        hipLaunchKernel(HIP_KERNEL_NAME(ratt_kernel),dim3(blks2),dim3(thrds2),0,s1, gpu_t, gpu_rf, tconv);

        hipLaunchKernel(HIP_KERNEL_NAME(rdsmh_kernel),dim3(blks2),dim3(thrds2),0,s1, gpu_t, gpu_eg, tconv);

        hipLaunchKernel(HIP_KERNEL_NAME(gr_base),dim3(blks2),dim3(thrds2),0,s2, gpu_p, gpu_t, gpu_y,
                                               gpu_c, tconv, pconv);

        hipLaunchKernel(HIP_KERNEL_NAME(ratt2_kernel),dim3(blks2),dim3(thrds2),0,s1,gpu_t, gpu_rf, gpu_rb,
                gpu_eg, tconv);
        hipLaunchKernel(HIP_KERNEL_NAME(ratt3_kernel),dim3(blks2),dim3(thrds2),0,s1, gpu_t, gpu_rf, gpu_rb,
                gpu_eg, tconv);
        hipLaunchKernel(HIP_KERNEL_NAME(ratt4_kernel),dim3(blks2),dim3(thrds2),0,s1, gpu_t, gpu_rf, gpu_rb,
                gpu_eg, tconv);
        hipLaunchKernel(HIP_KERNEL_NAME(ratt5_kernel),dim3(blks2),dim3(thrds2),0,s1, gpu_t, gpu_rf, gpu_rb,
                gpu_eg, tconv);
        hipLaunchKernel(HIP_KERNEL_NAME(ratt6_kernel), dim3(blks2),dim3(thrds2),0,s1, gpu_t, gpu_rf, gpu_rb,
                gpu_eg, tconv);
        hipLaunchKernel(HIP_KERNEL_NAME(ratt7_kernel),dim3(blks2),dim3(thrds2),0,s1,gpu_t, gpu_rf, gpu_rb,
                gpu_eg, tconv);
        hipLaunchKernel(HIP_KERNEL_NAME(ratt8_kernel), dim3(blks2),dim3(thrds2),0,s1,gpu_t, gpu_rf, gpu_rb,
                gpu_eg, tconv);
        hipLaunchKernel(HIP_KERNEL_NAME(ratt9_kernel), dim3(blks2),dim3(thrds2),0,s1, gpu_t, gpu_rf, gpu_rb,
                gpu_eg, tconv);
        hipLaunchKernel(HIP_KERNEL_NAME(ratt10_kernel), dim3(blks2), dim3(thrds2) ,0,s1,gpu_t, gpu_rklow, tconv);

        hipLaunchKernel(HIP_KERNEL_NAME(ratx_kernel), dim3(blks), dim3(thrds), 0,s1, gpu_t, gpu_c, gpu_rf, gpu_rb,
                gpu_rklow, tconv);
        hipLaunchKernel(HIP_KERNEL_NAME(ratxb_kernel), dim3(blks), dim3(thrds), 0,s1, gpu_t, gpu_c, gpu_rf, gpu_rb,
                gpu_rklow, tconv);
        hipLaunchKernel(HIP_KERNEL_NAME(ratx2_kernel), dim3(blks2), dim3(thrds2), 0, s1, gpu_c, gpu_rf, gpu_rb);
        hipLaunchKernel(HIP_KERNEL_NAME(ratx4_kernel), dim3(blks2), dim3(thrds2),0,s1, gpu_c, gpu_rf, gpu_rb);

        hipLaunchKernel(HIP_KERNEL_NAME(qssa_kernel),dim3(blks2),dim3(thrds2),0,s1,gpu_rf, gpu_rb, gpu_a);
        hipLaunchKernel(HIP_KERNEL_NAME(qssab_kernel),dim3(blks), dim3(thrds), 0,s1,gpu_rf, gpu_rb, gpu_a);
        hipLaunchKernel(HIP_KERNEL_NAME(qssa2_kernel),dim3(blks2),dim3(thrds2),0,s1,gpu_rf, gpu_rb, gpu_a);

        hipLaunchKernel(HIP_KERNEL_NAME(rdwdot_kernel), dim3(blks2), dim3(thrds2) ,0,s1, gpu_rf, gpu_rb, gpu_wdot,
                rateconv, gpu_molwt);
        hipLaunchKernel(HIP_KERNEL_NAME(rdwdot2_kernel), dim3(blks2), dim3(thrds2),0,s1, gpu_rf, gpu_rb, gpu_wdot,
                rateconv, gpu_molwt);
        hipLaunchKernel(HIP_KERNEL_NAME(rdwdot3_kernel), dim3(blks2), dim3(thrds2),0,s1, gpu_rf, gpu_rb, gpu_wdot,
                rateconv, gpu_molwt);
        hipLaunchKernel(HIP_KERNEL_NAME(rdwdot6_kernel), dim3(blks2), dim3(thrds2),0,s1, gpu_rf, gpu_rb, gpu_wdot,
                rateconv, gpu_molwt);
        hipLaunchKernel(HIP_KERNEL_NAME(rdwdot7_kernel), dim3(blks2), dim3(thrds2),0,s1, gpu_rf, gpu_rb, gpu_wdot,
                rateconv, gpu_molwt);
        hipLaunchKernel(HIP_KERNEL_NAME(rdwdot8_kernel), dim3(blks2), dim3(thrds2),0,s1, gpu_rf, gpu_rb, gpu_wdot,
                rateconv, gpu_molwt);
        hipLaunchKernel(HIP_KERNEL_NAME(rdwdot9_kernel),dim3(blks2),dim3(thrds2),0,s1, gpu_rf, gpu_rb, gpu_wdot,
                rateconv, gpu_molwt);
        hipLaunchKernel(HIP_KERNEL_NAME(rdwdot10_kernel), dim3(blks2), dim3(thrds2),0,s1, gpu_rf, gpu_rb, gpu_wdot,
                rateconv, gpu_molwt);
        CUDA_SAFE_CALL(hipEventRecord(stop, 0));
        CUDA_SAFE_CALL(hipEventSynchronize(stop));

        // Get elapsed transfer time
        float kernelTime = 0.0f;
        hipEventElapsedTime(&kernelTime, start, stop);
        kernelTime *= 1.e-3;

        // Copy back result
        CUDA_SAFE_CALL(hipEventRecord(start, 0));
        CUDA_SAFE_CALL(hipMemcpyAsync(host_wdot, gpu_wdot,
                WDOT_SIZE * n * sizeof(real), hipMemcpyDeviceToHost, s1));
        CUDA_SAFE_CALL(hipEventRecord(stop, 0));
        CUDA_SAFE_CALL(hipEventSynchronize(stop));

        // Get elapsed transfer time
        float oTransferTime = 0.0f;
        hipEventElapsedTime(&oTransferTime, start, stop);
        oTransferTime *= 1.e-3;

        // Approximately 10k flops per grid point (estimated by Ramanan)
        double gflops = ((n*10000.) / 1.e9);

        resultDB.AddResult(testName, toString(n) + "_gridPoints", "GFLOPS",
                gflops / kernelTime);
        resultDB.AddResult(testName + "_PCIe", toString(n) + "_gridPoints", "GFLOPS",
                        gflops / (kernelTime + iTransferTime + oTransferTime));
        resultDB.AddResult(testName + "_Parity", toString(n) + "_gridPoints", "N",
                (iTransferTime + oTransferTime) / kernelTime);
    }

//    // Print out answers to compare with CPU
//    for (int i=0; i<WDOT_SIZE; i++) {
//        printf("% 23.16E ", host_wdot[i*n]);
//        if (i % 3 == 2)
//            printf("\n");
//    }
//    printf("\n");

    // Destroy streams and events
    CUDA_SAFE_CALL(hipStreamDestroy(s1));
    CUDA_SAFE_CALL(hipStreamDestroy(s2));
    CUDA_SAFE_CALL(hipEventDestroy(start));
    CUDA_SAFE_CALL(hipEventDestroy(stop));

    // Free GPU memory
    CUDA_SAFE_CALL(hipFree(gpu_t));
    CUDA_SAFE_CALL(hipFree(gpu_p));
    CUDA_SAFE_CALL(hipFree(gpu_y));
    CUDA_SAFE_CALL(hipFree(gpu_wdot));
    CUDA_SAFE_CALL(hipFree(gpu_rf));
    CUDA_SAFE_CALL(hipFree(gpu_rb));
    CUDA_SAFE_CALL(hipFree(gpu_c));
    CUDA_SAFE_CALL(hipFree(gpu_rklow));
    CUDA_SAFE_CALL(hipFree(gpu_a));
    CUDA_SAFE_CALL(hipFree(gpu_eg));
    CUDA_SAFE_CALL(hipFree(gpu_molwt));

    // Free host memory
    CUDA_SAFE_CALL(hipHostFree(host_t));
    CUDA_SAFE_CALL(hipHostFree(host_p));
    CUDA_SAFE_CALL(hipHostFree(host_y));
    CUDA_SAFE_CALL(hipHostFree(host_wdot));
    CUDA_SAFE_CALL(hipHostFree(host_molwt));
}
