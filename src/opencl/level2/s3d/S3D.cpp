#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "OpenCLDeviceInfo.h"
#include "Event.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "support.h"
#include "S3D.h"

using namespace std;

// Forward declarations
template <class T>
void RunTest(const string& testName, cl_device_id dev, cl_context ctx,
             cl_command_queue queue, ResultDatabase &resultDB,
             OptionParser &op, string& compileFlags);

template <class T> inline std::string toString (const T& t)
{
    std::stringstream ss;
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
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void
addBenchmarkSpecOptions(OptionParser &op)
{
    ;
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the S3D benchmark
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: Feb. 9, 2010
//
// Modifications:
//
// ****************************************************************************
// OpenCL Sources
extern const char *cl_source_gr_base;
extern const char *cl_source_qssa;
extern const char *cl_source_qssab;
extern const char *cl_source_qssa2;
extern const char *cl_source_ratt;
extern const char *cl_source_ratt2;
extern const char *cl_source_ratt3;
extern const char *cl_source_ratt4;
extern const char *cl_source_ratt5;
extern const char *cl_source_ratt6;
extern const char *cl_source_ratt7;
extern const char *cl_source_ratt8;
extern const char *cl_source_ratt9;
extern const char *cl_source_ratt10;
extern const char *cl_source_ratx;
extern const char *cl_source_ratxb;
extern const char *cl_source_ratx2;
extern const char *cl_source_ratx4;
extern const char *cl_source_rdsmh;
extern const char *cl_source_rdwdot;
extern const char *cl_source_rdwdot2;
extern const char *cl_source_rdwdot3;
extern const char *cl_source_rdwdot6;
extern const char *cl_source_rdwdot7;
extern const char *cl_source_rdwdot8;
extern const char *cl_source_rdwdot9;
extern const char *cl_source_rdwdot10;

void
RunBenchmark(cl_device_id dev,
             cl_context ctx,
             cl_command_queue queue,
             ResultDatabase &resultDB, OptionParser &op)
{
    // Always run single precision test
    // OpenCL doesn't support templated kernels, so we have to use macros
    string spMacros = "-DSINGLE_PRECISION ";
    RunTest<float>("S3D-SP", dev, ctx, queue, resultDB, op, spMacros);

    // If double precision is supported, run the DP test
    if (checkExtension(dev, "cl_khr_fp64"))
    {
        cout << "DP Supported\n";
        string dpMacros = "-DK_DOUBLE_PRECISION ";
        RunTest<double>
        ("S3D-DP", dev, ctx, queue, resultDB, op, dpMacros);
    }
    else if (checkExtension(dev, "cl_amd_fp64"))
    {
        cout << "DP Supported\n";
        string dpMacros = "-DAMD_DOUBLE_PRECISION ";
        RunTest<double>
        ("S3D-DP", dev, ctx, queue, resultDB, op, dpMacros);
    }
    else
    {
        cout << "DP Not Supported\n";
        char atts[1024] = "DP_Not_Supported";
        // resultDB requires neg entry for every possible result
        int passes = op.getOptionInt("passes");
        for (int k = 0; k < passes; k++) {
            resultDB.AddResult("S3D-DP" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult("S3D-DP_PCIe" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult("S3D-DP_Parity" , atts, "GB/s", FLT_MAX);
        }
    }
}


template <class T>
void RunTest(const string& testName, cl_device_id dev, cl_context ctx,
             cl_command_queue queue, ResultDatabase &resultDB,
             OptionParser &op, string& compileFlags)
{
    int n_species = 22;
    int i, j, err;

    int probSizes_SP[4] = { 24, 32, 40, 48};
    int probSizes_DP[4] = { 16, 24, 32, 40};
    int *probSizes = (sizeof(T) == sizeof(double)) ? probSizes_DP : probSizes_SP;
    int size = probSizes[op.getOptionInt("size")-1];

    // The number of grid points
    int n = size * size * size;

    // For now these conversion factors are just 1
    T pconv = 1.0;    // 1418365.88544;
    T tconv = 1.0;    //120.0;
    T rateconv = 1.0; //11.0393507649917;

    // Host copies of data
    T* h_t     = new T[n];
    T* h_p     = new T[n];
    T* h_y     = new T[n*n_species];
    T* h_wdot  = new T[n*n_species];
    T* h_molwt = new T[n_species];

    // Device data
    cl_mem d_t;    // Temperatures array
    cl_mem d_p;    // Pressures array
    cl_mem d_y;    // Input variables
    cl_mem d_wdot; // Output variables

    // intermediate variables
    cl_mem d_rf, d_rb, d_rklow, d_c, d_a, d_eg, d_molwt;

    // Initialize host memory
    for (i=0; i<n; i++)
    {
        h_p[i] = 1.0132e6;
        h_t[i] = 1000.0;
    }

    for (j=0; j<22; j++)
    {
        for (i=0; i<n; i++)
        {
            h_y[(j*n)+i]= 0.0;
            if (j==14)
                h_y[(j*n)+i] = 0.064;
            if (j==3)
                h_y[(j*n)+i] = 0.218;
            if (j==21)
                h_y[(j*n)+i] = 0.718;
        }
    }

    for (int i=0; i<n_species; i++)
    {
        h_molwt[i] = 1.0f;
    }
//    // Initialize molecular weights
//    h_molwt[0]= 2.01594E-03;
//    h_molwt[1]= 1.00797E-03;
//    h_molwt[2]= 1.59994E-02;
//    h_molwt[3]= 3.19988E-02;
//    h_molwt[4]= 1.700737E-02;
//    h_molwt[5]= 1.801534E-02;
//    h_molwt[6]= 3.300677E-02;
//    h_molwt[7]= 3.401473999999999E-02;
//    h_molwt[8]= 1.503506E-02;
//    h_molwt[9] = 1.604303E-02;
//    h_molwt[10] = 2.801055E-02;
//    h_molwt[11] = 4.400995E-02;
//    h_molwt[12] = 3.002649E-02;
//    h_molwt[13] = 2.603824E-02;
//    h_molwt[14] = 2.805418E-02;
//    h_molwt[15] = 3.007012E-02;
//    h_molwt[16] = 4.102967E-02;
//    h_molwt[17] = 4.203764E-02;
//    h_molwt[18] = 4.405358E-02;
//    h_molwt[19] = 4.10733E-02;
//    h_molwt[20] = 4.208127E-02;
//    h_molwt[21] = 2.80134E-02;

    // Allocate device memory
    size_t base = n * sizeof(T);
    clMalloc(d_t, base);
    clMalloc(d_p, base);
    clMalloc(d_y, n_species*base);
    clMalloc(d_wdot, n_species*base);
    clMalloc(d_rf, 206*base);
    clMalloc(d_rb, 206*base);
    clMalloc(d_rklow, 21*base);
    clMalloc(d_c, C_SIZE*base);
    clMalloc(d_a, A_SIZE*base);
    clMalloc(d_eg, EG_SIZE*base);
    clMalloc(d_molwt, n_species*sizeof(T));

    // Copy over input params
    long inputTransferTime = 0;
    Event evTransfer("PCIe Transfer");

    clMemtoDevice(d_t, h_t, base);
    evTransfer.FillTimingInfo();
    inputTransferTime += evTransfer.StartEndRuntime();

    clMemtoDevice(d_p, h_p, base);
    evTransfer.FillTimingInfo();
    inputTransferTime += evTransfer.StartEndRuntime();

    clMemtoDevice(d_y, h_y, n_species*base);
    evTransfer.FillTimingInfo();
    inputTransferTime += evTransfer.StartEndRuntime();

    clMemtoDevice(d_molwt, h_molwt, n_species*sizeof(T));
    evTransfer.FillTimingInfo();
    inputTransferTime += evTransfer.StartEndRuntime();

    // Set up macros
    compileFlags += "-DDIM="  + toString(size) + " " +
                    "-DN_GP=" + toString(n)    + " ";

    unsigned int passes = op.getOptionInt("passes");
    for (unsigned int i = 0; i < passes; i++)
    {
        size_t globalWorkSize = n;
        size_t localWorkSize = 128;

        // -------------------- phase 1 -----------------

        // Setup Program Objects (phase 1)
        clProg(gr_prog, cl_source_gr_base);
        clProg(rdsmh_prog, cl_source_rdsmh);
        clProg(ratt_prog, cl_source_ratt);
        clProg(ratt2_prog, cl_source_ratt2);
        clProg(ratt3_prog, cl_source_ratt3);
        clProg(ratt4_prog, cl_source_ratt4);
        clProg(ratt5_prog, cl_source_ratt5);
        clProg(ratt6_prog, cl_source_ratt6);
        clProg(ratt7_prog, cl_source_ratt7);
        clProg(ratt8_prog, cl_source_ratt8);
        clProg(ratt9_prog, cl_source_ratt9);
        clProg(ratt10_prog, cl_source_ratt10);
        clProg(ratx_prog, cl_source_ratx);
        clProg(ratxb_prog, cl_source_ratxb);
        clProg(ratx2_prog, cl_source_ratx2);
        clProg(ratx4_prog, cl_source_ratx4);

        // Build the kernels (phase 1)
        cout << "Compiling kernels (phase 1)...";
        cout.flush();

        clBuild(gr_prog);
        clBuild(rdsmh_prog);
        clBuild(ratt_prog);
        clBuild(ratt2_prog);
        clBuild(ratt3_prog);
        clBuild(ratt4_prog);
        clBuild(ratt5_prog);
        clBuild(ratt6_prog);
        clBuild(ratt7_prog);
        clBuild(ratt8_prog);
        clBuild(ratt9_prog);
        clBuild(ratt10_prog);
        clBuild(ratx_prog);
        clBuild(ratxb_prog);
        clBuild(ratx2_prog);
        clBuild(ratx4_prog);

        cout << "done." << endl;

        // Extract out kernel objects (phase 1)
        cout << "Generating OpenCL Kernel Objects (phase 1)...";
        cout.flush();

        // GR Base Kernels
        cl_kernel grBase_kernel = clCreateKernel(gr_prog, "gr_base", &err);
        CL_CHECK_ERROR(err);

        // RDSMH Kernels
        cl_kernel rdsmh_kernel = clCreateKernel(rdsmh_prog, "rdsmh_kernel",
                &err);
        CL_CHECK_ERROR(err);

        // RATT Kernels
        cl_kernel ratt_kernel = clCreateKernel(ratt_prog, "ratt_kernel", &err);
        CL_CHECK_ERROR(err);
        cl_kernel ratt2_kernel = clCreateKernel(ratt2_prog, "ratt2_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cl_kernel ratt3_kernel = clCreateKernel(ratt3_prog, "ratt3_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cl_kernel ratt4_kernel = clCreateKernel(ratt4_prog, "ratt4_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cl_kernel ratt5_kernel = clCreateKernel(ratt5_prog, "ratt5_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cl_kernel ratt6_kernel = clCreateKernel(ratt6_prog, "ratt6_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cl_kernel ratt7_kernel = clCreateKernel(ratt7_prog, "ratt7_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cl_kernel ratt8_kernel = clCreateKernel(ratt8_prog, "ratt8_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cl_kernel ratt9_kernel = clCreateKernel(ratt9_prog, "ratt9_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cl_kernel ratt10_kernel = clCreateKernel(ratt10_prog, "ratt10_kernel",
                &err);
        CL_CHECK_ERROR(err);

        // RATX Kernels
        cl_kernel ratx_kernel = clCreateKernel(ratx_prog, "ratx_kernel", &err);
        CL_CHECK_ERROR(err);
        cl_kernel ratxb_kernel = clCreateKernel(ratxb_prog, "ratxb_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cl_kernel ratx2_kernel = clCreateKernel(ratx2_prog, "ratx2_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cl_kernel ratx4_kernel = clCreateKernel(ratx4_prog, "ratx4_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cout << "done." << endl;

        //Set kernel arguments (phase 1)
        err = clSetKernelArg(grBase_kernel, 0, sizeof(cl_mem), (void*)&d_p);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(grBase_kernel, 1, sizeof(cl_mem), (void*)&d_t);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(grBase_kernel, 2, sizeof(cl_mem), (void*)&d_y);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(grBase_kernel, 3, sizeof(cl_mem), (void*)&d_c);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(grBase_kernel, 4, sizeof(T), (void*)&tconv);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(grBase_kernel, 5, sizeof(T), (void*)&pconv);
        CL_CHECK_ERROR(err);

        err = clSetKernelArg(rdsmh_kernel, 0, sizeof(cl_mem), (void*)&d_t);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(rdsmh_kernel, 1, sizeof(cl_mem), (void*)&d_eg);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(rdsmh_kernel, 2, sizeof(T), (void*)&tconv);
        CL_CHECK_ERROR(err);

        err = clSetKernelArg(ratt_kernel, 0, sizeof(cl_mem), (void*)&d_t);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(ratt_kernel, 1, sizeof(cl_mem), (void*)&d_rf);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(ratt_kernel, 2, sizeof(T), (void*)&tconv);
        CL_CHECK_ERROR(err);
        clSetRattArg(ratt2_kernel);
        clSetRattArg(ratt3_kernel);
        clSetRattArg(ratt4_kernel);
        clSetRattArg(ratt5_kernel);
        clSetRattArg(ratt6_kernel);
        clSetRattArg(ratt7_kernel);
        clSetRattArg(ratt8_kernel);
        clSetRattArg(ratt9_kernel);
        err = clSetKernelArg(ratt10_kernel, 0, sizeof(cl_mem), (void*)&d_t);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(ratt10_kernel, 1, sizeof(cl_mem), (void*)&d_rklow);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(ratt10_kernel, 2, sizeof(T), (void*)&tconv);
        CL_CHECK_ERROR(err);

        clSetRatxArg(ratx_kernel);
        clSetRatxArg(ratxb_kernel);

        err = clSetKernelArg(ratx2_kernel, 0, sizeof(cl_mem), (void*)&d_c);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(ratx2_kernel, 1, sizeof(cl_mem), (void*)&d_rf);
        CL_CHECK_ERROR(err);

        err = clSetKernelArg(ratx4_kernel, 0, sizeof(cl_mem), (void*)&d_c);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(ratx4_kernel, 1, sizeof(cl_mem), (void*)&d_rb);
        CL_CHECK_ERROR(err);

        // Execute kernels (phase 1)
        cout << "Executing kernels (phase 1)...";
        cout.flush();

        Event evFirst_1("first kernel phase 1");
        Event evLast_1("last kernel phase 1");

        clLaunchKernelEv(grBase_kernel, evFirst_1.CLEvent());
        clLaunchKernel(ratt_kernel);
        clLaunchKernel(rdsmh_kernel);

        clLaunchKernel(ratt2_kernel);
        clLaunchKernel(ratt3_kernel);
        clLaunchKernel(ratt4_kernel);
        clLaunchKernel(ratt5_kernel);
        clLaunchKernel(ratt6_kernel);
        clLaunchKernel(ratt7_kernel);
        clLaunchKernel(ratt8_kernel);
        clLaunchKernel(ratt9_kernel);
        clLaunchKernel(ratt10_kernel);

        clLaunchKernel(ratx_kernel);
        clLaunchKernel(ratxb_kernel);
        clLaunchKernel(ratx2_kernel);
        clLaunchKernelEv(ratx4_kernel, evLast_1.CLEvent());

        err = clFinish(queue);
        CL_CHECK_ERROR(err);
        cout << "done. " << endl;

        evFirst_1.FillTimingInfo();
        evLast_1.FillTimingInfo();
        double total_phase1 = evLast_1.EndTime() - evFirst_1.StartTime();

        // Release Kernels (phase 1)
        clReleaseKernel(grBase_kernel);
        clReleaseKernel(rdsmh_kernel);
        clReleaseKernel(ratt_kernel);
        clReleaseKernel(ratt2_kernel);
        clReleaseKernel(ratt3_kernel);
        clReleaseKernel(ratt4_kernel);
        clReleaseKernel(ratt5_kernel);
        clReleaseKernel(ratt6_kernel);
        clReleaseKernel(ratt7_kernel);
        clReleaseKernel(ratt8_kernel);
        clReleaseKernel(ratt9_kernel);
        clReleaseKernel(ratt10_kernel);
        clReleaseKernel(ratx_kernel);
        clReleaseKernel(ratxb_kernel);
        clReleaseKernel(ratx2_kernel);
        clReleaseKernel(ratx4_kernel);

        // Release Programs (phase 1)
        clReleaseProgram(gr_prog);
        clReleaseProgram(rdsmh_prog);
        clReleaseProgram(ratt_prog);
        clReleaseProgram(ratt2_prog);
        clReleaseProgram(ratt3_prog);
        clReleaseProgram(ratt4_prog);
        clReleaseProgram(ratt5_prog);
        clReleaseProgram(ratt6_prog);
        clReleaseProgram(ratt7_prog);
        clReleaseProgram(ratt8_prog);
        clReleaseProgram(ratt9_prog);
        clReleaseProgram(ratt10_prog);
        clReleaseProgram(ratx_prog);
        clReleaseProgram(ratxb_prog);
        clReleaseProgram(ratx2_prog);
        clReleaseProgram(ratx4_prog);

        // -------------------- phase 2 -----------------
        // Setup Program Objects (phase 2)
        clProg(qssa_prog, cl_source_qssa);
        clProg(qssab_prog, cl_source_qssab);
        clProg(qssa2_prog, cl_source_qssa2);
        clProg(rdwdot_prog, cl_source_rdwdot);
        clProg(rdwdot2_prog, cl_source_rdwdot2);
        clProg(rdwdot3_prog, cl_source_rdwdot3);
        clProg(rdwdot6_prog, cl_source_rdwdot6);
        clProg(rdwdot7_prog, cl_source_rdwdot7);
        clProg(rdwdot8_prog, cl_source_rdwdot8);
        clProg(rdwdot9_prog, cl_source_rdwdot9);
        clProg(rdwdot10_prog, cl_source_rdwdot10);

        // Build the kernels (phase 2)
        cout << "Compiling kernels (phase 2)...";
        cout.flush();

        clBuild(qssa_prog);
        clBuild(qssab_prog);
        clBuild(qssa2_prog);
        clBuild(rdwdot_prog);
        clBuild(rdwdot2_prog);
        clBuild(rdwdot3_prog);
        clBuild(rdwdot6_prog);
        clBuild(rdwdot7_prog);
        clBuild(rdwdot8_prog);
        clBuild(rdwdot9_prog);
        clBuild(rdwdot10_prog);

        cout << "done." << endl;

        // Extract out kernel objects (phase 2)
        cout << "Generating OpenCL Kernel Objects (phase 2)...";
        cout.flush();

        // QSSA Kernels
        cl_kernel qssa_kernel = clCreateKernel(qssa_prog, "qssa_kernel", &err);
        CL_CHECK_ERROR(err);
        cl_kernel qssab_kernel = clCreateKernel(qssab_prog, "qssab_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cl_kernel qssa2_kernel = clCreateKernel(qssa2_prog, "qssa2_kernel",
                &err);
        CL_CHECK_ERROR(err);

        // RDWDOT Kernels
        cl_kernel rdwdot_kernel = clCreateKernel(rdwdot_prog, "rdwdot_kernel",
                &err);
        CL_CHECK_ERROR(err);
        cl_kernel rdwdot2_kernel = clCreateKernel(rdwdot2_prog,
                "rdwdot2_kernel", &err);
        CL_CHECK_ERROR(err);
        cl_kernel rdwdot3_kernel = clCreateKernel(rdwdot3_prog,
                "rdwdot3_kernel", &err);
        CL_CHECK_ERROR(err);
        cl_kernel rdwdot6_kernel = clCreateKernel(rdwdot6_prog,
                "rdwdot6_kernel", &err);
        CL_CHECK_ERROR(err);
        cl_kernel rdwdot7_kernel = clCreateKernel(rdwdot7_prog,
                "rdwdot7_kernel", &err);
        CL_CHECK_ERROR(err);
        cl_kernel rdwdot8_kernel = clCreateKernel(rdwdot8_prog,
                "rdwdot8_kernel", &err);
        CL_CHECK_ERROR(err);
        cl_kernel rdwdot9_kernel = clCreateKernel(rdwdot9_prog,
                "rdwdot9_kernel", &err);
        CL_CHECK_ERROR(err);
        cl_kernel rdwdot10_kernel = clCreateKernel(rdwdot10_prog,
                "rdwdot10_kernel", &err);
        CL_CHECK_ERROR(err);
        cout << "done." << endl;

        //Set kernel arguments (phase 2)
        clSetQssaArg(qssa_kernel);
        clSetQssaArg(qssab_kernel);
        clSetQssaArg(qssa2_kernel);

        clSetRdwdotArg(rdwdot_kernel);
        clSetRdwdotArg(rdwdot2_kernel);
        clSetRdwdotArg(rdwdot3_kernel);
        clSetRdwdotArg(rdwdot6_kernel);
        clSetRdwdotArg(rdwdot7_kernel);
        clSetRdwdotArg(rdwdot8_kernel);
        clSetRdwdotArg(rdwdot9_kernel);
        clSetRdwdotArg(rdwdot10_kernel);

        // Execute kernels (phase 2)
        cout << "Executing kernels (phase 2)...";
        cout.flush();

        Event evFirst_2("first kernel phase 2");
        Event evLast_2("last kernel phase 2");

        clLaunchKernelEv(qssa_kernel, evFirst_2.CLEvent());
        clLaunchKernel(qssab_kernel);
        clLaunchKernel(qssa2_kernel);

        clLaunchKernel(rdwdot_kernel);
        clLaunchKernel(rdwdot2_kernel);
        clLaunchKernel(rdwdot3_kernel);
        clLaunchKernel(rdwdot6_kernel);
        clLaunchKernel(rdwdot7_kernel);
        clLaunchKernel(rdwdot8_kernel);
        clLaunchKernel(rdwdot9_kernel);
        clLaunchKernelEv(rdwdot10_kernel, evLast_2.CLEvent());

        err = clFinish(queue);
        CL_CHECK_ERROR(err);
        cout << "done. " << endl;

        evFirst_2.FillTimingInfo();
        evLast_2.FillTimingInfo();
        double total_phase2 = evLast_2.EndTime() - evFirst_2.StartTime();

        // Release Kernels (phase 2)
        clReleaseKernel(qssa_kernel);
        clReleaseKernel(qssab_kernel);
        clReleaseKernel(qssa2_kernel);
        clReleaseKernel(rdwdot_kernel);
        clReleaseKernel(rdwdot2_kernel);
        clReleaseKernel(rdwdot3_kernel);
        clReleaseKernel(rdwdot6_kernel);
        clReleaseKernel(rdwdot7_kernel);
        clReleaseKernel(rdwdot8_kernel);
        clReleaseKernel(rdwdot9_kernel);
        clReleaseKernel(rdwdot10_kernel);

        // Release Programs (phase 2)
        clReleaseProgram(qssa_prog);
        clReleaseProgram(qssab_prog);
        clReleaseProgram(qssa2_prog);
        clReleaseProgram(rdwdot_prog);
        clReleaseProgram(rdwdot2_prog);
        clReleaseProgram(rdwdot3_prog);
        clReleaseProgram(rdwdot6_prog);
        clReleaseProgram(rdwdot7_prog);
        clReleaseProgram(rdwdot8_prog);
        clReleaseProgram(rdwdot9_prog);
        clReleaseProgram(rdwdot10_prog);

        // -------------------- timings -----------------

        double total = total_phase1 + total_phase2;
        // Estimate GFLOPs (roughly 10k flops / point)
        double gflops = (n*10000.) / total;
        // Copy results back
        err = clEnqueueReadBuffer(queue, d_wdot, true, 0,
                n*n_species*sizeof(T), h_wdot,
                0, NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        err = clFinish(queue);
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        double totalTransferTime = inputTransferTime +
            evTransfer.StartEndRuntime();
        double gflops_pcie = (n*10000.) / (total + totalTransferTime);

        resultDB.AddResult(testName, "cubic", "GFLOPS", gflops);
        resultDB.AddResult(testName+"_PCIe", "cubic", "GFLOPS", gflops_pcie);
        resultDB.AddResult(testName+"_Parity", "cubic", "n", totalTransferTime / total );
    }


    // Release Memory
    err = clReleaseMemObject(d_t);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_p);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_y);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_wdot);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_rf);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_rb);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_c);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_eg);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_rklow);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_a);
    CL_CHECK_ERROR(err);

    // Cleanup Host Memory Objects
    delete[] h_t;
    delete[] h_p;
    delete[] h_y;
    delete[] h_wdot;
    delete[] h_molwt;

}
