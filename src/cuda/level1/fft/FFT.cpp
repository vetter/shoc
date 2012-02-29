#include <math.h>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include "fftlib.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"
#include "support.h"

using namespace std;

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  The user is allowed to specify
//   the size of the input data in megabytes if they are not using a
//   predefined size (i.e. the -s option).
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Collin McCurdy
// Creation: September 08, 2009
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("MB", OPT_INT, "0", "data size (in megabytes)");
    op.addOption("dump-dp", OPT_BOOL, "false", "dump result after DP fft/ifft");
    op.addOption("dump-sp", OPT_BOOL, "false", "dump result after SP fft/ifft");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Calls single precision and, if viable, double precision FFT
//   benchmark.  Optionally dumps data arrays for correctness check.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Collin McCurdy
// Creation: September 08, 2009
//
// Modifications:
//
// ****************************************************************************
template <class T2> void runTest(const string& name, ResultDatabase &resultDB, OptionParser& op);
template <class T2> void dump(OptionParser& op);

void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{
    // Test to see if this device supports double precision
    cudaGetDevice(&fftDevice);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, fftDevice);
    bool has_dp = (deviceProp.major == 1 && deviceProp.minor >= 3) ||
        (deviceProp.major >= 2);

    if (op.getOptionBool("dump-sp")) {
        dump<float2>(op);
    }
    else if (op.getOptionBool("dump-dp")) {
        if (!has_dp) {
            cout << "dump-dp: no double precision support!\n";
            return;
        }
        dump<double2>(op);
    }
    else {
        cout << "Running single precision test" << endl;
        runTest<float2>("SP-FFT", resultDB, op);
        if (has_dp) {
            cout << "Running double precision test" << endl;
            runTest<double2>("DP-FFT", resultDB, op);
        } 
        else {
            cout << "Skipping double precision test" << endl;
            char atts[32] = "DP_Not_Supported";
            // resultDB requires neg entry for every possible result
            int passes = op.getOptionInt("passes");
            for (int k=0; k<passes; k++) {
                resultDB.AddResult("DP-FFT" , atts, "GB/s", FLT_MAX);
                resultDB.AddResult("DP-FFT_PCIe" , atts, "GB/s", FLT_MAX);
                resultDB.AddResult("DP-FFT_Parity" , atts, "GB/s", FLT_MAX);
                resultDB.AddResult("DP-FFT-INV" , atts, "GB/s", FLT_MAX);
                resultDB.AddResult("DP-FFT-INV_PCIe" , atts, "GB/s", FLT_MAX);
                resultDB.AddResult("DP-FFT-INV_Parity" , atts, "GB/s", FLT_MAX);
            }
        }
    }
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   This benchmark measures the performance of a single precision (or
//   double precision) fast fourier transform (FFT).  Data transfer
//   time over the PCIe bus is not included in this measurement.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Collin McCurdy
// Creation: September 08, 2009
//
// Modifications:
//   Jeremy Meredith, Thu Aug 19 15:43:55 EDT 2010
//   Added PCIe timings.  Added size index bounds check.
//
// ****************************************************************************
template <class T2> inline bool dp(void);
template <> inline bool dp<float2>(void) { return false; }
template <> inline bool dp<double2>(void) { return true; }

template <class T2>
void runTest(const string& name, ResultDatabase &resultDB, OptionParser& op)
{	
    int i, j;
    void* work, * chk;
    T2* source, * result;
    unsigned long bytes = 0;

    if (op.getOptionInt("MB") == 0) {
        int probSizes[4] = { 1, 8, 96, 256 };
	int sizeIndex = op.getOptionInt("size")-1;
	if (sizeIndex < 0 || sizeIndex >= 4) {
	    cerr << "Invalid size index specified\n";
	    exit(-1);
	}
        bytes = probSizes[sizeIndex];
    } 
    else {
        bytes = op.getOptionInt("MB");
    }

    // Convert to MB
    bytes *= 1024 * 1024;

    bool do_dp = dp<T2>();
    init(op, do_dp);

    int passes = op.getOptionInt("passes");

    // now determine how much available memory will be used
    int half_n_ffts = bytes / (512*sizeof(T2)*2);
    int n_ffts = half_n_ffts * 2;
    int half_n_cmplx = half_n_ffts * 512;
    unsigned long used_bytes = half_n_cmplx * 2 * sizeof(T2);
    double N = half_n_cmplx*2;
    
    // allocate host and device memory
    allocHostBuffer((void**)&source, used_bytes);
    allocHostBuffer((void**)&result, used_bytes);

    // init host memory...
    for (i = 0; i < half_n_cmplx; i++) {
        source[i].x = (rand()/(float)RAND_MAX)*2-1;
        source[i].y = (rand()/(float)RAND_MAX)*2-1;
        source[i+half_n_cmplx].x = source[i].x;
        source[i+half_n_cmplx].y = source[i].y;
    }

    // alloc device memory
    allocDeviceBuffer(&work, used_bytes);
    allocDeviceBuffer(&chk, 1);

    // Copy to device, and record transfer time
    fprintf(stderr, "used_bytes=%d, N=%g\n", used_bytes, N);

    int pcie_TH = Timer::Start();
    copyToDevice(work, source, used_bytes);
    double transfer_time = Timer::Stop(pcie_TH, "PCIe Transfer Time");

    char chk_init = 0;
    copyToDevice(chk, &chk_init, 1);

    const char *sizeStr;
    stringstream ss;
    ss << "N=" << (long)N;
    sizeStr = strdup(ss.str().c_str());

    for (int k=0; k<passes; k++) {
        // time fft kernel
        int TH = Timer::Start();
        forward(work, n_ffts);
        double t = Timer::Stop(TH, "fft");
        double fftsz = 512;
        double Gflops = n_ffts*(5*fftsz*log2(fftsz))/(t*1e9f);
        double gflopsPCIe = n_ffts*(5*fftsz*log2(fftsz)) /
                ((transfer_time+t)*1e9f);
        resultDB.AddResult(name, sizeStr, "GFLOPS", Gflops);
        resultDB.AddResult(name+"_PCIe", sizeStr, "GFLOPS", gflopsPCIe);
        resultDB.AddResult(name+"_Parity", sizeStr, "N", transfer_time / t);

        // time ifft kernel
        TH = Timer::Start();
        inverse(work, n_ffts);
        t = Timer::Stop(TH, "ifft");
        Gflops = n_ffts*(5*fftsz*log2(fftsz))/(t*1e9f);
        gflopsPCIe = n_ffts*(5*fftsz*log2(fftsz)) /
                ((transfer_time+t)*1e9f);
        resultDB.AddResult(name+"-INV", sizeStr, "GFLOPS", Gflops);
        resultDB.AddResult(name+"-INV_PCIe", sizeStr, "GFLOPS", gflopsPCIe);
        resultDB.AddResult(name+"-INV_Parity", sizeStr, "N", transfer_time / t);

        // time check kernel
        int failed = check(work, chk, half_n_ffts, half_n_cmplx);
        cout << "pass " << k << ((failed) ? ": failed\n" : ": passed\n");
    }
    
    freeDeviceBuffer(work);
    freeDeviceBuffer(chk);
    freeHostBuffer(source);
    freeHostBuffer(result);
}


// ****************************************************************************
// Function: dump
//
// Purpose:
//   Dump result array to stdout after FFT and IFFT.  For correctness 
//   checking.
//
// Arguments:
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Collin McCurdy
// Creation: September 30, 2010
//
// Modifications:
//
// ****************************************************************************
template <class T2> 
void dump(OptionParser& op)
{	
    int i, j;
    void* work;
    T2* source, * result;
    unsigned long bytes = 0;

    if (op.getOptionInt("MB") == 0) {
        int probSizes[4] = { 1, 8, 96, 256 };
	int sizeIndex = op.getOptionInt("size")-1;
	if (sizeIndex < 0 || sizeIndex >= 4) {
	    cerr << "Invalid size index specified\n";
	    exit(-1);
	}
        bytes = probSizes[sizeIndex];
    } else {
        bytes = op.getOptionInt("MB");
    }

    // Convert to MB
    bytes *= 1024 * 1024;

    bool do_dp = dp<T2>();
    init(op, do_dp);

    // now determine how much available memory will be used
    int half_n_ffts = bytes / (512*sizeof(T2)*2);
    int n_ffts = half_n_ffts * 2;
    int half_n_cmplx = half_n_ffts * 512;
    unsigned long used_bytes = half_n_cmplx * 2 * sizeof(T2);
    double N = half_n_cmplx*2;
    
    fprintf(stderr, "used_bytes=%d, N=%g\n", used_bytes, N);

    // allocate host and device memory
    allocHostBuffer((void**)&source, used_bytes);
    allocHostBuffer((void**)&result, used_bytes);

    // init host memory...
    for (i = 0; i < half_n_cmplx; i++) {
        source[i].x = (rand()/(float)RAND_MAX)*2-1;
        source[i].y = (rand()/(float)RAND_MAX)*2-1;
        source[i+half_n_cmplx].x = source[i].x;
        source[i+half_n_cmplx].y = source[i].y;
    }

    // alloc device memory
    allocDeviceBuffer(&work, used_bytes);

    copyToDevice(work, source, used_bytes);

    fprintf(stdout, "INITIAL:\n");
    for (i = 0; i < N; i++) {
        fprintf(stdout, "(%g, %g)\n", source[i].x, source[i].y);
    }
    
    forward(work, n_ffts);
    copyFromDevice(result, work, used_bytes);
        
    fprintf(stdout, "FORWARD:\n");
    for (i = 0; i < N; i++) {
        fprintf(stdout, "(%g, %g)\n", result[i].x, result[i].y);
    }
    
    inverse(work, n_ffts);
    copyFromDevice(result, work, used_bytes);
        
    fprintf(stdout, "\nINVERSE:\n");
    for (i = 0; i < N; i++) {
        fprintf(stdout, "(%g, %g)\n", result[i].x, result[i].y);
    }

    freeDeviceBuffer(work);
    freeHostBuffer(source);
    freeHostBuffer(result);
}
