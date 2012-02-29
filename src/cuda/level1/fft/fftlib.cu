#include "cudacommon.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "OptionParser.h"
#include "fftlib.h"

int fftDevice = -1;

bool do_dp;


//#define USE_CUFFT

#ifdef USE_CUFFT
cufftHandle plan;
// arrange blocks into 2D grid that fits into the GPU ( for powers of two only )
inline dim3 grid2D( int nblocks )
{
    int slices = 1;
    while( nblocks/slices > 65535 ) 
        slices *= 2;
    return dim3( nblocks/slices, slices );
}
#else
#include "codelets.h"
#endif

template <class T2> __global__ void 
chk512_device( T2* work, int half_n_cmplx, char* fail )
{	
    int i, tid = threadIdx.x;
    T2 a[8], b[8];

    work += (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;
	
    for (i = 0; i < 8; i++) {
        a[i] = work[i*64];
    }
    
    for (i = 0; i < 8; i++) {
        b[i] = work[half_n_cmplx+i*64];
    }
    
    for (i = 0; i < 8; i++) {
        if (a[i].x != b[i].x || a[i].y != b[i].y) {
            *fail = 1;
        }
    }
}	


template <class T2> __global__ void 
norm512_device( T2* work)
{	
    int i, tid = threadIdx.x;

    work += (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;
	
    for (i = 0; i < 8; i++) {
        work[i*64].x /= 512;
        work[i*64].y /= 512;
    }
}	


void
init(OptionParser& op, bool _do_dp)
{
    do_dp = _do_dp;
    if (fftDevice == -1) {
        if (op.getOptionVecInt("device").size() > 0) {
            fftDevice = op.getOptionVecInt("device")[0];
        }
        else {
            fftDevice = 0;
        }
        cudaSetDevice(fftDevice);
        cudaGetDevice(&fftDevice);
    }
}


void
forward(void* work, int n_ffts)
{
#ifdef USE_CUFFT
    if (!plan) {
        fprintf(stderr, "forward: initing plan, n_ffts=%d\n", n_ffts);
        if (do_dp) {
            cufftPlan1d(&plan, 512, CUFFT_Z2Z, n_ffts);
        }
        else {
            cufftPlan1d(&plan, 512, CUFFT_C2C, n_ffts);
        }
        CHECK_CUDA_ERROR();
        fprintf(stderr, "success...\n");
    }
    if (do_dp) {
        cufftExecZ2Z(plan, (cufftDoubleComplex*)work, 
                     (cufftDoubleComplex*)work, CUFFT_FORWARD);
    }
    else {
        cufftExecC2C(plan, (cufftComplex*)work, (cufftComplex*)work, CUFFT_FORWARD);
    }
    CHECK_CUDA_ERROR();
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR();
#else
    if (do_dp) {
        FFT512_device<double2, double><<<grid2D(n_ffts), 64>>>((double2*)work);
    }
    else {
        FFT512_device<float2, float><<<grid2D(n_ffts), 64>>>((float2*)work);
    }
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR();
#endif
}


void
inverse(void* work, int n_ffts)
{
#ifdef USE_CUFFT
    if (!plan) {
        if (do_dp) {
            cufftPlan1d(&plan, 512, CUFFT_Z2Z, n_ffts);
        }
        else {
            cufftPlan1d(&plan, 512, CUFFT_C2C, n_ffts);
        }
        CHECK_CUDA_ERROR();
    }
    if (do_dp) {
        cufftExecZ2Z(plan, (cufftDoubleComplex*)work, 
                     (cufftDoubleComplex*)work, CUFFT_INVERSE);
    }
    else {
        cufftExecC2C(plan, (cufftComplex*)work, (cufftComplex*)work, CUFFT_INVERSE);
    }
    CHECK_CUDA_ERROR();
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR();

    // normalize data...
    if (do_dp) {
        norm512_device<double2><<<grid2D(n_ffts), 64>>>((double2*)work);
    }
    else {
        norm512_device<float2><<<grid2D(n_ffts), 64>>>((float2*)work);
    }
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR();
#else
    if (do_dp) {
        IFFT512_device<double2, double><<<grid2D(n_ffts), 64>>>((double2*)work);
    }
    else {
        IFFT512_device<float2, float><<<grid2D(n_ffts), 64>>>((float2*)work);
    }
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR();
    // normalization built in to inverse...
#endif
}


int
check(void* work, void* check, int half_n_ffts, int half_n_cmplx)
{
    char result;

    if (do_dp) {
        chk512_device<double2><<<grid2D(half_n_ffts), 64>>>(
            (double2*)work, half_n_cmplx, (char*)check);
    }
    else {
        chk512_device<float2><<<grid2D(half_n_ffts), 64>>>(
            (float2*)work, half_n_cmplx, (char*)check);
    }
    cudaMemcpy(&result, check, 1, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR();

    return result;
}


void
allocHostBuffer(void** bufferp, unsigned long bytes)
{
    cudaMallocHost(bufferp, bytes);
    CHECK_CUDA_ERROR();
}

void
allocDeviceBuffer(void** bufferp, unsigned long bytes)
{
    cudaMalloc(bufferp, bytes);
    CHECK_CUDA_ERROR();
}

void
freeHostBuffer(void* buffer)
{
    cudaFreeHost(buffer);
    CHECK_CUDA_ERROR();
}


void
freeDeviceBuffer(void* buffer)
{
    cudaFree(buffer);
}

void
copyToDevice(void* to_device, void* from_host, unsigned long bytes)
{
    cudaMemcpy(to_device, from_host, bytes, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
}


void
copyFromDevice(void* to_host, void* from_device, unsigned long bytes)
{
    cudaMemcpy(to_host, from_device, bytes, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR();
}

