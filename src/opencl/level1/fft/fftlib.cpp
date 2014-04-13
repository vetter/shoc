#include <stdio.h>
#include <assert.h>
#include "OpenCLDeviceInfo.h"
#include "Event.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "support.h"
#include "fftlib.h"

#include <map>
static map<void*, cl_mem> memobjmap;

cl_device_id fftDev;
cl_context fftCtx;
cl_command_queue fftQueue;
Event fftEvent("FFT"), ifftEvent("IFFT"), chkEvent("CHK");

extern const char *cl_source_fft;

static cl_kernel fftKrnl, ifftKrnl, chkKrnl;
static cl_program fftProg;
static bool do_dp;

void
init(OptionParser& op, bool _do_dp)
{
    cl_int err;

    do_dp = _do_dp;

    if (!fftCtx) {
        // first get the device
        int device, platform = op.getOptionInt("platform");
        if (op.getOptionVecInt("device").size() > 0) {
            device = op.getOptionVecInt("device")[0];
        }
        else {
            device = 0;
        }
        fftDev = ListDevicesAndGetDevice(platform, device);

        // now get the context
        fftCtx = clCreateContext(NULL, 1, &fftDev, NULL, NULL, &err);
        CL_CHECK_ERROR(err);
    }

    if (!fftQueue) {
        // get a queue
        fftQueue = clCreateCommandQueue(fftCtx, fftDev, CL_QUEUE_PROFILING_ENABLE,
                                        &err);
        CL_CHECK_ERROR(err);
    }

    // create the program...
    fftProg = clCreateProgramWithSource(fftCtx, 1, &cl_source_fft, NULL, &err);
    CL_CHECK_ERROR(err);

    // ...and build it
    string args = " -cl-mad-enable ";
    if (op.getOptionBool("use-native")) {
        args += " -cl-fast-relaxed-math ";
    }
    if (!do_dp) {
        args += " -DSINGLE_PRECISION ";
    }
    else if (checkExtension(fftDev, "cl_khr_fp64")) {
        args += " -DK_DOUBLE_PRECISION ";
    }
    else if (checkExtension(fftDev, "cl_amd_fp64")) {
        args += " -DAMD_DOUBLE_PRECISION ";
    }
    err = clBuildProgram(fftProg, 0, NULL, args.c_str(), NULL, NULL);
    {
        char* log = NULL;
        size_t bytesRequired = 0;
        err = clGetProgramBuildInfo(fftProg,
                                    fftDev,
                                    CL_PROGRAM_BUILD_LOG,
                                    0,
                                    NULL,
                                    &bytesRequired );
        log = (char*)malloc( bytesRequired + 1 );
        err = clGetProgramBuildInfo(fftProg,
                                    fftDev,
                                    CL_PROGRAM_BUILD_LOG,
                                    bytesRequired,
                                    log,
                                    NULL );
        std::cout << log << std::endl;
        free( log );
    }
    if (err != CL_SUCCESS) {
        char log[50000];
        size_t retsize = 0;
        err = clGetProgramBuildInfo(fftProg, fftDev, CL_PROGRAM_BUILD_LOG,
                                    50000*sizeof(char),  log, &retsize);
        CL_CHECK_ERROR(err);
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        dumpPTXCode(fftCtx, fftProg, "oclFFT");
        exit(-1);
    }
    else {
        // dumpPTXCode(fftCtx, fftProg, "oclFFT");
    }

    // Create kernel for forward FFT
    fftKrnl = clCreateKernel(fftProg, "fft1D_512", &err);
    CL_CHECK_ERROR(err);
    // Create kernel for inverse FFT
    ifftKrnl = clCreateKernel(fftProg, "ifft1D_512", &err);
    CL_CHECK_ERROR(err);
    // Create kernel for check
    chkKrnl = clCreateKernel(fftProg, "chk1D_512", &err);
    CL_CHECK_ERROR(err);
}


void deinit() {
    for (map<void*, cl_mem>::iterator it = memobjmap.begin(); it != memobjmap.end(); ++it) {
        clEnqueueUnmapMemObject(fftQueue, it->second, it->first, 0, NULL, NULL);
        clReleaseMemObject(it->second);
    }

    clReleaseKernel(fftKrnl);
    clReleaseKernel(ifftKrnl);
    clReleaseKernel(chkKrnl);
    clReleaseProgram(fftProg);
}


void
forward(void* workp, int n_ffts)
{
    cl_int err;
    size_t localsz = 64;
    size_t globalsz = localsz * n_ffts;

    clSetKernelArg(fftKrnl, 0, sizeof(cl_mem), workp);
    err = clEnqueueNDRangeKernel(fftQueue, fftKrnl, 1, NULL,
                                 &globalsz, &localsz, 0,
                                 NULL, &fftEvent.CLEvent());
    CL_CHECK_ERROR(err);
    err = clWaitForEvents(1, &fftEvent.CLEvent());
    CL_CHECK_ERROR(err);
}


void
inverse(void* workp, int n_ffts)
{
    cl_int err;
    size_t localsz = 64;
    size_t globalsz = localsz * n_ffts;

    clSetKernelArg(ifftKrnl, 0, sizeof(cl_mem), workp);
    err = clEnqueueNDRangeKernel(fftQueue, ifftKrnl, 1, NULL,
                                 &globalsz, &localsz, 0,
                                 NULL, &ifftEvent.CLEvent());
    CL_CHECK_ERROR(err);
    err = clWaitForEvents(1, &ifftEvent.CLEvent());
    CL_CHECK_ERROR(err);
}


int
check(void* workp, void* checkp, int half_n_ffts, int half_n_cmplx)
{
    cl_int err;
    size_t localsz = 64;
    size_t globalsz = localsz * half_n_ffts;
    int result;

    clSetKernelArg(chkKrnl, 0, sizeof(cl_mem), workp);
    clSetKernelArg(chkKrnl, 1, sizeof(int), (void*)&half_n_cmplx);
    clSetKernelArg(chkKrnl, 2, sizeof(cl_mem), checkp);

    err = clEnqueueNDRangeKernel(fftQueue, chkKrnl, 1, NULL,
                                 &globalsz, &localsz, 0,
                                 NULL, &chkEvent.CLEvent());
    CL_CHECK_ERROR(err);

    err = clEnqueueReadBuffer(fftQueue, *(cl_mem*)checkp, CL_TRUE, 0, sizeof(result),
                              &result, 1, &chkEvent.CLEvent(), NULL);
    CL_CHECK_ERROR(err);

    return result;
}


void
allocHostBuffer(void** bufp, unsigned long bytes)
{
#if 1 // pinned memory?
    cl_int err;
    cl_mem memobj = clCreateBuffer(fftCtx,CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
                                   bytes, NULL, &err);
    CL_CHECK_ERROR(err);

    *bufp = clEnqueueMapBuffer(fftQueue, memobj, true,
                               CL_MAP_READ|CL_MAP_WRITE,
                               0,bytes,0,NULL,NULL,&err);

    memobjmap[*bufp] = memobj;
    CL_CHECK_ERROR(err);
#else
    *bufp = malloc(bytes);
#endif
}


void
freeHostBuffer(void* buf)
{
#if 1 // pinned memory?
    cl_int err;
    cl_mem memobj = memobjmap[buf];
    err = clEnqueueUnmapMemObject(fftQueue, memobj, buf, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(memobj);
    CL_CHECK_ERROR(err);
    memobjmap.erase(buf);
#else
    free(buf);
#endif
}


void
allocDeviceBuffer(void** objp, unsigned long bytes)
{
    cl_int err;

    *(cl_mem**)objp = new cl_mem;
    **(cl_mem**)objp = clCreateBuffer(fftCtx, CL_MEM_READ_WRITE, bytes,
                                      NULL, &err);
    CL_CHECK_ERROR(err);
}


void
freeDeviceBuffer(void* buffer)
{
    clReleaseMemObject(*(cl_mem*)buffer);
}


void
copyToDevice(void* to_device, void* from_host, unsigned long bytes)
{
    cl_int err = clEnqueueWriteBuffer(fftQueue, *(cl_mem*)to_device, CL_TRUE,
                                      0, bytes, from_host, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
}


void
copyFromDevice(void* to_host, void* from_device, unsigned long bytes)
{
    cl_int err = clEnqueueReadBuffer(fftQueue, *(cl_mem*)from_device, CL_TRUE,
                                     0, bytes, to_host, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
}

