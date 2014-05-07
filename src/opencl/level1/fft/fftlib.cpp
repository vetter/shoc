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

extern const char *cl_source_fft;

void
init(OptionParser& op,
     bool do_dp,
     cl_device_id fftDev,
     cl_context fftCtx,
     cl_command_queue fftQueue,
     cl_program& fftProg,
     cl_kernel& fftKrnl,
     cl_kernel& ifftKrnl,
     cl_kernel& chkKrnl)
{
    cl_int err;

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
        exit(-1);
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

void deinit(cl_command_queue fftQueue,
            cl_program& fftProg,
            cl_kernel& fftKrnl,
            cl_kernel& ifftKrnl,
            cl_kernel& chkKrnl)
{
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
transform(void* workp,
         const int n_ffts,
         Event& fftEvent,
         cl_kernel& fftKrnl,
         cl_command_queue& fftQueue)
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

int check(const void* workp,
          const void* checkp,
          const int half_n_ffts,
          const int half_n_cmplx,
          cl_kernel& chkKrnl,
          cl_command_queue& fftQueue)
{
    cl_int err;
    size_t localsz = 64;
    size_t globalsz = localsz * half_n_ffts;
    int result;
    Event chkEvent("CHK");

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

void allocHostBuffer(void** bufp,
                     const unsigned long bytes,
                     cl_context fftCtx,
                     cl_command_queue fftQueue)
{
    cl_int err;
    cl_mem memobj = clCreateBuffer(fftCtx,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, NULL, &err);
    CL_CHECK_ERROR(err);

    *bufp = clEnqueueMapBuffer(fftQueue, memobj, true,
                               CL_MAP_READ | CL_MAP_WRITE,
                               0,bytes,0,NULL,NULL,&err);
    memobjmap[*bufp] = memobj;
    CL_CHECK_ERROR(err);
}

void freeHostBuffer(void* buf,
                    cl_context fftCtx,
                    cl_command_queue fftQueue)
{
    cl_int err;
    cl_mem memobj = memobjmap[buf];
    err = clEnqueueUnmapMemObject(fftQueue, memobj, buf, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(memobj);
    CL_CHECK_ERROR(err);
    memobjmap.erase(buf);
}

void allocDeviceBuffer(void** bufferp,
                       const unsigned long bytes,
                       cl_context fftCtx,
                       cl_command_queue fftQueue)
{
    cl_int err;
    *(cl_mem**)bufferp = new cl_mem;
    **(cl_mem**)bufferp = clCreateBuffer(fftCtx, CL_MEM_READ_WRITE, bytes,
                                      NULL, &err);
    CL_CHECK_ERROR(err);
}

void freeDeviceBuffer(void* buffer,
                      cl_context fftCtx,
                      cl_command_queue fftQueue)
{
    clReleaseMemObject(*(cl_mem*)buffer);
}

void copyToDevice(void* to_device, void* from_host,
    const unsigned long bytes, cl_command_queue fftQueue)
{
    cl_int err = clEnqueueWriteBuffer(fftQueue, *(cl_mem*)to_device, CL_TRUE,
                                      0, bytes, from_host, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
}

void copyFromDevice(void* to_host, void* from_device,
    const unsigned long bytes, cl_command_queue fftQueue)
{
    cl_int err = clEnqueueReadBuffer(fftQueue, *(cl_mem*)from_device, CL_TRUE,
                                     0, bytes, to_host, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
}

