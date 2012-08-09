#ifndef _S3D_H
#define _S3D_H

#define C_SIZE               (22)
#define RF_SIZE             (206)
#define RB_SIZE             (206)
#define WDOT_SIZE            (22)
#define RKLOW_SIZE           (21)
#define Y_SIZE               (22)
#define A_DIM                (11)
#define A_SIZE    (A_DIM * A_DIM)
#define EG_SIZE              (32)

// Macros to reduce openCL boilerplate for simple api calls
#define clProg(name, source)                                                   \
	cl_program name = clCreateProgramWithSource(ctx, 1,                    \
            &source, NULL, &err);                                              \
    CL_CHECK_ERROR(err)


#define clMalloc(ptr, size) { ptr =                                            \
    clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);                  \
    CL_CHECK_ERROR(err); }

#define clMemtoDevice(ptr1, ptr2, size) {                                      \
    err = clEnqueueWriteBuffer(queue, ptr1, true, 0, size, ptr2, 0,            \
        NULL, &evTransfer.CLEvent());                                          \
    CL_CHECK_ERROR(err);                                                       \
    err = clFinish(queue);                                                     \
    CL_CHECK_ERROR(err); }

#define clBuild(progname)                                                      \
    err = clBuildProgram(progname, 0, NULL,                                    \
            compileFlags.c_str(), NULL, NULL);                                 \
    CL_CHECK_ERROR(err);                                                       \
    if (err != 0) {                                                            \
      char log[8000];                                                          \
      size_t retsize = 0;                                                      \
      err =  clGetProgramBuildInfo (progname, dev, CL_PROGRAM_BUILD_LOG,       \
             sizeof(log),  log, &retsize);                                     \
      CL_CHECK_ERROR(err);                                                     \
      cout << "Retsize: " << retsize << endl;                                  \
      cout << "Log: " << log << endl;                                          \
   }


#define clSetRattArg(kernel_name) {                                            \
  err = clSetKernelArg(kernel_name, 0, sizeof(cl_mem), (void*)&d_t);           \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 1, sizeof(cl_mem), (void*)&d_rf);          \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 2, sizeof(cl_mem), (void*)&d_rb);          \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 3, sizeof(cl_mem), (void*)&d_eg);          \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 4, sizeof(T), (void*)&tconv);       \
  CL_CHECK_ERROR(err);                                                         \
  }

#define clSetRatxArg(kernel_name) {                                            \
  err = clSetKernelArg(kernel_name, 0, sizeof(cl_mem), (void*)&d_t);           \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 1, sizeof(cl_mem), (void*)&d_c);           \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 2, sizeof(cl_mem), (void*)&d_rf);          \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 3, sizeof(cl_mem), (void*)&d_rb);          \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 4, sizeof(cl_mem), (void*)&d_rklow);       \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 5, sizeof(T), (void*)&tconv);       \
  CL_CHECK_ERROR(err);                                                         \
  }

#define clSetQssaArg(kernel_name) {                                            \
  err = clSetKernelArg(kernel_name, 0, sizeof(cl_mem), (void*)&d_rf);          \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 1, sizeof(cl_mem), (void*)&d_rb);          \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 2, sizeof(cl_mem), (void*)&d_a);           \
  CL_CHECK_ERROR(err);                                                         \
  }

#define clSetRdwdotArg(kernel_name) {                                          \
  err = clSetKernelArg(kernel_name, 0, sizeof(cl_mem), (void*)&d_rf);          \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 1, sizeof(cl_mem), (void*)&d_rb);          \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 2, sizeof(cl_mem), (void*)&d_wdot);        \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 3, sizeof(T), (void*)&rateconv);    \
  CL_CHECK_ERROR(err);                                                         \
  err = clSetKernelArg(kernel_name, 4, sizeof(cl_mem), (void*)&d_molwt);       \
  CL_CHECK_ERROR(err);                                                         \
  }

#define clLaunchKernel(kernel_name) {                                          \
    err = clEnqueueNDRangeKernel(queue, kernel_name,                           \
    1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);                  \
    CL_CHECK_ERROR(err);                                                       \
    err = clFlush(queue);                                                      \
    CL_CHECK_ERROR(err);                                                       \
   }

#define clLaunchKernelEv(kernel_name, event_name) {                            \
    err = clEnqueueNDRangeKernel(queue, kernel_name,                           \
    1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &event_name);           \
    CL_CHECK_ERROR(err);                                                       \
    err = clFlush(queue);                                                      \
    CL_CHECK_ERROR(err);                                                       \
   }

#endif // _S3D_H
