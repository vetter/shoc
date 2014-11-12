#include "OpenCLDeviceInfo.h"
#include <iostream>
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Spmv/util.h"
#include <math.h>
#include "Event.h"
#include "support.h"

using namespace std;

// Default Block size -- note this may be adjusted
// at runtime if it's not compatible with the device's
// capabilities
static const int BLOCK_SIZE = 128;

extern const char *cl_source_spmv;

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
        for (int j=rowDelimiters[i]; j<rowDelimiters[i+1]; j++)
        {
            int col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }

}

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
    for (int i=0; i<size; i++)
    {
        if (fabs(cpuResults[i] - gpuResults[i]) / cpuResults[i]
            > MAX_RELATIVE_ERROR)
        {
#ifdef VERBOSE_OUTPUT
           cout << "Mismatch at i: "<< i << " ref: " << cpuResults[i] <<
                " dev: " << gpuResults[i] << endl;
#endif
            passed = false;
        }
    }

    if (pass != -1)
    {
        cout << "Test ";
    }
    if (passed)
    {
        cout << "Passed" << endl;
    }
    else
    {
        cout << "Failed" << endl;
    }
    return passed;
}
// ****************************************************************************
// Function: ellPackTest
//
// Purpose:
//   Runs sparse matrix vector multiplication on the device using the ellpackr
//   data format
//
// Arguements:
//   dev: opencl device id
//   ctx: current opencl context
//   copmilerFlags: flags to use when compiling ellpackr kernel
//   queue: the current opencl command queue
//   resultDB: result database to store results
//   op: provides access to command line options
//   h_val: array holding the non-zero values for the matrix
//   h_cols: array of column indices for each element of A
//   h_rowDelimiters: array of size dim+1 holding indices to rows of A;
//                  last element is the index one past the last
//                  element of A
//   h_vec: dense vector of size dim to be used for multiplication
//   h_out: input - buffer for result of calculation
//   numRows: number of rows in amtrix
//   numNonZeroes: number of entries in matrix
//   refOut: solution computed on cpu
//   padded: whether using padding or not
//   paddedSize: size of matrix when padded
//
// Programmer: Lukasz Wesolowski
// Creation: June 23, 2010
// ****************************************************************************
template <typename floatType, typename clFloatType, bool devSupportsImages>
void ellPackTest(cl_device_id dev, cl_context ctx, string compileFlags,
                 cl_command_queue queue, ResultDatabase& resultDB,
                 OptionParser& op, floatType* h_val, int* h_cols,
                 int* h_rowDelimiters, floatType* h_vec, floatType* h_out,
                 int numRows, int numNonZeroes, floatType* refOut, bool padded,
                 int paddedSize, const size_t maxImgWidth)
{
    if (devSupportsImages)
    {
        char texflags[64];
        sprintf(texflags," -DUSE_TEXTURE -DMAX_IMG_WIDTH=%ld", maxImgWidth);
        compileFlags+=string(texflags);
    }

    // Set up OpenCL Program Object
    int err = 0;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &cl_source_spmv, NULL,
            &err);
    CL_CHECK_ERROR(err);

    // Build the openCL kernels
    err = clBuildProgram(prog, 1, &dev, compileFlags.c_str(), NULL, NULL);
    CL_CHECK_ERROR(err);

    // If there is a build error, print the output and return
    if (err != CL_SUCCESS)
    {
        char log[5000];
        size_t retsize = 0;
        err = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 50000
                * sizeof(char), log, &retsize);
        CL_CHECK_ERROR(err);
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        return;
    }

    int *h_rowLengths = new int[paddedSize];
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
    floatType *h_valcm = new floatType[maxrl * cmSize];
    int *h_colscm = new int[maxrl * cmSize];
    convertToColMajor(h_val, h_cols, numRows, h_rowDelimiters, h_valcm,
                              h_colscm, h_rowLengths, maxrl, padded);

    // Device data structures
    cl_mem d_val, d_vec, d_out; // floating point
    cl_mem d_cols, d_rowLengths; // integer

    // Allocate device memory
    d_val = clCreateBuffer(ctx, CL_MEM_READ_WRITE, maxrl * cmSize *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_cols = clCreateBuffer(ctx, CL_MEM_READ_WRITE, maxrl * cmSize *
        sizeof(int), NULL, &err);
    CL_CHECK_ERROR(err);
    int imgHeight = 0;
    if (devSupportsImages)
    {
        imgHeight=(numRows+maxImgWidth-1)/maxImgWidth;
        cl_image_format fmt; fmt.image_channel_data_type=CL_FLOAT;
        if(sizeof(floatType)==4)
        fmt.image_channel_order=CL_R;
        else
        fmt.image_channel_order=CL_RG;
        d_vec = clCreateImage2D( ctx, CL_MEM_READ_ONLY, &fmt, maxImgWidth,
            imgHeight, 0, NULL, &err);
        CL_CHECK_ERROR(err);
    } else {
        d_vec = clCreateBuffer(ctx, CL_MEM_READ_WRITE, numRows *
            sizeof(clFloatType), NULL, &err);
        CL_CHECK_ERROR(err);
    }
    d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, paddedSize *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_rowLengths = clCreateBuffer(ctx, CL_MEM_READ_WRITE, cmSize *
        sizeof(int), NULL, &err);
    CL_CHECK_ERROR(err);

    // Setup events for timing
    Event valTransfer("transfer Val data over PCIe bus");
    Event colsTransfer("transfer cols data over PCIe bus");
    Event vecTransfer("transfer vec data over PCIe bus");
    Event rowLengthsTransfer("transfer rowLengths data over PCIe bus");

    // Transfer data to device
    err = clEnqueueWriteBuffer(queue, d_val, true, 0, maxrl * cmSize *
        sizeof(clFloatType), h_valcm, 0, NULL, &valTransfer.CLEvent());
    CL_CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, d_cols, true, 0, maxrl * cmSize *
        sizeof(cl_int), h_colscm, 0, NULL, &colsTransfer.CLEvent());
    CL_CHECK_ERROR(err);

    if (devSupportsImages)
    {
        size_t offset[3]={0};
        size_t size[3]={maxImgWidth,(size_t)imgHeight,1};
        err = clEnqueueWriteImage(queue,d_vec, true, offset, size,
            0, 0, h_vec, 0, NULL, &vecTransfer.CLEvent());
        CL_CHECK_ERROR(err);
    } else {
        err = clEnqueueWriteBuffer(queue, d_vec, true, 0, numRows *
            sizeof(clFloatType), h_vec, 0, NULL, &vecTransfer.CLEvent());
        CL_CHECK_ERROR(err);
    }

    err = clEnqueueWriteBuffer(queue, d_rowLengths, true, 0, cmSize *
        sizeof(int), h_rowLengths, 0, NULL, &rowLengthsTransfer.CLEvent());
    CL_CHECK_ERROR(err);

    err = clFinish(queue);
    CL_CHECK_ERROR(err);

    valTransfer.FillTimingInfo();
    colsTransfer.FillTimingInfo();
    vecTransfer.FillTimingInfo();
    rowLengthsTransfer.FillTimingInfo();

    double iTransferTime =  valTransfer.StartEndRuntime() +
                           colsTransfer.StartEndRuntime() +
                            vecTransfer.StartEndRuntime() +
                     rowLengthsTransfer.StartEndRuntime();

    // Set up kernel arguments
    cl_kernel ellpackr = clCreateKernel(prog, "spmv_ellpackr_kernel", &err);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(ellpackr, 0, sizeof(cl_mem), (void*) &d_val);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(ellpackr, 1, sizeof(cl_mem), (void*) &d_vec);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(ellpackr, 2, sizeof(cl_mem), (void*) &d_cols);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(ellpackr, 3, sizeof(cl_mem), (void*) &d_rowLengths);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(ellpackr, 4, sizeof(cl_int), (void*) &cmSize);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(ellpackr, 5, sizeof(cl_mem), (void*) &d_out);
    CL_CHECK_ERROR(err);

    const size_t globalWorkSize = cmSize;
    const size_t localWorkSize = BLOCK_SIZE;
    Event kernelExec("ELLPACKR Kernel Execution");

    int passes = op.getOptionInt("passes");
    int iters  = op.getOptionInt("iterations");

    for (int k = 0; k < passes; k++)
    {
        double totalKernelTime = 0.0;
        for (int j = 0; j < iters; j++)
        {
            err = clEnqueueNDRangeKernel(queue, ellpackr, 1, NULL,
                &globalWorkSize, &localWorkSize, 0, NULL,
                &kernelExec.CLEvent());
            CL_CHECK_ERROR(err);
            err = clFinish(queue);
            CL_CHECK_ERROR(err);
            kernelExec.FillTimingInfo();
            totalKernelTime += kernelExec.StartEndRuntime();
        }

         Event outTransfer("d->h data transfer");
         err = clEnqueueReadBuffer(queue, d_out, true, 0, numRows *
             sizeof(clFloatType), h_out, 0, NULL, &outTransfer.CLEvent());
         CL_CHECK_ERROR(err);
         err = clFinish(queue);
         CL_CHECK_ERROR(err);
         outTransfer.FillTimingInfo();
         double oTransferTime = outTransfer.StartEndRuntime();

        // Compare reference solution to GPU result
        if (! verifyResults(refOut, h_out, numRows, k)) {
            return;
        }
        char atts[TEMP_BUFFER_SIZE];
        char benchName[TEMP_BUFFER_SIZE];
        double avgTime = totalKernelTime / (double)iters;
        sprintf(atts, "%d_elements_%d_rows", numNonZeroes, cmSize);
        double gflop = 2 * (double) numNonZeroes;
        bool dpTest = (sizeof(floatType) == sizeof(double));
        sprintf(benchName, "%sELLPACKR-%s", padded ? "Padded_":"",
                dpTest ? "DP":"SP");
        resultDB.AddResult(benchName, atts, "Gflop/s", gflop/avgTime);
        sprintf(benchName, "%s_PCIe", benchName);
        resultDB.AddResult(benchName, atts, "Gflop/s", gflop /
            (avgTime + iTransferTime + oTransferTime));
    }

    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(ellpackr);
    CL_CHECK_ERROR(err);

    // Free device memory
    err = clReleaseMemObject(d_rowLengths);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_vec);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_out);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_val);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_cols);
    CL_CHECK_ERROR(err);

    // Free host memory
    delete[] h_rowLengths;
    delete[] h_valcm;
    delete[] h_colscm;
}
// ****************************************************************************
// Function: csrTest
//
// Purpose:
//   Runs sparse matrix vector multiplication on the device using the compressed
//   sparse row format
//
// Arguements:
//   dev: opencl device id
//   ctx: current opencl context
//   copmilerFlags: flags to use when compiling ellpackr kernel
//   queue: the current opencl command queue
//   resultDB: result database to store results
//   op: provides access to command line options
//   h_val: array holding the non-zero values for the matrix
//   h_cols: array of column indices for each element of A
//   h_rowDelimiters: array of size dim+1 holding indices to rows of A;
//                  last element is the index one past the last
//                  element of A
//   h_vec: dense vector of size dim to be used for multiplication
//   h_out: input - buffer for result of calculation
//   numRows: number of rows in amtrix
//   numNonZeroes: number of entries in matrix
//   refOut: solution computed on cpu
//   padded: whether using padding or not
//
// Programmer: Lukasz Wesolowski
// Creation: June 23, 2010
// ****************************************************************************
template <typename floatType, typename clFloatType, bool devSupportsImages>
void csrTest(cl_device_id dev, cl_context ctx, string compileFlags,
             cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
             floatType* h_val, int* h_cols, int* h_rowDelimiters,
             floatType* h_vec, floatType* h_out, int numRows, int numNonZeroes,
             floatType* refOut, bool padded, const size_t maxImgWidth)
{
    if (devSupportsImages)
    {
        char texflags[64];
        sprintf(texflags," -DUSE_TEXTURE -DMAX_IMG_WIDTH=%ld", maxImgWidth);
        compileFlags+=string(texflags);
    }
    // Set up OpenCL Program Object
    int err = 0;

    cl_program prog = clCreateProgramWithSource(ctx, 1, &cl_source_spmv, NULL,
            &err);
    CL_CHECK_ERROR(err);

    // Build the openCL kernels
    err = clBuildProgram(prog, 1, &dev, compileFlags.c_str(), NULL, NULL);
    // CL_CHECK_ERROR(err);  // if we check and fail here, we never get to see
                            // the OpenCL compiler's build log

    // If there is a build error, print the output and return
    if (err != CL_SUCCESS)
    {
        char log[5000];
        size_t retsize = 0;
        err = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 5000
                * sizeof(char), log, &retsize);
        CL_CHECK_ERROR(err);
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        return;
    }

      // Device data structures
      cl_mem d_val, d_vec, d_out;
      cl_mem d_cols, d_rowDelimiters;

      // Allocate device memory
      d_val = clCreateBuffer(ctx, CL_MEM_READ_WRITE, numNonZeroes *
          sizeof(clFloatType), NULL, &err);
      CL_CHECK_ERROR(err);
      d_cols = clCreateBuffer(ctx, CL_MEM_READ_WRITE, numNonZeroes *
          sizeof(cl_int), NULL, &err);
      CL_CHECK_ERROR(err);
      int imgHeight = 0;
      if (devSupportsImages)
      {
          imgHeight=(numRows+maxImgWidth-1)/maxImgWidth;
          cl_image_format fmt; fmt.image_channel_data_type=CL_FLOAT;
          if(sizeof(floatType)==4)
              fmt.image_channel_order=CL_R;
          else
              fmt.image_channel_order=CL_RG;
          d_vec = clCreateImage2D( ctx, CL_MEM_READ_ONLY, &fmt, maxImgWidth,
              imgHeight, 0, NULL, &err);
          CL_CHECK_ERROR(err);
      } else {
          d_vec = clCreateBuffer(ctx, CL_MEM_READ_WRITE, numRows *
              sizeof(clFloatType), NULL, &err);
          CL_CHECK_ERROR(err);
      }
      d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, numRows *
          sizeof(clFloatType), NULL, &err);
      CL_CHECK_ERROR(err);
      d_rowDelimiters = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (numRows+1) *
          sizeof(cl_int), NULL, &err);
      CL_CHECK_ERROR(err);

      // Setup events for timing
      Event valTransfer("transfer Val data over PCIe bus");
      Event colsTransfer("transfer cols data over PCIe bus");
      Event vecTransfer("transfer vec data over PCIe bus");
      Event rowDelimitersTransfer("transfer rowDelimiters data over PCIe bus");

      // Transfer data to device
      err = clEnqueueWriteBuffer(queue, d_val, true, 0, numNonZeroes *
          sizeof(floatType), h_val, 0, NULL, &valTransfer.CLEvent());
      CL_CHECK_ERROR(err);
      err = clEnqueueWriteBuffer(queue, d_cols, true, 0, numNonZeroes *
          sizeof(int), h_cols, 0, NULL, &colsTransfer.CLEvent());
      CL_CHECK_ERROR(err);

      if (devSupportsImages)
      {
          size_t offset[3]={0};
          size_t size[3]={maxImgWidth,(size_t)imgHeight,1};
          err = clEnqueueWriteImage(queue,d_vec, true, offset, size,
              0, 0, h_vec, 0, NULL, &vecTransfer.CLEvent());
          CL_CHECK_ERROR(err);
      } else
      {
          err = clEnqueueWriteBuffer(queue, d_vec, true, 0, numRows *
              sizeof(floatType), h_vec, 0, NULL, &vecTransfer.CLEvent());
          CL_CHECK_ERROR(err);
      }

      err = clEnqueueWriteBuffer(queue, d_rowDelimiters, true, 0, (numRows+1) *
          sizeof(int), h_rowDelimiters, 0, NULL,
          &rowDelimitersTransfer.CLEvent());
      CL_CHECK_ERROR(err);
      err = clFinish(queue);
      CL_CHECK_ERROR(err);

      valTransfer.FillTimingInfo();
      colsTransfer.FillTimingInfo();
      vecTransfer.FillTimingInfo();
      rowDelimitersTransfer.FillTimingInfo();

      double iTransferTime = valTransfer.StartEndRuntime() +
                            colsTransfer.StartEndRuntime() +
                             vecTransfer.StartEndRuntime() +
                   rowDelimitersTransfer.StartEndRuntime();

      int passes = op.getOptionInt("passes");
      int iters  = op.getOptionInt("iterations");

      // Results description info
      char atts[TEMP_BUFFER_SIZE];
      sprintf(atts, "%d_elements_%d_rows", numNonZeroes, numRows);
      string prefix = "";
      prefix += (padded) ? "Padded_" : "";
      double gflop = 2 * (double) numNonZeroes;
      cout << "CSR Scalar Kernel\n";
      Event kernelExec("kernel Execution");

      // Set up CSR Kernels
      cl_kernel csrScalar, csrVector;
      csrScalar  = clCreateKernel(prog, "spmv_csr_scalar_kernel", &err);
      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrScalar, 0, sizeof(cl_mem), (void*) &d_val);
      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrScalar, 1, sizeof(cl_mem), (void*) &d_vec);
      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrScalar, 2, sizeof(cl_mem), (void*) &d_cols);
      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrScalar, 3, sizeof(cl_mem),
          (void*) &d_rowDelimiters);
      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrScalar, 4, sizeof(cl_int), (void*) &numRows);
      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrScalar, 5, sizeof(cl_mem), (void*) &d_out);
      CL_CHECK_ERROR(err);

      csrVector = clCreateKernel(prog, "spmv_csr_vector_kernel", &err);

      // Get preferred SIMD width
      int vecWidth = getPreferredWorkGroupSizeMultiple(ctx, csrVector);
      CL_CHECK_ERROR(err);

      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrVector, 0, sizeof(cl_mem), (void*) &d_val);
      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrVector, 1, sizeof(cl_mem), (void*) &d_vec);
      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrVector, 2, sizeof(cl_mem), (void*) &d_cols);
      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrVector, 3, sizeof(cl_mem),
          (void*) &d_rowDelimiters);
      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrVector, 4, sizeof(cl_int), (void*) &numRows);
      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrVector, 5, sizeof(cl_int), (void*) &vecWidth);
      CL_CHECK_ERROR(err);
      err = clSetKernelArg(csrVector, 6, sizeof(cl_mem), (void*) &d_out);
      CL_CHECK_ERROR(err);

      // Append correct suffix to resultsDB entry
      string suffix;
      if (sizeof(floatType) == sizeof(float))
      {
          suffix = "-SP";
      }
      else
      {
          suffix = "-DP";
      }

      const size_t scalarGlobalWSize = numRows;
      size_t localWorkSize = BLOCK_SIZE;

      for (int k = 0; k < passes; k++)
      {
          double scalarKernelTime = 0.0;
          // Run Scalar Kernel
          for (int j = 0; j < iters; j++)
          {
              err = clEnqueueNDRangeKernel(queue, csrScalar, 1, NULL,
                   &scalarGlobalWSize, &localWorkSize, 0, NULL,
                   &kernelExec.CLEvent());
              CL_CHECK_ERROR(err);
              err = clFinish(queue);
              CL_CHECK_ERROR(err);
              kernelExec.FillTimingInfo();
              scalarKernelTime += kernelExec.StartEndRuntime();
          }

          // Transfer data back to host
          Event outTransfer("d->h data transfer");
          err = clEnqueueReadBuffer(queue, d_out, true, 0, numRows *
              sizeof(floatType), h_out, 0, NULL, &outTransfer.CLEvent());
          CL_CHECK_ERROR(err);
          err = clFinish(queue);
          CL_CHECK_ERROR(err);
          outTransfer.FillTimingInfo();
          double oTransferTime = outTransfer.StartEndRuntime();

          // Compare reference solution to GPU result
          if (! verifyResults(refOut, h_out, numRows, k))
          {
               return;  // If results don't match, don't report performance
          }
          scalarKernelTime = scalarKernelTime / (double)iters;
          string testName = prefix+"CSR-Scalar"+suffix;
          double totalTransfer = iTransferTime + oTransferTime;
          resultDB.AddResult(testName, atts, "Gflop/s",
              gflop/(scalarKernelTime));
          resultDB.AddResult(testName+"_PCIe", atts, "Gflop/s",
              gflop / (scalarKernelTime+totalTransfer));
      }

      // Clobber correct answer, so we can be sure the vector kernel is correct
      err = clEnqueueWriteBuffer(queue, d_out, true, 0, numRows *
          sizeof(floatType), h_vec, 0, NULL, NULL);
      CL_CHECK_ERROR(err);

      cout << "CSR Vector Kernel\n";
      // Verify Local work group size
      size_t maxLocal = getMaxWorkGroupSize(ctx, csrVector);
      if (maxLocal < vecWidth)
      {
         cout << "Warning: CSRVector requires a work group size >= " << vecWidth << endl;
         cout << "Skipping this kernel." << endl;
         err = clReleaseMemObject(d_rowDelimiters);
         CL_CHECK_ERROR(err);
         err = clReleaseMemObject(d_vec);
         CL_CHECK_ERROR(err);
         err = clReleaseMemObject(d_out);
         CL_CHECK_ERROR(err);
         err = clReleaseMemObject(d_val);
         CL_CHECK_ERROR(err);
         err = clReleaseMemObject(d_cols);
         CL_CHECK_ERROR(err);
         err = clReleaseKernel(csrScalar);
         CL_CHECK_ERROR(err);
         err = clReleaseKernel(csrVector);
         CL_CHECK_ERROR(err);
         err = clReleaseProgram(prog);
         CL_CHECK_ERROR(err);
         return;
      }
      localWorkSize = vecWidth;
      while (localWorkSize+vecWidth <= maxLocal &&
          localWorkSize+vecWidth <= BLOCK_SIZE)
      {
         localWorkSize += vecWidth;
      }
      const size_t vectorGlobalWSize = numRows * vecWidth; // 1 warp per row

      for (int k = 0; k < passes; k++)
      {
          // Run Vector Kernel
          double vectorKernelTime = 0.0;
          for (int j = 0; j < iters; j++)
          {
             err = clEnqueueNDRangeKernel(queue, csrVector, 1, NULL,
                  &vectorGlobalWSize, &localWorkSize, 0, NULL,
                  &kernelExec.CLEvent());
             CL_CHECK_ERROR(err);
             err = clFinish(queue);
             CL_CHECK_ERROR(err);
             kernelExec.FillTimingInfo();
             vectorKernelTime += kernelExec.StartEndRuntime();
          }

         Event outTransfer("d->h data transfer");
         err = clEnqueueReadBuffer(queue, d_out, true, 0, numRows *
             sizeof(floatType), h_out, 0, NULL, &outTransfer.CLEvent());
         CL_CHECK_ERROR(err);
         err = clFinish(queue);
         CL_CHECK_ERROR(err);
         outTransfer.FillTimingInfo();
         double oTransferTime = outTransfer.StartEndRuntime();

          // Compare reference solution to GPU result
          if (! verifyResults(refOut, h_out, numRows, k))
          {
              return;  // If results don't match, don't report performance
          }
          vectorKernelTime = vectorKernelTime / (double)iters;
          string testName = prefix+"CSR-Vector"+suffix;
          double totalTransfer = iTransferTime + oTransferTime;
          resultDB.AddResult(testName, atts, "Gflop/s", gflop/vectorKernelTime);
          resultDB.AddResult(testName+"_PCIe", atts, "Gflop/s",
                            gflop/(vectorKernelTime+totalTransfer));
      }

      // Free device memory
      err = clReleaseMemObject(d_rowDelimiters);
      CL_CHECK_ERROR(err);
      err = clReleaseMemObject(d_vec);
      CL_CHECK_ERROR(err);
      err = clReleaseMemObject(d_out);
      CL_CHECK_ERROR(err);
      err = clReleaseMemObject(d_val);
      CL_CHECK_ERROR(err);
      err = clReleaseMemObject(d_cols);
      CL_CHECK_ERROR(err);
      err = clReleaseKernel(csrScalar);
      CL_CHECK_ERROR(err);
      err = clReleaseKernel(csrVector);
      CL_CHECK_ERROR(err);
      err = clReleaseProgram(prog);
      CL_CHECK_ERROR(err);
}

// ****************************************************************************
// Function: RunTest
//
// Purpose:
//   Executes a run of the sparse matrix - vector multiplication benchmark
//   in either single or double precision
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: stores results from the benchmark
//   op: the options parser / parameter database
//   compileFlags: used to specify either single or double precision floats
//   nRows: number of rows in generated matrix
//
// Returns:  nothing
//
// Programmer: Lukasz Wesolowski
// Creation: July 19, 2010
//
// Modifications:
//
// ****************************************************************************
template <typename floatType, typename clFloatType>
void RunTest(cl_device_id dev, cl_context ctx, cl_command_queue queue,
             ResultDatabase &resultDB, OptionParser &op, string compileFlags,
             int nRows=0)
{
    // Determine if the device is capable of using images in general
    cl_device_id device_id;
    cl_bool deviceSupportsImages;
    int err = 0;
    err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device_id),
                                &device_id, NULL);
    CL_CHECK_ERROR(err);

    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT,
            sizeof(deviceSupportsImages), &deviceSupportsImages, NULL);
    CL_CHECK_ERROR(err);

    size_t maxImgWidth = 0;
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH,
            sizeof(size_t), &maxImgWidth, NULL);
    CL_CHECK_ERROR(err);

    // Make sure our sampler type is supported
    cl_sampler sampler;
    sampler = clCreateSampler(ctx, CL_FALSE, CL_ADDRESS_NONE,
            CL_FILTER_NEAREST, &err);
    if (err != CL_SUCCESS)
    {
        cout << "Warning: Device does not support required sampler type";
        cout << " falling back to global memory\n";
        deviceSupportsImages = false;
    } else
    {
        clReleaseSampler(sampler);
    }




    // Host data structures
    // array of values in the sparse matrix
    floatType *h_val, *h_valPad;
    // array of column indices for each value in h_val
    int *h_cols, *h_colsPad;
    // array of indices to the start of each row in h_val/valPad
    int *h_rowDelimiters, *h_rowDelimitersPad;
    // Dense vector of values
    floatType *h_vec;
    // Output vector
    floatType *h_out;
    // Reference solution computed by cpu
    floatType *refOut;

    int nItems;            // number of non-zero elements in the matrix
    int nItemsPadded;
    int numRows;           // number of rows in the matrix

    // This benchmark either reads in a matrix market input file or
    // generates a random matrix
    string inFileName = op.getOptionString("mm_filename");
    if (inFileName == "random")
    {
        // If we're not opening a file, the dimension of the matrix
        // has been passed in as an argument
        numRows = nRows;
        nItems = numRows * numRows / 100; // 1% of entries will be non-zero
        float maxval = op.getOptionFloat("maxval");
        h_val = new floatType[nItems];
        h_cols = new int[nItems];
        h_rowDelimiters = new int[nRows+1];
        fill(h_val, nItems, maxval);
        initRandomMatrix(h_cols, h_rowDelimiters, nItems, numRows);
    }
    else
    {   char filename[FIELD_LENGTH];
        strcpy(filename, inFileName.c_str());
        readMatrix(filename, &h_val, &h_cols, &h_rowDelimiters,
                &nItems, &numRows);
    }

    // Final Image Check -- Make sure the image format is supported.
    int imgHeight = (numRows+maxImgWidth-1)/maxImgWidth;
    cl_image_format fmt;
    fmt.image_channel_data_type = CL_FLOAT;
    if(sizeof(floatType)==4)
    {
        fmt.image_channel_order=CL_R;
    }
    else
    {
        fmt.image_channel_order=CL_RG;
    }
    cl_mem d_vec = clCreateImage2D(ctx, CL_MEM_READ_ONLY, &fmt, maxImgWidth,
            imgHeight, 0, NULL, &err);
    if (err != CL_SUCCESS)
    {
        deviceSupportsImages = false;
    } else {
        clReleaseMemObject(d_vec);
    }

    // Set up remaining host data
    h_vec = new floatType[numRows];
    refOut = new floatType[numRows];
    h_rowDelimitersPad = new int[numRows+1];
    fill(h_vec, numRows, op.getOptionFloat("maxval"));

    // Set up the padded data structures
    int paddedSize = numRows + (PAD_FACTOR - numRows % PAD_FACTOR);
    h_out = new floatType[paddedSize];
    convertToPadded(h_val, h_cols, numRows, h_rowDelimiters, &h_valPad,
            &h_colsPad, h_rowDelimitersPad, &nItemsPadded);

    // Compute reference solution
    spmvCpu(h_val, h_cols, h_rowDelimiters, h_vec, numRows, refOut);

    // Dispatch based on whether or not device supports OpenCL images
    if (deviceSupportsImages)
    {
        cout << "CSR Test\n";
        csrTest<floatType, clFloatType, true>
            (dev, ctx, compileFlags, queue, resultDB, op, h_val, h_cols,
             h_rowDelimiters, h_vec, h_out, numRows, nItems, refOut,
             false, maxImgWidth);

        // Test CSR kernels on padded data
        cout << "CSR Test -- Padded Data\n";
        csrTest<floatType,clFloatType, true>
            (dev, ctx, compileFlags, queue, resultDB, op, h_valPad, h_colsPad,
             h_rowDelimitersPad, h_vec, h_out, numRows, nItemsPadded, refOut,
             true, maxImgWidth);

        // Test ELLPACKR kernel
        cout << "ELLPACKR Test\n";
        ellPackTest<floatType, clFloatType, true>
            (dev, ctx, compileFlags, queue, resultDB, op, h_val, h_cols,
             h_rowDelimiters, h_vec, h_out, numRows, nItems,
             refOut, false, paddedSize, maxImgWidth);
    } else {
        cout << "CSR Test\n";
        csrTest<floatType, clFloatType, false>
            (dev, ctx, compileFlags, queue, resultDB, op, h_val, h_cols,
             h_rowDelimiters, h_vec, h_out, numRows, nItems, refOut,
             false, 0);

        // Test CSR kernels on padded data
        cout << "CSR Test -- Padded Data\n";
        csrTest<floatType,clFloatType, false>
            (dev, ctx, compileFlags, queue, resultDB, op, h_valPad, h_colsPad,
             h_rowDelimitersPad, h_vec, h_out, numRows, nItemsPadded, refOut,
             true, 0);

        // Test ELLPACKR kernel
        cout << "ELLPACKR Test\n";
        ellPackTest<floatType, clFloatType, false>
            (dev, ctx, compileFlags, queue, resultDB, op, h_val, h_cols,
             h_rowDelimiters, h_vec, h_out, numRows, nItems,
             refOut, false, paddedSize, 0);
    }

    delete[] h_val;
    delete[] h_cols;
    delete[] h_rowDelimiters;
    delete[] h_vec;
    delete[] h_out;
    delete[] h_valPad;
    delete[] h_colsPad;
    delete[] h_rowDelimitersPad;
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the sparse matrix - vector multiplication benchmark
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: stores results from the benchmark
//   op: the options parser / parameter database
//
// Returns:  nothing
// Programmer: Lukasz Wesolowski
// Creation: August 13, 2010
//
// Modifications:
//
// ****************************************************************************
void
RunBenchmark(cl_device_id dev,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    //create list of problem sizes
    int probSizes[4] = {1024, 8192, 12288, 16384};
    int sizeClass = op.getOptionInt("size") - 1;

    // Always run single precision test
    // OpenCL doesn't support templated kernels, so we have to use macros
    cout <<"Single precision tests:\n";
    string spMacros = "-DSINGLE_PRECISION ";
    RunTest<float, cl_float>
        (dev, ctx, queue, resultDB, op, spMacros, probSizes[sizeClass]);

    // If double precision is supported, run the DP test
    if (checkExtension(dev, "cl_khr_fp64"))
    {
        cout << "Double precision tests\n";
        string dpMacros = "-DK_DOUBLE_PRECISION ";
        RunTest<double, cl_double>
            (dev, ctx, queue, resultDB, op, dpMacros, probSizes[sizeClass]);
    }
    else if (checkExtension(dev, "cl_amd_fp64"))
    {
        cout << "Double precision tests\n";
        string dpMacros = "-DAMD_DOUBLE_PRECISION ";
        RunTest<double, cl_double>
            (dev, ctx, queue, resultDB, op, dpMacros, probSizes[sizeClass]);
    }
    else
    {
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
    }
}
