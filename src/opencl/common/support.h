#ifndef SUPPORT_H
#define SUPPORT_H

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

// ****************************************************************************
// File:  support.h
//
// Purpose:
//   various OpenCL-related support routines
//
// Programmer:  Jeremy Meredith
// Creation:    June 12, 2009
//
// ****************************************************************************

inline const char *CLErrorString(cl_int err)
{
    switch (err)
    {
      case CL_SUCCESS:                         return "CL_SUCCESS";                         // break;
      case CL_DEVICE_NOT_FOUND:                return "CL_DEVICE_NOT_FOUND";                // break;
      case CL_DEVICE_NOT_AVAILABLE:            return "CL_DEVICE_NOT_AVAILABLE";            // break;
      case CL_COMPILER_NOT_AVAILABLE:          return "CL_COMPILER_NOT_AVAILABLE";          // break;
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:   return "CL_MEM_OBJECT_ALLOCATION_FAILURE";   // break;
      case CL_OUT_OF_RESOURCES:                return "CL_OUT_OF_RESOURCES";                // break;
      case CL_OUT_OF_HOST_MEMORY:              return "CL_OUT_OF_HOST_MEMORY";              // break;
      case CL_PROFILING_INFO_NOT_AVAILABLE:    return "CL_PROFILING_INFO_NOT_AVAILABLE";    // break;
      case CL_MEM_COPY_OVERLAP:                return "CL_MEM_COPY_OVERLAP";                // break;
      case CL_IMAGE_FORMAT_MISMATCH:           return "CL_IMAGE_FORMAT_MISMATCH";           // break;
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";      // break;
      case CL_BUILD_PROGRAM_FAILURE:           return "CL_BUILD_PROGRAM_FAILURE";           // break;
      case CL_MAP_FAILURE:                     return "CL_MAP_FAILURE";                     // break;
      case CL_INVALID_VALUE:                   return "CL_INVALID_VALUE";                   // break;
      case CL_INVALID_DEVICE_TYPE:             return "CL_INVALID_DEVICE_TYPE";             // break;
      case CL_INVALID_PLATFORM:                return "CL_INVALID_PLATFORM";                // break;
      case CL_INVALID_DEVICE:                  return "CL_INVALID_DEVICE";                  // break;
      case CL_INVALID_CONTEXT:                 return "CL_INVALID_CONTEXT";                 // break;
      case CL_INVALID_QUEUE_PROPERTIES:        return "CL_INVALID_QUEUE_PROPERTIES";        // break;
      case CL_INVALID_COMMAND_QUEUE:           return "CL_INVALID_COMMAND_QUEUE";           // break;
      case CL_INVALID_HOST_PTR:                return "CL_INVALID_HOST_PTR";                // break;
      case CL_INVALID_MEM_OBJECT:              return "CL_INVALID_MEM_OBJECT";              // break;
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; // break;
      case CL_INVALID_IMAGE_SIZE:              return "CL_INVALID_IMAGE_SIZE";              // break;
      case CL_INVALID_SAMPLER:                 return "CL_INVALID_SAMPLER";                 // break;
      case CL_INVALID_BINARY:                  return "CL_INVALID_BINARY";                  // break;
      case CL_INVALID_BUILD_OPTIONS:           return "CL_INVALID_BUILD_OPTIONS";           // break;
      case CL_INVALID_PROGRAM:                 return "CL_INVALID_PROGRAM";                 // break;
      case CL_INVALID_PROGRAM_EXECUTABLE:      return "CL_INVALID_PROGRAM_EXECUTABLE";      // break;
      case CL_INVALID_KERNEL_NAME:             return "CL_INVALID_KERNEL_NAME";             // break;
      case CL_INVALID_KERNEL_DEFINITION:       return "CL_INVALID_KERNEL_DEFINITION";       // break;
      case CL_INVALID_KERNEL:                  return "CL_INVALID_KERNEL";                  // break;
      case CL_INVALID_ARG_INDEX:               return "CL_INVALID_ARG_INDEX";               // break;
      case CL_INVALID_ARG_VALUE:               return "CL_INVALID_ARG_VALUE";               // break;
      case CL_INVALID_ARG_SIZE:                return "CL_INVALID_ARG_SIZE";                // break;
      case CL_INVALID_KERNEL_ARGS:             return "CL_INVALID_KERNEL_ARGS";             // break;
      case CL_INVALID_WORK_DIMENSION:          return "CL_INVALID_WORK_DIMENSION";          // break;
      case CL_INVALID_WORK_GROUP_SIZE:         return "CL_INVALID_WORK_GROUP_SIZE";         // break;
      case CL_INVALID_WORK_ITEM_SIZE:          return "CL_INVALID_WORK_ITEM_SIZE";          // break;
      case CL_INVALID_GLOBAL_OFFSET:           return "CL_INVALID_GLOBAL_OFFSET";           // break;
      case CL_INVALID_EVENT_WAIT_LIST:         return "CL_INVALID_EVENT_WAIT_LIST";         // break;
      case CL_INVALID_EVENT:                   return "CL_INVALID_EVENT";                   // break;
      case CL_INVALID_OPERATION:               return "CL_INVALID_OPERATION";               // break;
      case CL_INVALID_GL_OBJECT:               return "CL_INVALID_GL_OBJECT";               // break;
      case CL_INVALID_BUFFER_SIZE:             return "CL_INVALID_BUFFER_SIZE";             // break;
      case CL_INVALID_MIP_LEVEL:               return "CL_INVALID_MIP_LEVEL";               // break;
      case CL_INVALID_GLOBAL_WORK_SIZE:        return "CL_INVALID_GLOBAL_WORK_SIZE";        // break;
      case CL_INVALID_PROPERTY:                return "CL_INVALID_PROPERTY";                // break;
      default:                                 return "UNKNOWN";                            // break;
  }
}

#if defined(USE_CL_EXCEPTIONS)

#define CL_CHECK_ERROR(err) \
    { \
        if(err != CL_SUCCESS) \
        { \
            std::ostringstream msgstr; \
            msgstr << __FILE__ << ':' << __LINE__ << ": " \
                << CLErrorString(err); \
            throw SHOC_OpenCLException(err, msgstr.str()); \
        } \
    }

#else // defined(USE_CL_EXCEPTIONS)

#define CL_CHECK_ERROR(err) \
    {                       \
        if (err != CL_SUCCESS)                  \
        { \
            std::cerr << __FILE__ << ':' << __LINE__ << ": " << CLErrorString(err) << std::endl; \
            exit(1); \
        } \
    }
#endif // defined(USE_CL_EXCEPTIONS)



inline std::string DeviceTypeToString(cl_device_type type)
{
    struct {int val; const char *name;} mapping[] = {
        {CL_DEVICE_TYPE_CPU,        "CPU"},
        {CL_DEVICE_TYPE_GPU,        "GPU"},
        {CL_DEVICE_TYPE_ACCELERATOR,"Accelerator"},
        {CL_DEVICE_TYPE_DEFAULT,    "Default"},
        {0,NULL}
    };
    std::string retval = "Unknown";
    for (int i=0; mapping[i].name != NULL; i++)
    {
        if (type & mapping[i].val)
        {
            if (retval == "Unknown")
                retval = mapping[i].name;
            else
                retval += std::string(" | ") + mapping[i].name;
        }
    }
    return retval;
}

inline std::string QueuePropsToString(cl_bitfield type)
{
    struct {int val; const char *name;} mapping[] = {
        {CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, "out-of-order-exec"},
        {CL_QUEUE_PROFILING_ENABLE,              "profiling"},
        {0,NULL}
    };
    std::string retval = "Unknown";
    for (int i=0; mapping[i].name != NULL; i++)
    {
        if (type & mapping[i].val)
        {
            if (retval == "Unknown")
                retval = mapping[i].name;
            else
                retval += std::string(" | ") + mapping[i].name;
        }
    }
    return retval;
}

// ****************************************************************************
// Method:  getMaxWorkGroupSize
//
// Purpose:
//   Finds the maximum valid work group size for the specified kernel and
//   OpenCL context.
//
// Arguments:
//       ctx         OpenCL context
//       ker         OpenCL kernel
//
// Programmer:  Gabriel Marin
// Creation:    July 14, 2009
//
// ****************************************************************************
inline size_t
getMaxWorkGroupSize (cl_context &ctx, cl_kernel &ker)
{
    int err;
    // Find the maximum work group size
    size_t retSize = 0;
    size_t maxGroupSize = 0;
    // we must find the device asociated with this context first
    cl_device_id devid;   // we create contexts with a single device only
    err = clGetContextInfo (ctx, CL_CONTEXT_DEVICES, sizeof(devid), &devid, &retSize);
    CL_CHECK_ERROR(err);
    if (retSize < sizeof(devid))  // we did not get any device, pass 0 to the function
       devid = 0;
    err = clGetKernelWorkGroupInfo (ker, devid, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                &maxGroupSize, &retSize);
    CL_CHECK_ERROR(err);
    return (maxGroupSize);
}

// ****************************************************************************
// Method:  getPreferredWorkGroupSizeMultiple
//
// Purpose:
//   Finds the preferred work group size mulitple for the specified kernel and
//   OpenCL context.  This is actually a hint about the SIMD width, since the
//   workgroup should want to be a multiple of the SIMD width.
//
// Arguments:
//       ctx         OpenCL context
//       ker         OpenCL kernel
//
// Programmer:  Graham Lopez
// Creation:    September 24, 2014
//
// ****************************************************************************
inline size_t
getPreferredWorkGroupSizeMultiple (cl_context &ctx, cl_kernel &ker)
{
    int err;
    // Find the maximum work group size
    size_t retSize = 0;
    size_t prefGroupSize = 0;
    // we must find the device asociated with this context first
    cl_device_id devid;   // we create contexts with a single device only
    err = clGetContextInfo (ctx, CL_CONTEXT_DEVICES, sizeof(devid), &devid, &retSize);
    CL_CHECK_ERROR(err);
    if (retSize < sizeof(devid))  // we did not get any device, pass 0 to the function
       devid = 0;
    err = clGetKernelWorkGroupInfo (ker, devid, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                sizeof(size_t), &prefGroupSize, &retSize);
    CL_CHECK_ERROR(err);
    return (prefGroupSize);
}

// ****************************************************************************
// Method:  getDeviceMaxWorkGroupSize
//
// Purpose:
//   Finds the maximum valid work group size for the specified device.
//
// Arguments:
//       devid         OpenCL device
//
// Programmer:  Gabriel Marin
// Creation:    November 17, 2010
//
// ****************************************************************************
inline size_t
getMaxWorkGroupSize (cl_device_id devid)
{
    // Find the maximum work group size
    size_t maxWorkGroupSize = 0;
    clGetDeviceInfo (devid, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                           sizeof(size_t), &maxWorkGroupSize, NULL);
    return (maxWorkGroupSize);
}

// ****************************************************************************
// Method:  getMaxComputeUnits
//
// Purpose:
//   Find the maximum number of compute units (SMs).
//
// Arguments:
//       devid       OpenCL device
//
// Programmer:  Gabriel Marin
// Creation:    June 8, 2010
//
// ****************************************************************************
inline cl_uint
getMaxComputeUnits (cl_device_id devid)
{
    // Find the maximum work group size
    cl_uint maxComputeUnits = 0;
    clGetDeviceInfo (devid, CL_DEVICE_MAX_COMPUTE_UNITS,
                           sizeof(cl_uint), &maxComputeUnits, NULL);
    return (maxComputeUnits);
}

// ****************************************************************************
// Method:  getLocalMemSize
//
// Purpose:
//   Find the size of local memory.
//
// Arguments:
//       devid       OpenCL device
//
// Programmer:  Gabriel Marin
// Creation:    June 8, 2010
//
// ****************************************************************************
inline cl_ulong
getLocalMemSize (cl_device_id devid)
{
    // Find the maximum work group size
    cl_ulong localMemSize = 0;
    clGetDeviceInfo (devid, CL_DEVICE_LOCAL_MEM_SIZE,
                           sizeof(cl_ulong), &localMemSize, NULL);
    return (localMemSize);
}

// ****************************************************************************
// Method:  oclGetProgBinary
//
// Purpose:
//   Get the binary (PTX) of the program associated with the device
//
// Arguments:
//       cpProgram    OpenCL program
//       cdDevice     device of interest
//       binary       returned code
//       length       length of returned code
//
// Copyright 1993-2009 NVIDIA Corporation
//
// ****************************************************************************
inline void
oclGetProgBinary (cl_program cpProgram, cl_device_id cdDevice, char** binary, size_t* length)
{
    // Grab the number of devices associated witht the program
    cl_uint num_devices;
    clGetProgramInfo(cpProgram, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);

    // Grab the device ids
    cl_device_id* devices = (cl_device_id*) malloc(num_devices * sizeof(cl_device_id));
    clGetProgramInfo(cpProgram, CL_PROGRAM_DEVICES, num_devices * sizeof(cl_device_id), devices, 0);

    // Grab the sizes of the binaries
    size_t* binary_sizes = (size_t*)malloc(num_devices * sizeof(size_t));
    clGetProgramInfo(cpProgram, CL_PROGRAM_BINARY_SIZES, num_devices * sizeof(size_t), binary_sizes, NULL);

    // Now get the binaries
    char** ptx_code = (char**) malloc(num_devices * sizeof(char*));
    for( unsigned int i=0; i<num_devices; ++i) {
        ptx_code[i]= (char*)malloc(binary_sizes[i]);
    }
    clGetProgramInfo(cpProgram, CL_PROGRAM_BINARIES, 0, ptx_code, NULL);

    // Find the index of the device of interest
    unsigned int idx = 0;
    while( idx<num_devices && devices[idx] != cdDevice ) ++idx;

    // If it is associated prepare the result
    if( idx < num_devices )
    {
        *binary = ptx_code[idx];
        *length = binary_sizes[idx];
    }

    // Cleanup
    free( devices );
    free( binary_sizes );
    for( unsigned int i=0; i<num_devices; ++i) {
        if( i != idx ) free(ptx_code[i]);
    }
    free( ptx_code );
}

// ****************************************************************************
// Method:  oclGetFirstDev
//
// Purpose:
//   Gets the id of the first device from the context
//
// Arguments:
//       cxMainContext         OpenCL context
//
// Copyright 1993-2009 NVIDIA Corporation
//
// ****************************************************************************
inline cl_device_id
oclGetFirstDev(cl_context cxMainContext)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxMainContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);

    clGetContextInfo(cxMainContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    cl_device_id first = cdDevices[0];
    free(cdDevices);

    return first;
}


// ****************************************************************************
// Method:  dumpPTXCode
//
// Purpose:
//
//
// Arguments:
//   ctx          context
//   prg          program
//   name         file name prefix to output to
//
// Programmer:  Gabriel Marin
// Creation:    July 14, 2009
//
// ****************************************************************************
inline bool
dumpPTXCode (cl_context ctx, cl_program prog, const char *name)
{
    std::cout << "Dumping the PTX code" << std::endl;
    size_t ptx_length;
    char* ptx_code;
    char buf[64];
    oclGetProgBinary (prog, oclGetFirstDev(ctx), &ptx_code, &ptx_length);

    FILE* ptxFile = NULL;
    sprintf (buf, "%.59s.ptx", name);
#ifdef WIN32
    fopen_s (&ptxFile, buf, "w");
#else
    ptxFile = fopen (buf,"w");
#endif
    if (ptxFile)
    {
        fwrite (ptx_code, ptx_length, 1, ptxFile);
        fclose (ptxFile);
    }
    free (ptx_code);
    return (ptx_code!=0);
}


// ****************************************************************************
// Method:  findAvailBytes
//
// Purpose: returns maximum number of bytes *allocatable* (likely less than
//          device memory size) on the device.
//
// Arguments:
//   device   id of the device
//
// Programmer:  Collin McCurdy
// Creation:    June 8, 2010
//
// ****************************************************************************
inline
unsigned long
findAvailBytes( cl_device_id devID )
{
    cl_int err;

    cl_ulong avail_bytes = 0;
    err = clGetDeviceInfo( devID,
                            CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            sizeof(avail_bytes),
                            &avail_bytes,
                            NULL );
    CL_CHECK_ERROR(err);

    return avail_bytes;
}


inline
bool
checkExtension( cl_device_id devID, const std::string& ext )
{
    cl_int err;

    size_t nBytesNeeded = 0;
    err = clGetDeviceInfo( devID,
                        CL_DEVICE_EXTENSIONS,
                        0,
                        NULL,
                        &nBytesNeeded );
    CL_CHECK_ERROR(err);
    char* extensions = new char[nBytesNeeded+1];
    err = clGetDeviceInfo( devID,
                        CL_DEVICE_EXTENSIONS,
                        nBytesNeeded + 1,
                        extensions,
                        NULL );

    std::string extString = extensions;
    delete[] extensions;

    return (extString.find(ext) != std::string::npos);
}



#endif

