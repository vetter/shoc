#include "OpenCLDeviceInfo.h"
#include "Utility.h"
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace std;

namespace SHOC {
    const int OpenCLDeviceInfo::MAGIC_KEY_DEVICE_INFO = 0x7B16C9D8;

    static const int END_ENUMERATION_MARKER = -1;

    // cl_uint
    int clIntCharacteristics[] = {
        CL_DEVICE_VENDOR_ID
        ,CL_DEVICE_MAX_COMPUTE_UNITS
        ,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
        ,CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR
        ,CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT
        ,CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT
        ,CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG
        ,CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT
        ,CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE
        ,CL_DEVICE_MAX_CLOCK_FREQUENCY
        ,CL_DEVICE_MAX_READ_IMAGE_ARGS
        ,CL_DEVICE_MAX_WRITE_IMAGE_ARGS
        ,CL_DEVICE_MAX_SAMPLERS
        ,CL_DEVICE_MEM_BASE_ADDR_ALIGN
        ,CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE
        ,CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
        ,CL_DEVICE_MAX_CONSTANT_ARGS
        ,END_ENUMERATION_MARKER
    };
    const char* clIntCharacteristicsNames[] = {
        "VendorId"
        ,"MaxComputeUnits"
        ,"MaxWorkItemDimensions"
        ,"PreferredVectorWidthChar"
        ,"PreferredVectorWidthShort"
        ,"PreferredVectorWidthInt"
        ,"PreferredVectorWidthLong"
        ,"PreferredVectorWidthFloat"
        ,"PreferredVectorWidthDouble"
        ,"MaxClockFrequency"
        ,"MaxReadImageArgs"
        ,"MaxWriteImageArgs"
        ,"MaxSamplers"
        ,"MemBaseAddrAlign"
        ,"MinDataTypeAlignSize"
        ,"GlobalMemCachelineSize"
        ,"MaxConstantArgs"
    };

    // size_t
    int clSizeTCharacteristics[] = {
        CL_DEVICE_MAX_WORK_GROUP_SIZE
        ,CL_DEVICE_IMAGE2D_MAX_WIDTH
        ,CL_DEVICE_IMAGE2D_MAX_HEIGHT
        ,CL_DEVICE_IMAGE3D_MAX_WIDTH
        ,CL_DEVICE_IMAGE3D_MAX_HEIGHT
        ,CL_DEVICE_IMAGE3D_MAX_DEPTH
        ,CL_DEVICE_MAX_PARAMETER_SIZE
        ,CL_DEVICE_PROFILING_TIMER_RESOLUTION
        ,END_ENUMERATION_MARKER
    };
    const char* clSizeTCharacteristicsNames[] = {
        "MaxWorkGroupSize"
        ,"Image2dMaxWidth"
        ,"Image2dMaxHeight"
        ,"Image3dMaxWidth"
        ,"Image3dMaxHeight"
        ,"Image3dMaxDepth"
        ,"MaxParameterSize"
        ,"ProfilingTimerResolution(ns)"
    };

    // cl_ulong
    int clLongCharacteristics[] = {
        CL_DEVICE_MAX_MEM_ALLOC_SIZE
        ,CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
        ,CL_DEVICE_GLOBAL_MEM_SIZE
        ,CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
        ,CL_DEVICE_LOCAL_MEM_SIZE
        ,END_ENUMERATION_MARKER
    };
    const char* clLongCharacteristicsNames[] = {
        "MaxMemAllocSize"
        ,"GlobalMemCacheSize"
        ,"GlobalMemSize"
        ,"MaxConstantBufferSize"
        ,"LocalMemSize"
    };

    // cl_bool
    int clBoolCharacteristics[] = {
        CL_DEVICE_IMAGE_SUPPORT
        ,CL_DEVICE_ERROR_CORRECTION_SUPPORT
        ,CL_DEVICE_ENDIAN_LITTLE
        ,CL_DEVICE_AVAILABLE
        ,CL_DEVICE_COMPILER_AVAILABLE
        ,END_ENUMERATION_MARKER
    };
    const char* clBoolCharacteristicsNames[] = {
        "ImageSupport"
        ,"ErrorCorrectionSupport"
        ,"LittleEndian"
        ,"DeviceAvailable"
        ,"CompilerAvailable"
    };

    // char[]
    int clStringCharacteristics[] = {
        CL_DEVICE_NAME
        ,CL_DEVICE_VENDOR
        ,CL_DRIVER_VERSION
        ,CL_DEVICE_PROFILE
        ,CL_DEVICE_VERSION
        ,CL_DEVICE_EXTENSIONS
        ,END_ENUMERATION_MARKER
    };
    const char* clStringCharacteristicsNames[] = {
        "DeviceName"
        ,"DeviceVendor"
        ,"DriverVersion"
        ,"DeviceProfile"
        ,"DeviceVersion"
        ,"DeviceExtensions"
    };

    int fpPropertiesList[] = {
        CL_FP_DENORM
        ,CL_FP_INF_NAN
        ,CL_FP_ROUND_TO_NEAREST
        ,CL_FP_ROUND_TO_ZERO
        ,CL_FP_ROUND_TO_INF
        ,CL_FP_FMA
        ,END_ENUMERATION_MARKER
    };
    const char* fpPropertiesNames[] = {
        "Denormals"
        ,"InfAndQuietNANs"
        ,"RoundToNearest"
        ,"RoundToZero"
        ,"RoundToInf"
        ,"FusedMultiplyAdds"
    };

    int typePropertiesList[] = {
        CL_DEVICE_TYPE_CPU
        ,CL_DEVICE_TYPE_GPU
        ,CL_DEVICE_TYPE_ACCELERATOR
        ,CL_DEVICE_TYPE_DEFAULT
        ,END_ENUMERATION_MARKER
    };
    const char* typePropertiesNames[] = {
        "CPU"
        ,"GPU"
        ,"Accelerator"
        ,"default"
    };

    int addressBitsPropertiesList[] = {
#ifdef CL_DEVICE_ADDRESS_32_BITS
        CL_DEVICE_ADDRESS_32_BITS,
        CL_DEVICE_ADDRESS_64_BITS,
#endif
        END_ENUMERATION_MARKER
    };
    const char* addressBitsPropertiesNames[] = {
        "32bitAddressSpace"
        ,"64bitAddressSpace"
    };

    int cacheTypePropertiesList[] = {
        CL_NONE
        ,CL_READ_ONLY_CACHE
        ,CL_READ_WRITE_CACHE
        ,END_ENUMERATION_MARKER
    };
    const char* cacheTypePropertiesNames[] = {
        "None"
        ,"ReadOnly"
        ,"ReadWrite"
    };

    int localMemTypePropertiesList[] = {
        CL_LOCAL
        ,CL_GLOBAL
        ,END_ENUMERATION_MARKER
    };
    const char* localMemTypePropertiesNames[] = {
        "LocalSRAM"
        ,"Global"
    };

    int execPropertiesList[] = {
        CL_EXEC_KERNEL
        ,CL_EXEC_NATIVE_KERNEL
        ,END_ENUMERATION_MARKER
    };
    const char* execPropertiesNames[] = {
        "OpenCL"
        ,"Native"
    };

    int queuePropertiesList[] = {
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
        ,CL_QUEUE_PROFILING_ENABLE
        ,END_ENUMERATION_MARKER
    };
    const char* queuePropertiesNames[] = {
        "OutOfOrderExecution"
        ,"Profiling"
    };
};

using namespace SHOC;

// ****************************************************************************
// Method: OpenCLDeviceInfo::OpenCLDeviceInfo
//
// Purpose:
//   Constructor. Creates a new OpenCLDeviceInfo object that is either
//   instantiated with data for the device with the specified ID, or
//   left uninitialized if the device ID is not set.
//
// Arguments:
//   deviceId : the OpenCL device ID.
//
// Returns:
//
// Note:
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
OpenCLDeviceInfo::OpenCLDeviceInfo (cl_device_id deviceId) : id(deviceId)
{
    int i, len;
    numIntValues = 0;
    numSizeTValues = 0;
    numLongValues = 0;
    numBoolValues = 0;
    numStringValues = 0;
    maxKeyLength = 10;

    hashKey = 0;

    for (i=0 ; clIntCharacteristics[i]!=END_ENUMERATION_MARKER ; ++i) {
        numIntValues += 1;
        if (maxKeyLength<(len=strlen(clIntCharacteristicsNames[i])))
            maxKeyLength = len;
    }

    for (i=0 ; clSizeTCharacteristics[i]!=END_ENUMERATION_MARKER ; ++i) {
        numSizeTValues += 1;
        if (maxKeyLength<(len=strlen(clSizeTCharacteristicsNames[i])))
            maxKeyLength = len;
    }

    for (i=0 ; clLongCharacteristics[i]!=END_ENUMERATION_MARKER ; ++i) {
        numLongValues += 1;
        if (maxKeyLength<(len=strlen(clLongCharacteristicsNames[i])))
            maxKeyLength = len;
    }

    for (i=0 ; clBoolCharacteristics[i]!=END_ENUMERATION_MARKER ; ++i) {
        numBoolValues += 1;
        if (maxKeyLength<(len=strlen(clBoolCharacteristicsNames[i])))
            maxKeyLength = len;
    }

    for (i=0 ; clStringCharacteristics[i]!=END_ENUMERATION_MARKER ; ++i) {
        numStringValues += 1;
        if (maxKeyLength<(len=strlen(clStringCharacteristicsNames[i])))
            maxKeyLength = len;
    }

    // allocate space for the values of all these characteristics
    intValues = new cl_uint[numIntValues];
    sizeTValues = new size_t[numSizeTValues];
    longValues = new cl_ulong[numLongValues];
    boolValues = new cl_bool[numBoolValues];
    stringValues = new string[numStringValues];

    numDimensions = 1;
    maxWorkSizes = 0;

#ifdef CL_DEVICE_HALF_FP_CONFIG
    hasHalfFp = 1;
#else
    hasHalfFp = 0;
#endif
    hasDoubleFp = (checkExtension(id, "cl_khr_fp64") || checkExtension(id, "cl_amd_fp64"));

    if (id != 0)
       FillDeviceInformation ();
}

// ****************************************************************************
// Method: OpenCLDeviceInfo::OpenCLDeviceInfo
//
// Purpose:
//   Copy constructor: creates a new OpenCL device that is a copy
//   of the specified device.
//
// Arguments:
//   fdi : the OpenCLDeviceInfo object to be cloned.
//
// Returns:
//
// Note:
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
OpenCLDeviceInfo::OpenCLDeviceInfo(const OpenCLDeviceInfo& fdi)
{
    int i;
    id = fdi.id;
    hashKey = fdi.hashKey;

    numIntValues = fdi.numIntValues;
    numSizeTValues = fdi.numSizeTValues;
    numLongValues = fdi.numLongValues;
    numBoolValues = fdi.numBoolValues;
    numStringValues = fdi.numStringValues;
    maxKeyLength = fdi.maxKeyLength;

    // allocate space for the values of all these characteristics
    intValues = new cl_uint[numIntValues];
    sizeTValues = new size_t[numSizeTValues];
    longValues = new cl_ulong[numLongValues];
    boolValues = new cl_bool[numBoolValues];
    stringValues = new string[numStringValues];

    for (i=0 ; i<numIntValues ; ++i) intValues[i] = fdi.intValues[i];
    for (i=0 ; i<numSizeTValues ; ++i) sizeTValues[i] = fdi.sizeTValues[i];
    for (i=0 ; i<numLongValues ; ++i) longValues[i] = fdi.longValues[i];
    for (i=0 ; i<numBoolValues ; ++i) boolValues[i] = fdi.boolValues[i];
    for (i=0 ; i<numStringValues ; ++i) stringValues[i] = fdi.stringValues[i];

    numDimensions = fdi.numDimensions;
    if (fdi.maxWorkSizes) {
        maxWorkSizes = new size_t[numDimensions];
        for (i=0 ; i<numDimensions ; ++i)
            maxWorkSizes[i] = fdi.maxWorkSizes[i];
    }
    else
        maxWorkSizes = 0;

    hasHalfFp = fdi.hasHalfFp;
    hasDoubleFp = fdi.hasDoubleFp;

    // copy also all the string values
    typeValue = fdi.typeValue;
    addrBitsValue = fdi.addrBitsValue;
    cacheTypeValue = fdi.cacheTypeValue;
    localMemValue = fdi.localMemValue;
    execCapabilitiesValue = fdi.execCapabilitiesValue;
    queuePropertiesValue = fdi.queuePropertiesValue;
    halfFpValue = fdi.halfFpValue;
    singleFpValue = fdi.singleFpValue;
    doubleFpValue = fdi.doubleFpValue;
}

// ****************************************************************************
// Method: OpenCLDeviceInfo::operator=
//
// Purpose:
//   Copy operator: copy the details of the specified device into
//   this object.
//
// Arguments:
//   fdi : the OpenCLDeviceInfo object to be copied.
//
// Returns: a reference to this device object
//
// Note:
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
OpenCLDeviceInfo&
OpenCLDeviceInfo::operator= (const OpenCLDeviceInfo& fdi)
{
    int i;
    id = fdi.id;
    hashKey = fdi.hashKey;

    // allocate space for the values of all these characteristics
    if (numIntValues != fdi.numIntValues) {
        delete[] intValues;
        numIntValues = fdi.numIntValues;
        intValues = new cl_uint[numIntValues];
    }
    if (numSizeTValues != fdi.numSizeTValues) {
        delete[] sizeTValues;
        numSizeTValues = fdi.numSizeTValues;
        sizeTValues = new size_t[numSizeTValues];
    }
    if (numLongValues != fdi.numLongValues) {
        delete[] longValues;
        numLongValues = fdi.numLongValues;
        longValues = new cl_ulong[numLongValues];
    }
    if (numBoolValues != fdi.numBoolValues) {
        delete[] boolValues;
        numBoolValues = fdi.numBoolValues;
        boolValues = new cl_bool[numBoolValues];
    }
    if (numStringValues != fdi.numStringValues) {
        delete[] stringValues;
        numStringValues = fdi.numStringValues;
        stringValues = new string[numStringValues];
    }

    maxKeyLength = fdi.maxKeyLength;

    for (i=0 ; i<numIntValues ; ++i) intValues[i] = fdi.intValues[i];
    for (i=0 ; i<numSizeTValues ; ++i) sizeTValues[i] = fdi.sizeTValues[i];
    for (i=0 ; i<numLongValues ; ++i) longValues[i] = fdi.longValues[i];
    for (i=0 ; i<numBoolValues ; ++i) boolValues[i] = fdi.boolValues[i];
    for (i=0 ; i<numStringValues ; ++i) stringValues[i] = fdi.stringValues[i];

    if (maxWorkSizes && (numDimensions!=fdi.numDimensions || !fdi.maxWorkSizes)) {
        delete[] maxWorkSizes;
    }
    numDimensions = fdi.numDimensions;
    if (fdi.maxWorkSizes) {
        maxWorkSizes = new size_t[numDimensions];
        for (i=0 ; i<numDimensions ; ++i)
            maxWorkSizes[i] = fdi.maxWorkSizes[i];
    }
    else
        maxWorkSizes = 0;

    hasHalfFp = fdi.hasHalfFp;
    hasDoubleFp = fdi.hasDoubleFp;

    // copy also all the string values
    typeValue = fdi.typeValue;
    addrBitsValue = fdi.addrBitsValue;
    cacheTypeValue = fdi.cacheTypeValue;
    localMemValue = fdi.localMemValue;
    execCapabilitiesValue = fdi.execCapabilitiesValue;
    queuePropertiesValue = fdi.queuePropertiesValue;
    halfFpValue = fdi.halfFpValue;
    singleFpValue = fdi.singleFpValue;
    doubleFpValue = fdi.doubleFpValue;

    return (*this);
}

// ****************************************************************************
// Method: OpenCLDeviceInfo::operator<
//
// Purpose:
//   Less operator: compares two OpenCLDeviceInfo objects based on
//   an assumed ordering.
//
// Arguments:
//   fdi : the OpenCLDeviceInfo object to be compared against this instance.
//
// Returns: true - if this device precedes the specified device
//          false - otherwise
//
// Note:
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
bool
OpenCLDeviceInfo::operator< (const OpenCLDeviceInfo& fdi) const
{
#define LESS_VAR(x) do{\
                        if (x < fdi.x) return (true); \
                        if (x > fdi.x) return (false); \
                    } while(0)

    int i;
    LESS_VAR (numDimensions);
    LESS_VAR (numIntValues);
    LESS_VAR (numSizeTValues);
    LESS_VAR (numLongValues);
    LESS_VAR (numBoolValues);
    LESS_VAR (numStringValues);
    for (i=0 ; i<numDimensions ; ++i) {
        LESS_VAR (maxWorkSizes[i]);
    }
    for (i=0 ; i<numIntValues ; ++i) {
        LESS_VAR (intValues[i]);
    }
    for (i=0 ; i<numSizeTValues ; ++i) {
        LESS_VAR (sizeTValues[i]);
    }
    for (i=0 ; i<numLongValues ; ++i) {
        LESS_VAR (longValues[i]);
    }
    for (i=0 ; i<numBoolValues ; ++i) {
        LESS_VAR (boolValues[i]);
    }
    LESS_VAR (hasHalfFp);
    LESS_VAR (hasDoubleFp);

    for (i=0 ; i<numStringValues ; ++i) {
        LESS_VAR (stringValues[i]);
    }

    LESS_VAR (typeValue);
    LESS_VAR (addrBitsValue);
    LESS_VAR (cacheTypeValue);
    LESS_VAR (localMemValue);
    LESS_VAR (execCapabilitiesValue);
    LESS_VAR (queuePropertiesValue);
    LESS_VAR (halfFpValue);
    LESS_VAR (singleFpValue);
    LESS_VAR (doubleFpValue);

    return (false);
}

// ****************************************************************************
// Method: OpenCLDeviceInfo::operator>
//
// Purpose:
//   Greater operator: compares two OpenCLDeviceInfo objects based on
//   an assumed ordering.
//
// Arguments:
//   fdi : the OpenCLDeviceInfo object to be compared against this instance.
//
// Returns: true - if this device succeedes the specified device
//          false - otherwise
//
// Note:
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
bool
OpenCLDeviceInfo::operator> (const OpenCLDeviceInfo& fdi) const
{
#define GREATER_VAR(x) do{\
                           if (x > fdi.x) return (true); \
                           if (x < fdi.x) return (false); \
                       } while(0)

    int i;
    GREATER_VAR (numDimensions);
    GREATER_VAR (numIntValues);
    GREATER_VAR (numSizeTValues);
    GREATER_VAR (numLongValues);
    GREATER_VAR (numBoolValues);
    GREATER_VAR (numStringValues);
    for (i=0 ; i<numDimensions ; ++i) {
        GREATER_VAR (maxWorkSizes[i]);
    }
    for (i=0 ; i<numIntValues ; ++i) {
        GREATER_VAR (intValues[i]);
    }
    for (i=0 ; i<numSizeTValues ; ++i) {
        GREATER_VAR (sizeTValues[i]);
    }
    for (i=0 ; i<numLongValues ; ++i) {
        GREATER_VAR (longValues[i]);
    }
    for (i=0 ; i<numBoolValues ; ++i) {
        GREATER_VAR (boolValues[i]);
    }
    GREATER_VAR (hasHalfFp);
    GREATER_VAR (hasDoubleFp);

    for (i=0 ; i<numStringValues ; ++i) {
        GREATER_VAR (stringValues[i]);
    }

    GREATER_VAR (typeValue);
    GREATER_VAR (addrBitsValue);
    GREATER_VAR (cacheTypeValue);
    GREATER_VAR (localMemValue);
    GREATER_VAR (execCapabilitiesValue);
    GREATER_VAR (queuePropertiesValue);
    GREATER_VAR (halfFpValue);
    GREATER_VAR (singleFpValue);
    GREATER_VAR (doubleFpValue);

    return (false);
}

// ****************************************************************************
// Method: OpenCLDeviceInfo::operator==
//
// Purpose:
//   Equality operator: compares two OpenCLDeviceInfo objects based on
//   an assumed ordering.
//
// Arguments:
//   fdi : the OpenCLDeviceInfo object to be compared against this instance.
//
// Returns: true - if this device is equal to the specified device
//          false - otherwise
//
// Note:
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
bool
OpenCLDeviceInfo::operator== (const OpenCLDeviceInfo& fdi) const
{
#define EQUAL_VAR(name) if (name != fdi.name) return (false)

    int i;
    EQUAL_VAR (numDimensions);
    EQUAL_VAR (numIntValues);
    EQUAL_VAR (numSizeTValues);
    EQUAL_VAR (numLongValues);
    EQUAL_VAR (numBoolValues);
    EQUAL_VAR (numStringValues);
    for (i=0 ; i<numDimensions ; ++i) {
        EQUAL_VAR (maxWorkSizes[i]);
    }
    for (i=0 ; i<numIntValues ; ++i) {
        EQUAL_VAR (intValues[i]);
    }
    for (i=0 ; i<numSizeTValues ; ++i) {
        EQUAL_VAR (sizeTValues[i]);
    }
    for (i=0 ; i<numLongValues ; ++i) {
        EQUAL_VAR (longValues[i]);
    }
    for (i=0 ; i<numBoolValues ; ++i) {
        EQUAL_VAR (boolValues[i]);
    }
    EQUAL_VAR (hasHalfFp);
    EQUAL_VAR (hasDoubleFp);

    for (i=0 ; i<numStringValues ; ++i) {
        EQUAL_VAR (stringValues[i]);
    }

    EQUAL_VAR (typeValue);
    EQUAL_VAR (addrBitsValue);
    EQUAL_VAR (cacheTypeValue);
    EQUAL_VAR (localMemValue);
    EQUAL_VAR (execCapabilitiesValue);
    EQUAL_VAR (queuePropertiesValue);
    EQUAL_VAR (halfFpValue);
    EQUAL_VAR (singleFpValue);
    EQUAL_VAR (doubleFpValue);

    return (true);
}

// ****************************************************************************
// Method: OpenCLDeviceInfo::~OpenCLDeviceInfo
//
// Purpose:
//   Destructor: frees memory occupied by this device object
//
// Arguments:
//
// Returns:
//
// Note:
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
OpenCLDeviceInfo::~OpenCLDeviceInfo ()
{
    delete[] intValues;
    delete[] sizeTValues;
    delete[] longValues;
    delete[] boolValues;
    delete[] stringValues;

    if (maxWorkSizes)
       delete[] maxWorkSizes;
}

// ****************************************************************************
// Method: OpenCLDeviceInfo::FillDeviceInformation
//
// Purpose:
//   Queries the OpenCL run-time system and initializes all information
//   for this device object.
//
// Arguments:
//
// Returns:
//
// Note:
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
void
OpenCLDeviceInfo::FillDeviceInformation ()
{
    char buf[4096];
    int i;
    size_t len;

    // read all properties in order. Start with the String properties
    for (i=0 ; i<numStringValues ; ++i)
    {
        // I do not really need len if I use strdup
        clGetDeviceInfo (id, clStringCharacteristics[i], 4096, buf, &len);
        stringValues[i] = buf;
    }

    for (i=0 ; i<numIntValues ; ++i)
    {
        clGetDeviceInfo (id, clIntCharacteristics[i],
                         sizeof(cl_uint), &intValues[i], NULL);
        if (clIntCharacteristics[i] == CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)
            numDimensions = intValues[i];
    }

    for (i=0 ; i<numSizeTValues ; ++i)
    {
        clGetDeviceInfo (id, clSizeTCharacteristics[i],
                         sizeof(size_t), &sizeTValues[i], NULL);
    }

    for (i=0 ; i<numLongValues ; ++i)
    {
        clGetDeviceInfo (id, clLongCharacteristics[i],
                         sizeof(cl_long), &longValues[i], NULL);
    }

    for (i=0 ; i<numBoolValues ; ++i)
    {
        clGetDeviceInfo (id, clBoolCharacteristics[i],
                         sizeof(cl_bool), &boolValues[i], NULL);
    }

    maxWorkSizes = new size_t[numDimensions];
    clGetDeviceInfo (id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                     sizeof(size_t)*numDimensions, maxWorkSizes, NULL);

    // temporary bitfield values; final values are stored in strings
    cl_device_type type;
    cl_bitfield addressBits;
    cl_device_mem_cache_type memCacheType;
    cl_device_local_mem_type localMemType;
    cl_device_exec_capabilities execCapabilities;
    cl_command_queue_properties queueProperties;

    cl_device_fp_config halfFpConfig;
    cl_device_fp_config singleFpConfig;
    cl_device_fp_config doubleFpConfig;

    clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    typeValue = string("");
    for (i=0 ; typePropertiesList[i]!=END_ENUMERATION_MARKER ; ++i)
        if (typePropertiesList[i] & type)
            typeValue = typeValue + typePropertiesNames[i] + " ";

    clGetDeviceInfo (id, CL_DEVICE_ADDRESS_BITS, sizeof(cl_bitfield),
                  &addressBits, NULL);
    addrBitsValue = string("");
    for (i=0 ; addressBitsPropertiesList[i]!=END_ENUMERATION_MARKER ; ++i)
        if (addressBitsPropertiesList[i] & addressBits)
            addrBitsValue = addrBitsValue + addressBitsPropertiesNames[i] + " ";

    clGetDeviceInfo (id, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                  sizeof(cl_device_mem_cache_type), &memCacheType, NULL);
    cacheTypeValue = string("");
    for (i=0 ; cacheTypePropertiesList[i]!=END_ENUMERATION_MARKER ; ++i)
        if (cacheTypePropertiesList[i] & memCacheType)
            cacheTypeValue = cacheTypeValue + cacheTypePropertiesNames[i] + " ";

    clGetDeviceInfo (id, CL_DEVICE_LOCAL_MEM_TYPE,
                  sizeof(cl_device_local_mem_type), &localMemType, NULL);
    localMemValue = string("");
    for (i=0 ; localMemTypePropertiesList[i]!=END_ENUMERATION_MARKER ; ++i)
        if (localMemTypePropertiesList[i] & localMemType)
            localMemValue = localMemValue + localMemTypePropertiesNames[i] + " ";

    clGetDeviceInfo (id, CL_DEVICE_EXECUTION_CAPABILITIES,
                  sizeof(cl_device_exec_capabilities), &execCapabilities, NULL);
    execCapabilitiesValue = string("");
    for (i=0 ; execPropertiesList[i]!=END_ENUMERATION_MARKER ; ++i)
        if (execPropertiesList[i] & execCapabilities)
            execCapabilitiesValue = execCapabilitiesValue + execPropertiesNames[i] + " ";

    clGetDeviceInfo (id, CL_DEVICE_QUEUE_PROPERTIES,
                  sizeof(cl_command_queue_properties), &queueProperties, NULL);
    queuePropertiesValue = string("");
    for (i=0 ; queuePropertiesList[i]!=END_ENUMERATION_MARKER ; ++i)
        if (queuePropertiesList[i] & queueProperties)
            queuePropertiesValue = queuePropertiesValue + queuePropertiesNames[i] + " ";

    halfFpValue = string("");
#ifdef CL_DEVICE_HALF_FP_CONFIG
    clGetDeviceInfo (id, CL_DEVICE_HALF_FP_CONFIG,
                  sizeof(cl_device_fp_config), &halfFpConfig, NULL);
    for (i=0 ; fpPropertiesList[i]!=END_ENUMERATION_MARKER ; ++i)
        if (fpPropertiesList[i] & halfFpConfig)
            halfFpValue = halfFpValue + fpPropertiesNames[i] + " ";
#endif
    clGetDeviceInfo (id, CL_DEVICE_SINGLE_FP_CONFIG,
                  sizeof(cl_device_fp_config), &singleFpConfig, NULL);
    singleFpValue = string("");
    for (i=0 ; fpPropertiesList[i]!=END_ENUMERATION_MARKER ; ++i)
        if (fpPropertiesList[i] & singleFpConfig)
            singleFpValue = singleFpValue + fpPropertiesNames[i] + " ";

    doubleFpValue = string("");
    if (hasDoubleFp
#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
         || true
#endif
       )
    {
       clGetDeviceInfo (id, CL_DEVICE_DOUBLE_FP_CONFIG,
                     sizeof(cl_device_fp_config), &doubleFpConfig, NULL);
       for (i=0 ; fpPropertiesList[i]!=END_ENUMERATION_MARKER ; ++i)
           if (fpPropertiesList[i] & doubleFpConfig)
               doubleFpValue = doubleFpValue + fpPropertiesNames[i] + " ";
    }
}

// ****************************************************************************
// Method: OpenCLDeviceInfo::Print
//
// Purpose:
//   Pretty prints the attributes of this device
//
// Arguments:
//   os: stream used for writing
//
// Returns:
//
// Note:
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
void
OpenCLDeviceInfo::Print(ostream &os) const
{
    int i;
    os << "--> Device id=" << id << " <--" << endl;

    for (i=0 ; i<numStringValues ; ++i)
    {
        os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
           << clStringCharacteristicsNames[i] << " = "
           << stringValues[i] << endl;
    }
    os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
       << "DeviceType" << " = " << typeValue << endl;

    for (i=0 ; i<numIntValues ; ++i)
    {
        os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
           << clIntCharacteristicsNames[i] << " = "
           << intValues[i] << endl;
    }

    for (i=0 ; i<numSizeTValues ; ++i)
    {
        os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
           << clSizeTCharacteristicsNames[i] << " = "
           << sizeTValues[i] << endl;
    }

    // print the maximum Work Sizes on each dimension
    os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
       << "MaxWorkItemSizes" << " = (" << maxWorkSizes[0];
    for (i=1 ; i<numDimensions ; ++i)
        os << "," << maxWorkSizes[i];
    os << ")" << endl;

    for (i=0 ; i<numLongValues ; ++i)
    {
        os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
           << clLongCharacteristicsNames[i] << " = "
           << HumanReadable(longValues[i], 0) << "B" << endl;
    }

    for (i=0 ; i<numBoolValues ; ++i)
    {
        os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
           << clBoolCharacteristicsNames[i] << " = "
           << (boolValues[i]?"yes":"no") << endl;
    }

    os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
       << "SupportedAddressSpaces" << " = " << addrBitsValue << endl;

    os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
       << "GlobalMemoryCacheType" << " = " << cacheTypeValue << endl;

    os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
       << "TypeLocalMemory" << " = " << localMemValue << endl;

    os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
       << "ExecutionCapabilities" << " = " << execCapabilitiesValue << endl;

    os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
       << "SupportedQueueProperties" << " = " << queuePropertiesValue << endl;

    os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
       << "HalfFpSupport" << " = " << (hasHalfFp?"yes":"no") << endl;
    if (hasHalfFp)
        os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
           << "HalfFpCapabilities" << " = " << halfFpValue << endl;

    os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
       << "SingleFpSupport" << " = " << "yes" << endl;
    os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
       << "SingleFpCapabilities" << " = " << singleFpValue << endl;

    os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
       << "DoubleFpSupport" << " = " << (hasDoubleFp?"yes":"no") << endl;
    if (hasDoubleFp)
        os << "   " << setiosflags(ios::left) << setw(maxKeyLength)
           << "DoubleFpCapabilities" << " = " << doubleFpValue << endl;
    os << endl;
}

// ****************************************************************************
// Method: OpenCLDeviceInfo::writeObject
//
// Purpose:
//   Implements the serialization method of the SerializeObject interface.
//
// Arguments:
//   oss: output string stream object used for writing the serialized
//        representation of the object
//
// Returns:
//
// Note:
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
void
OpenCLDeviceInfo::writeObject(ostringstream &oss) const
{
    int i;
    // should I send the number of int values, long values, etc. ?
    // If they are different from one node to another, then I should send the
    // name of the property as well. Hmmm ...
    // Although these are standard OpenCL parameters. They might change, but
    // then the program will change on all nodes. I do not need to send them.
    oss << " " << MAGIC_KEY_DEVICE_INFO
//        << " " << id
        << " " << numDimensions
        << " " << hasHalfFp
        << " " << hasDoubleFp;

    for (i=0 ; i<numIntValues ; ++i)
    {
        oss << " " << intValues[i];
    }

    for (i=0 ; i<numSizeTValues ; ++i)
    {
        oss << " " << sizeTValues[i];
    }

    // serialize the maximum Work Sizes on each dimension
    for (i=0 ; i<numDimensions ; ++i)
        oss << " " << maxWorkSizes[i];

    for (i=0 ; i<numLongValues ; ++i)
    {
        oss << " " << longValues[i];
    }

    for (i=0 ; i<numBoolValues ; ++i)
    {
        oss << " " << boolValues[i];
    }
    oss << "\n";

    for (i=0 ; i<numStringValues ; ++i)
    {
        oss << stringValues[i] << "\n";
    }
    oss << typeValue << "\n";
    oss << addrBitsValue << "\n";
    oss << cacheTypeValue << "\n";
    oss << localMemValue << "\n";
    oss << execCapabilitiesValue << "\n";
    oss << queuePropertiesValue << "\n";

    if (hasHalfFp)
        oss << halfFpValue << "\n";
    oss << singleFpValue << "\n";
    if (hasDoubleFp)
        oss << doubleFpValue << "\n";
}

// ****************************************************************************
// Method: OpenCLDeviceInfo::readObject
//
// Purpose:
//   Implements the unserialization method of the SerializeObject interface.
//
// Arguments:
//   iss: input string stream object used for reading the serialized
//        representation of the object
//
// Returns:
//
// Note:  It overwrittes the current device information with the data read
//    from the input stream.
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
void
OpenCLDeviceInfo::readObject(istringstream &iss)
{
    int i;
    int receivedKey = 0;
    int oldNumDimensions = numDimensions;

    iss >> receivedKey;
    if (receivedKey != MAGIC_KEY_DEVICE_INFO)  // wrong magic key
    {
        cerr << "Wrong magic key received " << receivedKey
             << " while unserializing an OpenCLDeviceInfo object." << endl;
        exit (-2);
    }
    // should I send the number of int values, long values, etc. ?
    // If they are different from one node to another, then I should send the
    // name of the property as well. Hmmm ...
    // Although these are standard OpenCL parameters. They might change, but
    // then the program will change on all nodes. I do not need to send them.
//    iss >> (long)id
    iss >> numDimensions
        >> hasHalfFp
        >> hasDoubleFp;

    for (i=0 ; i<numIntValues ; ++i)
    {
        iss >> intValues[i];
    }

    for (i=0 ; i<numSizeTValues ; ++i)
    {
        iss >> sizeTValues[i];
    }

    // serialize the maximum Work Sizes on each dimension
    // do I need to reallocate memory for maxWorkSizes?
    if (maxWorkSizes && numDimensions!=oldNumDimensions)
    {
        delete[] maxWorkSizes;
        maxWorkSizes = 0;
    }
    if (!maxWorkSizes)
        maxWorkSizes = new size_t[numDimensions];

    for (i=0 ; i<numDimensions ; ++i)
        iss >> maxWorkSizes[i];

    for (i=0 ; i<numLongValues ; ++i)
    {
        iss >> longValues[i];
    }

    for (i=0 ; i<numBoolValues ; ++i)
    {
        iss >> boolValues[i];
    }

    string dummy;
    getline (iss, dummy);  // read the newline before the first string value

    // strings are one per line, \n is the separator
    for (i=0 ; i<numStringValues ; ++i)
    {
        getline (iss, stringValues[i]);
    }
    getline (iss, typeValue);
    getline (iss, addrBitsValue);
    getline (iss, cacheTypeValue);
    getline (iss, localMemValue);
    getline (iss, execCapabilitiesValue);
    getline (iss, queuePropertiesValue);

    if (hasHalfFp)
        getline (iss, halfFpValue);
    getline (iss, singleFpValue);
    if (hasDoubleFp)
        getline (iss, doubleFpValue);
}

// ****************************************************************************
// Method: ListDevicesAndGetDevice
//
// Purpose:
//   Get the OpenCL device ID for the device with the specified index on
//   the specified platform.
//
// Arguments:
//   platform: platform index
//   device:   device index
//
// Returns: the OpenCL device ID if indeces are valid
//
// Note:  Function exits the program if specified platform or device
//    are not found.
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//   Jeremy Meredith, Mon Nov 16 11:48:22 EST 2009
//   Improved output to include device name and index.
//
// ****************************************************************************
cl_device_id
ListDevicesAndGetDevice(int platformIdx, int deviceIdx, bool output)
{
    cl_int err;

    // TODO remove duplication between this function and GetNumOclDevices.
    cl_uint nPlatforms = 0;
    err = clGetPlatformIDs(0, NULL, &nPlatforms);
    CL_CHECK_ERROR(err);

    if (nPlatforms <= 0)
    {
        cerr << "No OpenCL platforms found. Exiting." << endl;
        exit(0);
    }
    if (platformIdx<0 || platformIdx>=nPlatforms)  // platform ID out of range
    {
        cerr << "Platform index " << platformIdx << " is out of range. "
             << "Specify a platform index between 0 and "
             << nPlatforms-1 << endl;
        exit(-4);
    }

    cl_platform_id* platformIDs = new cl_platform_id[nPlatforms];
    err = clGetPlatformIDs(nPlatforms, platformIDs, NULL);
    CL_CHECK_ERROR(err);

    // query devices
    cl_uint nDevices = 0;
    err = clGetDeviceIDs(platformIDs[platformIdx],
                        CL_DEVICE_TYPE_ALL,
                        0,
                        NULL,
                        &nDevices );
    CL_CHECK_ERROR(err);
    cl_device_id* devIDs = new cl_device_id[nDevices];
    err = clGetDeviceIDs(platformIDs[platformIdx],
                        CL_DEVICE_TYPE_ALL,
                        nDevices,
                        devIDs,
                        NULL );
    CL_CHECK_ERROR(err);

    if (nDevices <= 0)
    {
        cerr << "No OpenCL devices found. Exiting." << endl;
        exit(0);
    }
    if (deviceIdx<0 || deviceIdx>=nDevices)  // platform ID out of range
    {
        cerr << "Device index " << deviceIdx << " is out of range. "
             << "Specify a device index between 0 and " << nDevices-1
             << endl;
        exit(-5);
    }

    cl_device_id retval = devIDs[deviceIdx];
    if( output )
    {
        size_t nBytesNeeded = 0;
        err = clGetDeviceInfo( retval,
                                CL_DEVICE_NAME,
                                0,
                                NULL,
                                &nBytesNeeded );
        CL_CHECK_ERROR(err);
        char* devName = new char[nBytesNeeded+1];
        err = clGetDeviceInfo( retval,
                                CL_DEVICE_NAME,
                                nBytesNeeded+1,
                                devName,
                                NULL );
        
        cout << "Chose device:"
             << " name='"<< devName <<"'"
             << " index="<<deviceIdx
             << " id="<<retval
             << endl;

        delete[] devName;
    }

    delete[] platformIDs;
    delete[] devIDs;

    return retval;
}

// ****************************************************************************
// Method: GetNumOclDevices
//
// Purpose:
//   Gets the number of available OpenCL devices in the specified
//   platform
//
// Arguments:
//   platform: platform index
//
// Returns: the number of ocl devices
//
//
// Programmer: Kyle Spafford
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
int
GetNumOclDevices(int platformIndex)
{
    cl_int err;

    
    cl_uint nPlatforms = 0;
    err = clGetPlatformIDs(0, NULL, &nPlatforms);   // determine number of platforms available
    CL_CHECK_ERROR(err);

    if (nPlatforms <= 0)
    {
        cerr << "No OpenCL platforms found. Exiting." << endl;
        exit(-1);
    }
    if (platformIndex<0 || platformIndex>=nPlatforms)  // platform index out of range
    {

        cerr << "Platform index " << platformIndex << " is out of range. "
             << "Specify a platform index between 0 and "
             << nPlatforms-1 << endl;
        exit(-4);
    }

    cl_platform_id* platformIDs = new cl_platform_id[nPlatforms];
    err = clGetPlatformIDs(nPlatforms, platformIDs, NULL);
    CL_CHECK_ERROR(err);

    // query devices for the indicated platform
    cl_uint nDevices = 0;
    err = clGetDeviceIDs( platformIDs[platformIndex],
                            CL_DEVICE_TYPE_ALL,
                            0,
                            NULL,
                            &nDevices );
    CL_CHECK_ERROR(err);
    cl_device_id* devIDs = new cl_device_id[nDevices];
    err = clGetDeviceIDs( platformIDs[platformIndex],
                            CL_DEVICE_TYPE_ALL,
                            nDevices,
                            devIDs,
                            NULL );
    CL_CHECK_ERROR(err);

    delete[] platformIDs;
    delete[] devIDs;

    return (int)nDevices;
}

