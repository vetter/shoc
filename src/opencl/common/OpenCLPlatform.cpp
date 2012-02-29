#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "OpenCLPlatform.h"
#include "support.h"
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <assert.h>

using namespace std;
using namespace SHOC;

const int SHOC::OpenCLPlatform::MAGIC_KEY_OPENCL_PLATFORM = 0x5a9ef20c;

// ****************************************************************************
// Method: OpenCLPlatform::OpenCLPlatform
//
// Purpose: 
//   Constructor. Creates a new empty OpenCL platform.
//
// Arguments:
//
// Returns: 
//
// Note: 
//
// Programmer: Gabriel Marin
// Creation: September 22, 2009
//
// Modifications:
//
// ****************************************************************************
OpenCLPlatform::OpenCLPlatform () : Platform<OpenCLDeviceInfo>()
{
}

// ****************************************************************************
// Method: OpenCLPlatform::OpenCLPlatform
//
// Purpose: 
//   Constructor. Creates a new OpenCL platform representation for the 
//   specified patform ID.
//
// Arguments:
//   clPlatform: the platform to be created
//
// Returns: 
//
// Note: 
//
// Programmer: Gabriel Marin
// Creation: September 22, 2009
//
// Modifications:
//
// ****************************************************************************
OpenCLPlatform::OpenCLPlatform (cl::Platform &clPlatform) : 
           Platform<OpenCLDeviceInfo>()
{
    int err;
    platformName = clPlatform.getInfo<CL_PLATFORM_NAME>();
    platformVendor = clPlatform.getInfo<CL_PLATFORM_VENDOR>();
    platformVersion = clPlatform.getInfo<CL_PLATFORM_VERSION>();
    platformExtensions = clPlatform.getInfo<CL_PLATFORM_EXTENSIONS>();

    // query devices
    std::vector<cl::Device> devs;
    err = clPlatform.getDevices( CL_DEVICE_TYPE_ALL, &devs );
    // I should not print an error message here if no devices are present.
    // I will just report that there are zero devices.
//    CL_CHECK_ERROR( err );

    for( vector<cl::Device>::iterator diter = devs.begin();
        diter != devs.end();
        diter++ )
    {
        OpenCLDeviceInfo* openclDevice = new OpenCLDeviceInfo( (*diter)() );
        devices.push_back( openclDevice );
        ++ deviceCount;
    }
}


// ****************************************************************************
// Method: OpenCLPlatform::OpenCLPlatform
//
// Purpose: 
//   Copy constructor. Creates a new OpenCL platform which is an exact
//   copy of the specified platform.
//
// Arguments:
//   ocp: the platform to be duplicated
//
// Returns: 
//
// Note: 
//
// Programmer: Gabriel Marin
// Creation: September 22, 2009
//
// Modifications:
//
// ****************************************************************************
OpenCLPlatform::OpenCLPlatform (const OpenCLPlatform &ocp) :
             Platform<OpenCLDeviceInfo> (ocp)
{
    platformName = ocp.platformName;
    platformVendor = ocp.platformVendor;
    platformVersion = ocp.platformVersion;
    platformExtensions = ocp.platformExtensions;
}

// ****************************************************************************
// Method: OpenCLPlatform::operator=
//
// Purpose: 
//   Copy operator. Copies the content of the specified platform into
//   this platform instance.
//
// Arguments:
//   ocp: the platform to be duplicated
//
// Returns:  a reference to this platform object
//
// Note: 
//
// Programmer: Gabriel Marin
// Creation: September 22, 2009
//
// Modifications:
//
// ****************************************************************************
OpenCLPlatform& 
OpenCLPlatform::operator= (const OpenCLPlatform &ocp)
{
    this->Platform<OpenCLDeviceInfo>::operator= (ocp);
    platformName = ocp.platformName;
    platformVendor = ocp.platformVendor;
    platformVersion = ocp.platformVersion;
    platformExtensions = ocp.platformExtensions;
    return (*this);
}


// ****************************************************************************
// Method: OpenCLPlatform::Print
//
// Purpose: 
//   Pretty print the content of this platform.
//
// Arguments:
//   os: the output stream where the writing is done
//
// Returns: 
//
// Note: 
//
// Programmer: Gabriel Marin
// Creation: September 22, 2009
//
// Modifications:
//
// ****************************************************************************
void
OpenCLPlatform::Print (ostream &os) const
{
    os << "PlatformName = '" << platformName << "'" << endl;
    os << "Vendor = '" << platformVendor << "'" << endl;
    os << "Version = '" << platformVersion << "'" << endl;
    os << "Extensions = '" << platformExtensions << "'" << endl;
    Platform<OpenCLDeviceInfo>::Print (os);
}

// ****************************************************************************
// Method: OpenCLPlatform::writeObject
//
// Purpose: 
//   Implements the serialization method of the SerializableObject
//   abstract class
//
// Arguments:
//   oss: the output string stream where the serialized representation 
//        is written
//
// Returns: 
//
// Note: 
//
// Programmer: Gabriel Marin
// Creation: September 22, 2009
//
// Modifications:
//
// ****************************************************************************
void 
OpenCLPlatform::writeObject(ostringstream &oss) const
{
    oss << " " << MAGIC_KEY_OPENCL_PLATFORM << "\n";
    oss << platformName << "\n"
        << platformVendor << "\n"
        << platformVersion << "\n"
        << platformExtensions << "\n";
    Platform<OpenCLDeviceInfo>::writeObject (oss);
}

// ****************************************************************************
// Method: OpenCLPlatform::readObject
//
// Purpose: 
//   Implements the unserialization method of the SerializableObject
//   abstract class
//
// Arguments:
//   iss: the input string stream from where the serialized representation 
//        is read
//
// Returns: 
//
// Note: 
//
// Programmer: Gabriel Marin
// Creation: September 22, 2009
//
// Modifications:
//
// ****************************************************************************
void 
OpenCLPlatform::readObject(istringstream &iss)
{
    int receivedKey = 0;
    
    iss >> receivedKey;
    if (receivedKey != MAGIC_KEY_OPENCL_PLATFORM)  // wrong magic key
    {
        cerr << "Wrong magic key received " << receivedKey 
             << " while unserializing an OpenCLPlatform object." << endl;
        exit (-2);
    }
    
    string dummy;
    getline (iss, dummy);  // read the newline before the first string value
    getline (iss, platformName);
    getline (iss, platformVendor);
    getline (iss, platformVersion);
    getline (iss, platformExtensions);
    Platform<OpenCLDeviceInfo>::readObject (iss);
}

// ****************************************************************************
// Method: OpenCLPlatform::operator<
//
// Purpose: 
//   Less operator: compares two OpenCL platform objects based on
//   an assumed ordering.
//
// Arguments:
//   ocp: the OpenCL platform to be compared against this instance.
//
// Returns: true - if this platform precedes the specified platform
//          false - otherwise
//
// Note: 
//
// Programmer: Gabriel Marin
// Creation: September 22, 2009
//
// Modifications:
//
// ****************************************************************************
bool 
OpenCLPlatform::operator< (const OpenCLPlatform &ocp) const
{
    if (platformName < ocp.platformName)
        return (true);
    if (platformName > ocp.platformName)
        return (false);
    if (platformVendor < ocp.platformVendor)
        return (true);
    if (platformVendor > ocp.platformVendor)
        return (false);
    if (platformVersion < ocp.platformVersion)
        return (true);
    if (platformVersion > ocp.platformVersion)
        return (false);
    if (platformExtensions < ocp.platformExtensions)
        return (true);
    if (platformExtensions > ocp.platformExtensions)
        return (false);
    return (Platform<OpenCLDeviceInfo>::operator< (ocp));
}

// ****************************************************************************
// Method: OpenCLPlatform::operator>
//
// Purpose: 
//   Greater operator: compares two OpenCL platform objects based on
//   an assumed ordering.
//
// Arguments:
//   ocp: the OpenCL platform to be compared against this instance.
//
// Returns: true - if this platform succeeds the specified platform
//          false - otherwise
//
// Note: 
//
// Programmer: Gabriel Marin
// Creation: September 22, 2009
//
// Modifications:
//
// ****************************************************************************
bool 
OpenCLPlatform::operator> (const OpenCLPlatform &ocp) const
{
    if (platformName > ocp.platformName)
        return (true);
    if (platformName < ocp.platformName)
        return (false);
    if (platformVendor > ocp.platformVendor)
        return (true);
    if (platformVendor < ocp.platformVendor)
        return (false);
    if (platformVersion > ocp.platformVersion)
        return (true);
    if (platformVersion < ocp.platformVersion)
        return (false);
    if (platformExtensions > ocp.platformExtensions)
        return (true);
    if (platformExtensions < ocp.platformExtensions)
        return (false);
    return (Platform<OpenCLDeviceInfo>::operator> (ocp));
}

// ****************************************************************************
// Method: OpenCLPlatform::operator==
//
// Purpose: 
//   Equality operator: compares two OpenCL platform objects based on
//   an assumed ordering.
//
// Arguments:
//   ocp: the OpenCL platform to be compared against this instance.
//
// Returns: true - if this platform is equal to the specified platform
//          false - otherwise
//
// Note: 
//
// Programmer: Gabriel Marin
// Creation: September 22, 2009
//
// Modifications:
//
// ****************************************************************************
bool 
OpenCLPlatform::operator== (const OpenCLPlatform &ocp) const
{
    if (platformName != ocp.platformName)
        return (false);
    if (platformVendor != ocp.platformVendor)
        return (false);
    if (platformVersion != ocp.platformVersion)
        return (false);
    if (platformExtensions != ocp.platformExtensions)
        return (false);
    return (Platform<OpenCLDeviceInfo>::operator== (ocp));
}
