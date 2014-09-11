#ifndef OPENCL_PLATFORM_H
#define OPENCL_PLATFORM_H

#include <iostream>
#include <string>
#include <list>
#include "support.h"
#include "OpenCLDeviceInfo.h"
#include "Platform.h"

using namespace std;

namespace SHOC {

// ****************************************************************************
// Class: OpenCLPlatform
//
// Purpose:
//   Implements an OpenCL platform. A platform contains information about
//   zero or more devices.
//
// Notes:     Extends the generic platform class
//
// Programmer: Gabriel Marin
// Creation: September 22, 2009
//
// Modifications:
//
// ****************************************************************************
    class OpenCLPlatform : public Platform<OpenCLDeviceInfo>
    {
    private:
        string platformName;
        string platformVendor;
        string platformVersion;
        string platformExtensions;
        static const int MAGIC_KEY_OPENCL_PLATFORM;

        static std::string LookupInfo( cl_platform_id platformID, cl_platform_info paramName );

    public:
        // constructer collects information about all devices on this node
        OpenCLPlatform ();
        OpenCLPlatform (cl_platform_id platformID);
        OpenCLPlatform (const OpenCLPlatform &ocp);
        OpenCLPlatform& operator= (const OpenCLPlatform &ocp);

        ~OpenCLPlatform () { }

        void Print (ostream &os) const;

        virtual void writeObject (ostringstream &oss) const;
        virtual void readObject (istringstream &iss);

        bool operator< (const OpenCLPlatform &ocp) const;
        bool operator> (const OpenCLPlatform &ocp) const;
        bool operator== (const OpenCLPlatform &ocp) const;
    };
};


#endif
