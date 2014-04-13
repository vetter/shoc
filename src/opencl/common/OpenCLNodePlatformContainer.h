#ifndef OPENCL_NODE_PLATFORM_CONTAINER_H
#define OPENCL_NODE_PLATFORM_CONTAINER_H

#include <iostream>
#include <string>
#include <list>
#include "support.h"
#include "OpenCLPlatform.h"
#include "NodePlatformContainer.h"

using namespace std;

// ****************************************************************************
// Class: OpenCLNodePlatformContainer
//
// Purpose:
//   A container for all OpenCL platforms on a node.
//
// Notes:     Extends the generic node platform container class
//
// Programmer: Gabriel Marin
// Creation: September 22, 2009
//
// Modifications:
//
// ****************************************************************************
namespace SHOC {

    class OpenCLNodePlatformContainer : public NodePlatformContainer<OpenCLPlatform>
    {
    private:
        static const int MAGIC_KEY_OPENCL_NODE_CONTAINER;

    public:
        // constructor collects information about all platforms on this node
        OpenCLNodePlatformContainer (bool do_initialize = true);
        OpenCLNodePlatformContainer (const OpenCLNodePlatformContainer &ondc);
        OpenCLNodePlatformContainer& operator= (const OpenCLNodePlatformContainer &ondc);

        ~OpenCLNodePlatformContainer () { }

        void Print (ostream &os) const;

        void initialize();

        virtual void writeObject (ostringstream &oss) const;
        virtual void readObject (istringstream &iss);

        bool operator< (const OpenCLNodePlatformContainer &ndc) const;
        bool operator> (const OpenCLNodePlatformContainer &ndc) const;
        bool operator== (const OpenCLNodePlatformContainer &ndc) const;
    };
};


#endif
