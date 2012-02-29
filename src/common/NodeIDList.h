
#ifndef _NODE_ID_LIST_H
#define _NODE_ID_LIST_H

#include <list>
#include <string>

// ****************************************************************************
// Type: NodeIDList
//
// Purpose:
//   Defines a list of strings for holding host name information
//   associated with a particular system configuration.
//
// Programmer: Gabriel Marin
// Creation: August 25, 2009
//
// Modifications:
//
// ****************************************************************************

namespace SHOC {
    const int MAGIC_KEY_NODE_ID_LIST = 0x1071badc;
    typedef std::list <std::string> NodeIDList;
};

#endif
