
#ifndef _SERIALIZABLE_OBJECT_H
#define _SERIALIZABLE_OBJECT_H

#include <iostream>
#include <sstream>

// ****************************************************************************
// Class: SerializableObject
//
// Purpose:
//   Abstract class with two pure virtual methods for serializing and
//   unserializing an object to string.
//
// Notes:
//   All Devices, Platforms, Node Containers and Multi-Node Containers
//   that are sent over the network must implement this interface.
//
// Programmer: Gabriel Marin
// Creation: August 21, 2009
//
// Modifications:
//
// ****************************************************************************
class SerializableObject
{
public:
    SerializableObject() {}
    virtual void writeObject (std::ostringstream &oss) const = 0;
    virtual void readObject (std::istringstream &iss) = 0;
};

#endif
