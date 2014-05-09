#ifndef SHOC_PLATFORM_H
#define SHOC_PLATFORM_H

#include <iostream>
#include <string>
#include <list>
#include <map>
#include <stdlib.h>
#include "SerializableObject.h"

using namespace std;

// ****************************************************************************
// Class: Platform
//
// Purpose:
//   Generic Platform container, to be extended by the OpenCL and CUDA
//   specific implementations. A Platform contains zero or more devices
//   of type DeviceType.
//   Platform implements the SerializableObject interface.
//
// Programmer: Gabriel Marin
// Creation:   September 22, 2009
//
// Modifications:
//
// ****************************************************************************

namespace SHOC {

    template <typename DeviceType>
    class Platform : public SerializableObject
    {
    protected:
        typedef std::list<DeviceType*> DeviceList;

        DeviceList devices;

        static const int MAGIC_KEY_PLATFORM;

    public:
        // Base constructor. Devices are instantiated by the derived class
        // that is OpenCL or CUDA specific.
        Platform ()
        { }

        // Destructor.
        ~Platform ()
        {
            // deallocate the list of devices
            typename DeviceList::iterator lit = devices.begin();
            for ( ; lit!=devices.end() ; ++lit)
               delete (*lit);
            devices.clear ();
        }

        // return number of devices for this platform
        int getDeviceCount() const        { return devices.size(); }

        // copy constructor
        Platform (const Platform<DeviceType> &pl)
        {
            typename DeviceList::const_iterator lit = pl.devices.begin();
            for ( ; lit!=pl.devices.end() ; ++lit)
               devices.push_back (new DeviceType (*(*lit)));
        }

        // assignment operator
        Platform& operator= (const Platform<DeviceType> &pl)
        {
            // first clear any devices that we have
            typename DeviceList::const_iterator lit = devices.begin();
            for ( ; lit!=devices.end() ; ++lit)
               delete (*lit);
            devices.clear ();

            // now copy the devices from the other container
            for (lit = pl.devices.begin() ; lit!=pl.devices.end() ; ++lit)
               devices.push_back (new DeviceType (*(*lit)));
            return (*this);
        }

        // pretty print a platform and its devices.
        void Print (ostream &os) const
        {
            os << "Number of devices = " << this->getDeviceCount() << endl;

            typename DeviceList::const_iterator lit = devices.begin();
            for ( ; lit!=devices.end() ; ++lit)
               (*lit)->Print (os);
        }

        // implements the serialization method of the SerializableObject
        // abstract class
        void writeObject(ostringstream &oss) const
        {
            oss << " " << MAGIC_KEY_PLATFORM
                << " " << this->getDeviceCount() << "\n";

            typename DeviceList::const_iterator lit = devices.begin();
            for ( ; lit!=devices.end() ; ++lit)
               (*lit)->writeObject (oss);
        }

        // implements the un-serialization method of the SerializableObject
        // abstract class
        void readObject(istringstream &iss)
        {
            int i, receivedKey = 0;

            iss >> receivedKey;
            if (receivedKey != MAGIC_KEY_PLATFORM)  // wrong magic key
            {
                cerr << "Wrong magic key received " << receivedKey
                     << " while unserializing a Platform object." << endl;
                exit (-2);
            }

            unsigned int nDevices = 0;
            iss >> nDevices;

            // deallocate existing devices first
            typename DeviceList::iterator lit = devices.begin();
            for ( ; lit!=devices.end() ; ++lit)
               delete (*lit);
            devices.clear ();

            for (i=0 ; i<nDevices ; ++i)
            {
                DeviceType *dev = new DeviceType();
                dev->readObject (iss);
                devices.push_back (dev);
            }
        }

        // compare two Platforms and return true if this Platform is
        // considered to precede the second Platform.
        bool operator< (const Platform &pl) const
        {
            int i;

            if (this->getDeviceCount() < pl.getDeviceCount())
                return (true);
            if (this->getDeviceCount() > pl.getDeviceCount())
                return (false);

            // test each device in the list next
            typename DeviceList::const_iterator lit1 = devices.begin();
            typename DeviceList::const_iterator lit2 = pl.devices.begin();
            for (i=0 ; i<this->getDeviceCount() ; ++i, ++lit1, ++lit2)
            {
                // better test for equality first because we expect most nodes to have
                // equal configurations. Configuration differences should be the
                // exception, not the rule.
                if (*(*lit1) == *(*lit2)) continue;
                if (*(*lit1) < *(*lit2))
                    return (true);
                else
                    return (false);
            }
            return (false);
        }

        // compare two Platforms and return true if this Platform is
        // considered to succeed the second Platform.
        bool operator> (const Platform &pl) const
        {
            int i;

            if (this->getDeviceCount() > pl.getDeviceCount())
                return (true);
            if (this->getDeviceCount() < pl.getDeviceCount())
                return (false);

            // test each device in the list next
            typename DeviceList::const_iterator lit1 = devices.begin();
            typename DeviceList::const_iterator lit2 = pl.devices.begin();
            for (i=0 ; i<this->getDeviceCount() ; ++i, ++lit1, ++lit2)
            {
                // better test for equality first because we expect most nodes to have
                // equal configurations. Configuration differences should be the
                // exception, not the rule.
                if (*(*lit1) == *(*lit2)) continue;
                if (*(*lit1) > *(*lit2))
                    return (true);
                else
                    return (false);
            }
            return (false);
        }

        // compare two Platforms and return true if the two Platforms
        // are considered equal based on our ordering.
        bool operator== (const Platform &pl) const
        {
            int i;

            if (this->getDeviceCount() != pl.getDeviceCount())
                return (false);

            // test each device in the list next
            typename DeviceList::const_iterator lit1 = devices.begin();
            typename DeviceList::const_iterator lit2 = pl.devices.begin();
            for (i=0 ; i<this->getDeviceCount() ; ++i, ++lit1, ++lit2)
            {
                if (! (*(*lit1) == *(*lit2)))
                    return (false);
            }
            return (true);
        }

    };

    template <typename DeviceType>
    const int Platform<DeviceType>::MAGIC_KEY_PLATFORM = 0x82e6abc3;
};


#endif
