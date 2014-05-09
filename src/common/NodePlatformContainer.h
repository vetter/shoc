#ifndef NODE_PLATFORM_CONTAINER_H
#define NODE_PLATFORM_CONTAINER_H

#include <iostream>
#include <string>
#include <list>
#include <map>
#include <stdlib.h>
#include <unistd.h>
#include "SerializableObject.h"

using namespace std;

// ****************************************************************************
// Class: NodePlatformContainer
//
// Purpose:
//   Generic Node platform container, to be extended by the OpenCL and
//   CUDA specific implementations. A node container contains zero or more
//   platforms of type PlatformType.
//   NodePlatformContainer implements the SerializableObject interface.
//
// Programmer: Gabriel Marin
// Creation:   September 22, 2009
//
// Modifications:
//
// ****************************************************************************

namespace SHOC {

    template <typename PlatformType>
    class NodePlatformContainer : public SerializableObject
    {
    protected:
        typedef std::list<PlatformType*> PlatformList;

        string nodeName;
        PlatformList platforms;

        static const int MAGIC_KEY_NODE_CONTAINER;

    public:
        // Base constructer collects information about the current host node.
        // Platforms are instantiated by the OpenCL and CUDA specific
        // implementations.
        NodePlatformContainer ()
        {
            // Node information
            int res;
            char buf[1024];

            res = gethostname (buf, 1024);
            if (res < 0) {
                fprintf (stderr, "gethostname failed\n"); fflush(stderr);
                exit (-1);
            }
            nodeName = buf;
        }


        // Destructor
        ~NodePlatformContainer ()
        {
            // deallocate the list of platforms
            typename PlatformList::iterator lit = platforms.begin();
            for ( ; lit!=platforms.end() ; ++lit)
               delete (*lit);
            platforms.clear ();
        }

        // return the name of the host
        const string& getNodeName() const   { return (nodeName); }

        // return the number of Platforms on this node
        int getPlatformCount() const        { return platforms.size(); }

        // copy constructor
        NodePlatformContainer (const NodePlatformContainer<PlatformType> &ndc)
        {
            nodeName = ndc.nodeName;
            typename PlatformList::const_iterator lit = ndc.platforms.begin();
            for ( ; lit!=ndc.platforms.end() ; ++lit)
               platforms.push_back (new PlatformType (*(*lit)));
        }

        // copy/assignment operator
        NodePlatformContainer& operator= (const NodePlatformContainer<PlatformType> &ndc)
        {
            nodeName = ndc.nodeName;
            // first clear any platforms that we have
            typename PlatformList::const_iterator lit = platforms.begin();
            for ( ; lit!=platforms.end() ; ++lit)
               delete (*lit);
            platforms.clear ();

            // now copy the platforms from the other container
            for (lit = ndc.platforms.begin() ; lit!=ndc.platforms.end() ; ++lit)
               platforms.push_back (new PlatformType (*(*lit)));
            return (*this);
        }

        // pretty print a node container and all its platforms
        void Print (ostream &os) const
        {
            os << "Host name = '" << nodeName << "'" << endl;
            os << "Number of platforms = " << this->getPlatformCount() << endl;

            typename PlatformList::const_iterator lit = platforms.begin();
            for ( ; lit!=platforms.end() ; ++lit)
            {
               os << endl;     // leave a blank line before each platform
               (*lit)->Print (os);
            }
        }

        // implements the serialization method of the SerializableObject
        // abstract class
        void writeObject(ostringstream &oss) const
        {
            oss << " " << MAGIC_KEY_NODE_CONTAINER
                << " " << this->getPlatformCount() << "\n";
            oss << nodeName << "\n";

            typename PlatformList::const_iterator lit = platforms.begin();
            for ( ; lit!=platforms.end() ; ++lit)
               (*lit)->writeObject (oss);
        }

        // implements the un-serialization method of the SerializableObject
        // abstract class
        void readObject(istringstream &iss)
        {
            int i, receivedKey = 0;

            iss >> receivedKey;
            if (receivedKey != MAGIC_KEY_NODE_CONTAINER)  // wrong magic key
            {
                cerr << "Wrong magic key received " << receivedKey
                     << " while unserializing a NodePlatformContainer object." << endl;
                exit (-2);
            }

            unsigned int nPlatforms;
            iss >> nPlatforms;
            string dummy;
            getline (iss, dummy);  // read the newline before the first string value
            getline (iss, nodeName);

            // deallocate existing platforms first
            typename PlatformList::iterator lit = platforms.begin();
            for ( ; lit!=platforms.end() ; ++lit)
               delete (*lit);
            platforms.clear ();

            for (i=0 ; i<nPlatforms ; ++i)
            {
                PlatformType *plf = new PlatformType();
                plf->readObject (iss);
                platforms.push_back (plf);
            }
        }

        // compare two node containers and return true if this container
        // is considered to precede the second container.
        bool operator< (const NodePlatformContainer &ndc) const
        {
            int i;

            if (this->getPlatformCount() < ndc.getPlatformCount())
                return (true);
            if (this->getPlatformCount() > ndc.getPlatformCount())
                return (false);

            // test each platform in the list next
            typename PlatformList::const_iterator lit1 = platforms.begin();
            typename PlatformList::const_iterator lit2 = ndc.platforms.begin();
            for (i=0 ; i<this->getPlatformCount() ; ++i, ++lit1, ++lit2)
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

        // compare two node containers and return true if this container
        // is considered to succeed the second container.
        bool operator> (const NodePlatformContainer &ndc) const
        {
            int i;

            if (this->getPlatformCount() > ndc.getPlatformCount())
                return (true);
            if (this->getPlatformCount() < ndc.getPlatformCount())
                return (false);

            // test each platform in the list next
            typename PlatformList::const_iterator lit1 = platforms.begin();
            typename PlatformList::const_iterator lit2 = ndc.platforms.begin();
            for (i=0 ; i<this->getPlatformCount() ; ++i, ++lit1, ++lit2)
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

        // compare two node containers and return true if this container
        // is equal to the second container based on our ordering.
        bool operator== (const NodePlatformContainer &ndc) const
        {
            int i;

            if (this->getPlatformCount() != ndc.getPlatformCount())
                return (false);

            // test each platform in the list next
            typename PlatformList::const_iterator lit1 = platforms.begin();
            typename PlatformList::const_iterator lit2 = ndc.platforms.begin();
            for (i=0 ; i<this->getPlatformCount() ; ++i, ++lit1, ++lit2)
            {
                if (! (*(*lit1) == *(*lit2)))
                    return (false);
            }
            return (true);
        }

    };

    template <typename PlatformType>
    const int NodePlatformContainer<PlatformType>::MAGIC_KEY_NODE_CONTAINER = 0x3178af12;
};


#endif
