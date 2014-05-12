
#ifndef MULTI_NODE_CONTAINER_H
#define MULTI_NODE_CONTAINER_H

#include "NodeIDList.h"
#include "SerializableObject.h"
#include <map>
#include <stdlib.h>
#include <iomanip>

#ifdef PARALLEL
#include <ParallelMerge.h>
#endif

// ****************************************************************************
// Class: MultiNodeContainer
//
// Purpose:
//   A generic container for aggregating and storing different node
//   configurations represented by objects of type NodeContainer.
//
// Notes:     The template parameter type must have well defined
//   comparison operators '<', '>' and '==', and implement the
//   SerializableObject interface.
//
// Programmer: Gabriel Marin
// Creation: August 25, 2009
//
// Modifications:
//
// ****************************************************************************

namespace SHOC {

    template <typename NodeContainer>
    class MultiNodeContainer : public SerializableObject
#ifdef PARALLEL
                      , public ParallelTreeMerge<char>
#endif
    {
    private:
        // ********************************************************************
        // Class: CompareNodeContainers
        //
        // Purpose:
        //   A template comparison function for NodeContainer objects.
        //
        // Programmer: Gabriel Marin
        // Creation: August 25, 2009
        //
        // Modifications:
        //
        // ********************************************************************
        template <typename Container>
        class CompareNodeContainers : public std::binary_function<Container,
                     Container, bool>
        {
        public:
            bool operator () (Container const &n1, Container const &n2) const
            {
                return (n1 < n2);
            }
        };

        typedef std::map <NodeContainer, NodeIDList,
               CompareNodeContainers<NodeContainer> > NodeConfigurationMap;
        NodeConfigurationMap configs;
        string workbuf;

        static const int MAGIC_KEY_MULTI_NODE_CONTAINER;

#ifdef PARALLEL
        // getMergeData and processMergeData are pure virtual functions
        // of the class ParallelTreeMerge. They act as callbacks before
        // and after each merge step for the sender and the receiver
        // processes respectively
        virtual const char* getMergeData (int *dsize, int _key = 0)
        {
            ostringstream oss;
            writeObject (oss);
            workbuf = oss.str();
            if (dsize!=0) *dsize = workbuf.size()+1;
            return (workbuf.c_str());
        }

        virtual void processMergeData (const char *_data, int size,
                     int _key = 0)
        {
            string stemp (_data);
            istringstream iss (stemp);
            MultiNodeContainer<NodeContainer> tempContainer;
            tempContainer.readObject(iss);
            this->merge (tempContainer);
        }
#endif

    public:
        MultiNodeContainer ()
        {
        }

        // Copy constructor
        MultiNodeContainer (NodeContainer &ndc)
        {
            NodeIDList nlist;
            nlist.push_back (ndc.getNodeName ());
            configs.insert (typename NodeConfigurationMap::value_type (ndc, nlist));
        }

        // destructor
        ~MultiNodeContainer ()
        {
            configs.clear();
        }

        // add a new node configuration to this object
        void addNodeConfiguration (NodeContainer &ndc)
        {
            typename NodeConfigurationMap::iterator nit = configs.find (ndc);
            if (nit == configs.end())  // not found
            {
                NodeIDList nlist;
                nlist.push_back (ndc.getNodeName ());
                configs.insert (typename NodeConfigurationMap::value_type (ndc, nlist));
            } else
            {
                nit->second.push_back (ndc.getNodeName ());
            }
        }

        // merge two MultiNodeContainers, placing the result into this
        // container.
        void merge (MultiNodeContainer<NodeContainer> &mnc)
        {
            typename NodeConfigurationMap::iterator nit2 = mnc.configs.begin();
            for ( ; nit2!=mnc.configs.end() ; ++nit2)
            {
                typename NodeConfigurationMap::iterator nit1 = configs.find(nit2->first);
                if (nit1 == configs.end())  // not found
                {
                    configs.insert (typename NodeConfigurationMap::value_type (nit2->first, nit2->second));
                } else
                {
                    nit1->second.insert (nit1->second.end(), nit2->second.begin(), nit2->second.end());
                }
            }
        }

        // pretty print the content of this container
        void Print (ostream &os) const
        {
            int i=1, numConfigs = configs.size();
            os << "Number distinct configurations: " << numConfigs << endl;
            typename NodeConfigurationMap::const_iterator nit = configs.begin();
            for ( ; nit!=configs.end() ; ++nit, ++i) {
                os << i << ". [" << endl;
                nit->first.Print(os);
                os << "], Hosts = {";
                int j = 0;
                NodeIDList::const_iterator lit = nit->second.begin();
                for ( ; lit!=nit->second.end() ; ++lit, ++j)
                {
                    if (j%3 == 0)
                        os << "\n";
                    else
                        os << "  ";
                    os << setiosflags(ios::left) << setw(25) << (*lit);
                }
                os << "\n}\n\n";
            }
        }

        // implements the serialization method of the SerializableObject
        // abstract class
        virtual void writeObject (ostringstream &oss) const
        {
            int numConfigs = configs.size();
            oss << " " << MAGIC_KEY_MULTI_NODE_CONTAINER
                << " " << numConfigs << "\n";

            typename NodeConfigurationMap::const_iterator mit = configs.begin();
            for ( ; mit!=configs.end() ; ++mit)
            {
                mit->first.writeObject (oss);

                // now write the Node ID List. It is not a class, so I have to
                // inline the code here
                int numHosts = mit->second.size();
                oss << " " << MAGIC_KEY_NODE_ID_LIST
                    << " " << numHosts << "\n";
                NodeIDList::const_iterator lit = mit->second.begin();
                for ( ; lit!=mit->second.end() ; ++lit)
                    oss << (*lit) << "\n";
            }
        }

        // implements the un-serialization method of the SerializableObject
        // abstract class
        virtual void readObject (istringstream &iss)
        {
            int i, j;
            int receivedKey = 0;

            iss >> receivedKey;
            if (receivedKey != MAGIC_KEY_MULTI_NODE_CONTAINER)  // wrong magic key
            {
                cerr << "Wrong magic key received " << receivedKey
                     << " while unserializing a MultiNodeContainer object." << endl;
                exit (-2);
            }

            int numConfigs;
            iss >> numConfigs;
            string dummy;
            getline (iss, dummy);  // read the newline before the first string value

            // before reading the new configs, I have to deallocate any
            // prior configs
            configs.clear();

            for (i=0 ; i<numConfigs ; ++i)
            {
                NodeIDList nlist;
                NodeContainer nc(false);
                nc.readObject (iss);

                int numHosts;
                iss >> receivedKey;
                if (receivedKey != MAGIC_KEY_NODE_ID_LIST)  // wrong magic key
                {
                    cerr << "Wrong magic key received " << receivedKey
                         << " while unserializing a NodeIDList object." << endl;
                    exit (-2);
                }
                iss >> numHosts;
                getline (iss, dummy);  // read the newline before the first string value

                for (j=0 ; j<numHosts ; ++j) {
                    iss >> dummy;
                    nlist.push_back (dummy);
                }
                configs.insert (typename NodeConfigurationMap::value_type (nc, nlist));
            }
        }
    };

    template <typename NodeContainer>
    const int MultiNodeContainer<NodeContainer>::MAGIC_KEY_MULTI_NODE_CONTAINER = 0x5b071e23;
};

#endif
