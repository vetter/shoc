#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

#include "Timer.h"
#include "OptionParser.h"
#include "ResultDatabase.h"

using namespace std;


extern "C" int FindKeyspaceSize(int, int);
extern "C" void md5_2words(unsigned int *, unsigned int, unsigned int *);
extern "C" void IndexToKey(unsigned int, int, int, unsigned char *);
extern "C" double FindKeyWithDigest(const unsigned int *, const int, const int,
                                    int *, unsigned char *, unsigned int *);

// ****************************************************************************
// Function:  AsHex
//
// Purpose:
///   For a given key string, return the raw hex string for its bytes.
//
// Arguments:
//   vals       key string
//   len        length of key string
//
// Programmer:  Jeremy Meredith (OpenACC version: Graham Lopez)
// Creation:    Aug 2014
//
// Modifications:
// ****************************************************************************
std::string AsHex(unsigned char *vals, int len)
{
    ostringstream out;
    char tmp[256];
    for (int i=0; i<len; ++i)
    {
        sprintf(tmp, "%2.2X", vals[i]);
        out << tmp;
    }
    return out.str();
}


// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith (OpenACC version: Graham Lopez)
// Creation: Aug 2014
//
// Modifications:
//
// ****************************************************************************
void
addBenchmarkSpecOptions(OptionParser &op)
{
   ;
}


// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the MD5 Hash benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith (OpenACC modifications: Graham Lopez)
// Creation: Aug 2014
//
// Modifications:
//
// ****************************************************************************
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &opts)
{
    bool verbose = opts.getOptionBool("verbose");

    int size = opts.getOptionInt("size");
    if (size < 1 || size > 4)
    {
        cerr << "ERROR: Invalid size parameter\n";
        return;
    }

    //
    // Determine the shape/size of key space
    //
    const int sizes_byteLength[]  = { 7,  5,  6,  5};
    const int sizes_valsPerByte[] = {10, 36, 26, 70};

    const int byteLength = sizes_byteLength[size-1];   
    const int valsPerByte = sizes_valsPerByte[size-1];

    char atts[1024];
    sprintf(atts, "%dx%d", byteLength, valsPerByte);

    if (verbose)
        cout << "Searching keys of length " << byteLength << " bytes "
             << "and " << valsPerByte << " values per byte" << endl;

    const int keyspace = FindKeyspaceSize(byteLength, valsPerByte);
    if (keyspace < 0)
    {
        cerr << "Error: more than 2^31 bits of entropy is unsupported.\n";
        return;
    }

    if (byteLength > 7)
    {
        cerr << "Error: more than 7 byte key length is unsupported.\n";
        return;
    }

    if (verbose)
        cout << "|keyspace| = " << keyspace << " ("<<int(keyspace/1e6)<<"M)" << endl;

    //
    // Choose a random key from the keyspace, and calculate its hash.
    //
    //srandom(12345);
    srandom(time(NULL));

    int passes = opts.getOptionInt("passes");

    for (int pass = 0 ; pass < passes ; ++pass)
    {
        int randomIndex = random() % keyspace;;
        unsigned char randomKey[8] = {0,0,0,0, 0,0,0,0};
        unsigned int randomDigest[4];
        IndexToKey(randomIndex, byteLength, valsPerByte, randomKey);
        md5_2words((unsigned int*)randomKey, byteLength, randomDigest);

        if (verbose)
        {
            cout << endl;
            cout << "--- pass " << pass << " ---" << endl;
            cout << "Looking for random key:" << endl;
            cout << " randomIndex = " << randomIndex << endl;
            cout << " randomKey   = 0x" << AsHex(randomKey, 8/*byteLength*/) << endl;
            cout << " randomDigest= " << AsHex((unsigned char*)randomDigest, 16) << endl;
        }

        //
        // Use the GPU to brute force search the keyspace for this key.
        //
        unsigned int foundDigest[4] = {0,0,0,0};
        int foundIndex = -1;
        unsigned char foundKey[8] = {0,0,0,0, 0,0,0,0};

        double t; // in seconds
        t = FindKeyWithDigest(randomDigest, byteLength, valsPerByte,
                              &foundIndex, foundKey, foundDigest);

        //
        // Calculate the rate and add it to the results
        //
        double rate = (double(keyspace) / double(t)) / 1.e9;
        if (verbose)
        {
            cout << "time = " << t << " sec, rate = " << rate << " GHash/sec\n";
        }

        //
        // Double check everything matches (index, key, hash).
        //
        if (foundIndex != randomIndex)
        {
            cerr << "\nERROR: mismatch in key index found.\n";
            rate = FLT_MAX;
        }
        else if (foundKey[0] != randomKey[0] ||
            foundKey[1] != randomKey[1] ||
            foundKey[2] != randomKey[2] ||
            foundKey[3] != randomKey[3] ||
            foundKey[4] != randomKey[4] ||
            foundKey[5] != randomKey[5] ||
            foundKey[6] != randomKey[6] ||
            foundKey[7] != randomKey[7])
        {
            cerr << "\nERROR: mismatch in key value found.\n";
            rate = FLT_MAX;
        }        
        else if (foundDigest[0] != randomDigest[0] ||
            foundDigest[1] != randomDigest[1] ||
            foundDigest[2] != randomDigest[2] ||
            foundDigest[3] != randomDigest[3])
        {
            cerr << "\nERROR: mismatch in digest of key.\n";
            rate = FLT_MAX;
        }
        else
        {
            if (verbose)
                cout << endl << "Successfully found match (index, key, hash):" << endl;
        }

        //
        // Add the calculated performancethe results
        //
        resultDB.AddResult("MD5Hash", atts, "GHash/s", rate);

        if (verbose)
        {
            cout << " foundIndex  = " << foundIndex << endl;
            cout << " foundKey    = 0x" << AsHex(foundKey, 8/*byteLength*/) << endl;
            cout << " foundDigest = " << AsHex((unsigned char*)foundDigest, 16) << endl;
            cout << endl;
        }
    }

    return;


}

