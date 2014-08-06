#include <iostream>
#include <sstream>
#include <string>
#include <stdlib.h>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"

#ifndef _WIN32
#include <sys/time.h>
#endif

using namespace std;

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

extern "C" int epDouble(const char* x, int M, double* gflops);
extern "C" int epSingle(const char* x, int M, double* gflops);

// ********************************************************
// Function: toString
//
// Purpose:
//   Simple templated function to convert objects into
//   strings using stringstream
//
// Arguments:
//   t: the object to convert to a string
//
// Returns:  a string representation of t
//
// Modifications:
//
// ********************************************************
template<class T> inline std::string toString(const T& t)
{
    stringstream ss;
    ss << t;
    return ss.str();
}

// ********************************************************
// Function: error
//
// Purpose:
//   Simple routine to print an error message and exit
//
// Arguments:
//   message: an error message to print before exiting
//
// ********************************************************
void error(char *message)
{
    cerr << "ERROR: " << message << endl;
    exit(1);
}

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  The user is allowed to specify
//   the size of the input data in kiB.
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Anthony Danalis
// Creation: September 08, 2009
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("class", OPT_STRING, "A", "specify input class", 'x');
}

void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{
    cout << "Running double precision test" << endl;
    RunTest<double>("EP-DP", resultDB, op);
}

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op)
{
    string x = op.getOptionString("class");
    int M = 28;
    if (x == "S") M = 24;
    else if (x == "W") M = 25;
    else if (x == "A") M = 28;
    else if (x == "B") M = 30;
    else if (x == "C") M = 32;
    else if (x == "D") M = 36;
    else if (x == "E") M = 40;

    double gflops;

    epDouble(x.c_str(), M, &gflops);

    resultDB.AddResult(testName, x, "Gflops", gflops);
}

