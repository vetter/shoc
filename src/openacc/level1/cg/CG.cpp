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

extern "C" void cgDouble(int NA, int NONZER, int NITER, double SHIFT, double RCOND, double* gflops);

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
    RunTest<double>("CG-DP", resultDB, op);
}

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op)
{
    string x = op.getOptionString("class");
    int NA = 14000;
    int NONZER = 11;
    int NITER = 15;
    double SHIFT = 20.0;
    double RCOND = 1.0e-1;
    if (x == "S")      { NA = 1400;     NONZER = 7;     NITER = 15;  SHIFT = 10.0;   }
    else if (x == "W") { NA = 7000;     NONZER = 8;     NITER = 15;  SHIFT = 12.0;   }
    else if (x == "A") { NA = 14000;    NONZER = 11;    NITER = 15;  SHIFT = 20.0;   }
    else if (x == "B") { NA = 75000;    NONZER = 13;    NITER = 75;  SHIFT = 60.0;   }
    else if (x == "C") { NA = 150000;   NONZER = 15;    NITER = 75;  SHIFT = 110.0;  }
    else if (x == "D") { NA = 1500000;  NONZER = 21;    NITER = 100; SHIFT = 500.0;  }
    else if (x == "E") { NA = 9000000;  NONZER = 26;    NITER = 100; SHIFT = 1.5e3;  }

    double gflops;

    cgDouble(NA, NONZER, NITER, SHIFT, RCOND, &gflops);

    resultDB.AddResult(testName, x, "Gflops", gflops);
}

