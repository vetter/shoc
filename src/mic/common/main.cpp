// This example from an alpha release of the Scalable HeterOgeneous Computing
// (SHOC) Benchmark Suite Alpha v1.1.4a-mic for Intel MIC architecture
// Contact: Kyle Spafford <kys@ornl.gov>
//          Rezaur Rahman <rezaur.rahman@intel.com>
//
// Copyright (c) 2011, UT-Battelle, LLC
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//   
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Oak Ridge National Laboratory, nor UT-Battelle, LLC, 
//    nor the names of its contributors may be used to endorse or promote 
//    products derived from this software without specific prior written 
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
// THE POSSIBILITY OF SUCH DAMAGE.

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include "omp.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include "omp.h"

#include "offload.h"
#include "Timer.h"

#include "OptionParser.h"
#include "ResultDatabase.h"

#ifdef  __MIC__ || __MIC2__
//#include <lmmintrin.h>
#include <pthread.h>
//#include <pthread_affinity_np.h>
#endif


// Forward Declarations
void addBenchmarkSpecOptions(OptionParser &op);
void RunBenchmark(OptionParser &op, ResultDatabase& resultDB);

int main(int argc, char **argv)
{

	OptionParser op;
  op.addOption("verbose", OPT_BOOL, "", "enable verbose output", 'v');
  op.addOption("passes", OPT_INT, "10", "specify number of passes", 'n');
  op.addOption("size", OPT_INT, "1", "specify problem size", 's');
  op.addOption("target", OPT_INT, "0", "specify MIC target device number", 't');
  
  // If benchmark has any specific options, add those
  addBenchmarkSpecOptions(op);
  
  if (!op.parse(argc, argv))
  {
     op.usage();
     return -1;
  }

  ResultDatabase resultDB;
  // Run the test
  RunBenchmark(op, resultDB);

  // Print out results to stdout
  resultDB.DumpDetailed(cout);

	return 0;
}

