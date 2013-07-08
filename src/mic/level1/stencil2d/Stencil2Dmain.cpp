// This example from an alpha release of the Scalable HeterOgeneous Computing
// (SHOC) Benchmark Suite Alpha v1.1.4a-mic for Intel MIC architecture
// Contact: Jeffrey Vetter <vetter@ornl.gov>
//          Rezaur Rahman <rezaur.rahman@intel.com>
//
// Copyright (c) 2011-2013, UT-Battelle, LLC
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

#if defined(PARALLEL)
#include "mpi.h"
#endif // defined(PARALLEL)

#include <iostream>
#include <sstream>
#include <assert.h>
#include "omp.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"
#include "BadCommandLine.h"
#include "InvalidArgValue.h"
#include "Matrix2D.h"
#include "HostStencilFactory.h"
#include "HostStencil.h"
#include "MICStencilFactory.h"
#include "MICStencil.h"
#include "InitializeMatrix2D.h"
#include "InitializeMatrix2D.cpp"
#include "ValidateMatrix2D.h"
#include "ValidateMatrix2D.cpp"
#include "StencilUtil.h"
#include "StencilUtil.cpp"
#include "SerialStencilUtil.h"
#include "SerialStencilUtil.cpp"
#include "StencilFactory.cpp"
#include "CommonMICStencilFactory.cpp"
#include "HostStencil.cpp"
#include "MICStencil.cpp"
//#include <lmmintrin.h>
#if defined(PARALLEL)
#include "ParallelResultDatabase.h"
#include "MPIHostStencilFactory.cpp"
#include "MPIHostStencil.cpp"
#include "MPIStencilUtil.cpp"
#include "MPI2DGridProgram.cpp"
#else
#include "HostStencilFactory.cpp"
#include "MICStencilFactory.cpp"
#endif // defined(PARALLEL)


// prototypes of auxiliary functions defined in this file or elsewhere
void CheckOptions( const OptionParser& opts );

void EnsureStencilInstantiation( void );


template<class T>
void
DoTest( const char* timerDesc, ResultDatabase& resultDB, OptionParser& opts )
{
    StencilFactory<T>* stdStencilFactory = NULL;
    Stencil<T>* stdStencil = NULL;
    StencilFactory<T>* testStencilFactory = NULL;
    Stencil<T>* testStencil = NULL;

    try
    {
        stdStencilFactory = new HostStencilFactory<T>;
        testStencilFactory = new MICStencilFactory<T>;
        assert( (stdStencilFactory != NULL) && (testStencilFactory != NULL) );

        // do a sanity check on option values
        CheckOptions( opts );
        stdStencilFactory->CheckOptions( opts );
        testStencilFactory->CheckOptions( opts );

        // extract and validate options
        std::vector<long long> arrayDims = opts.getOptionVecInt( "customSize" );
        if( arrayDims.size() != 2 )
        {
            cerr << "Dim size: " << arrayDims.size() << "\n";
            //throw InvalidArgValue( "all overall dimensions must be positive" );
        }
        if (arrayDims[0] == 0) // User has not specified a custom size
        {
            const int probSizes[4] = { 768, 1408, 2048, 4096 };
            int sizeClass = opts.getOptionInt("size");
            if (!(sizeClass >= 0 && sizeClass < 5))
            {
                //throw InvalidArgValue( "Size class must be between 1-4" );
            }
            arrayDims[0] = arrayDims[1] =probSizes[sizeClass - 1];
        }

        long int seed = (long)opts.getOptionInt( "seed" );
        bool beVerbose = opts.getOptionBool( "verbose" );
        unsigned int nIters = (unsigned int)opts.getOptionInt( "num-iters" );
        double valErrThreshold = (double)opts.getOptionFloat( "val-threshold" );
        unsigned int nValErrsToPrint = (unsigned int)opts.getOptionInt( "val-print-limit" );

#if defined(PARALLEL)
        unsigned int haloWidth = (unsigned int)opts.getOptionInt( "iters-per-exchange" );
#else
        unsigned int haloWidth = 1;
#endif // defined(PARALLEL)

        float haloVal = (float)opts.getOptionFloat( "haloVal" );

        // build a description of this experiment
        std::ostringstream experimentDescriptionStr;
        experimentDescriptionStr 
            << nIters << ':'
            << arrayDims[0] << 'x' << arrayDims[1];

        unsigned int nPasses =(unsigned int)opts.getOptionInt( "passes" );
        unsigned int nWarmupPasses = (unsigned int)opts.getOptionInt( "warmupPasses" );

        // compute the expected result on the host
#if defined(PARALLEL)
        int cwrank;
        MPI_Comm_rank( MPI_COMM_WORLD, &cwrank );
        if( cwrank == 0 )
        {
#endif // defined(PARALLEL)
        std::cout << "\nPerforming stencil operation on host for later comparison with MIC output\n"
            << "Depending on host capabilities, this may take a while."
            << std::endl;
#if defined(PARALLEL)
        }
#endif // defined(PARALLEL)
        Matrix2D<T> exp( arrayDims[0] + 2*haloWidth, 
                            arrayDims[1] + 2*haloWidth );
        Initialize<T> init( seed,
                        haloWidth,
                        haloVal );	

        init( exp );
        if( beVerbose )
        {
            std::cout << "initial state:\n" << exp << std::endl;
        }
        Stencil<T>* stdStencil = stdStencilFactory->BuildStencil( opts );
        (*stdStencil)( exp, nIters );
        if( beVerbose )
        {
            std::cout << "expected result:\n" << exp << std::endl;
        }
	

        // compute the result on the MIC device
        Matrix2D<T> data( arrayDims[0] + 2*haloWidth, 
                            arrayDims[1] + 2*haloWidth );
        Stencil<T>* testStencil = testStencilFactory->BuildStencil( opts );

        // Compute the number of floating point operations we will perform.
        //
        // Note: in the truly-parallel case, we count flops for redundant 
        // work due to the need for a halo.
        // But we do not add to the count for the local 1-wide halo since
        // we aren't computing new values for those items.
        unsigned long npts = (arrayDims[0] + 2*haloWidth - 2) *
                            (arrayDims[1] + 2*haloWidth - 2);
#if defined(PARALLEL)
        MPICUDAStencil<T>* mpiTestStencil = static_cast<MPICUDAStencil<T>*>( testStencil );
        assert( mpiTestStencil != NULL );
        int participating = mpiTestStencil->ParticipatingInProgram() ? 1 : 0;
        int numParticipating = 0;
        MPI_Allreduce( &participating,      // src
                        &numParticipating,  // dest
                        1,                  // count
                        MPI_INT,            // type
                        MPI_SUM,            // op
                        MPI_COMM_WORLD );   // communicator
        npts *= numParticipating;
#endif // defined(PARALLEL)

        // In our 9-point stencil, there are 11 floating point operations 
        // per point (3 multiplies and 11 adds):
        //
        // newval = weight_center * centerval +
        //      weight_cardinal * (northval + southval + eastval + westval) + 
        //      weight_diagnoal * (neval + nwval + seval + swval)
        // 
        // we do this stencil operation 'nIters' times
        unsigned long nflops = npts * 11 * nIters;

#if defined(PARALLEL)
        if( cwrank == 0 )
        {
#endif // defined(PARALLEL)
            std::cout << "Performing " << nWarmupPasses << " warmup passes...";
#if defined(PARALLEL)
        }
#endif // defined(PARALLEL)

    for( unsigned int pass = 0; pass < nWarmupPasses; pass++ )
    {
        init(data);
        (*testStencil)( data, nIters );
    }
#if defined(PARALLEL)
        if( cwrank == 0 )
        {
#endif // defined(PARALLEL)
            std::cout << "done." << std::endl;
#if defined(PARALLEL)
        }
#endif // defined(PARALLEL)


#if defined(PARALLEL)
        MPI_Comm_rank( MPI_COMM_WORLD, &cwrank );
        if( cwrank == 0 )
        {
#endif // defined(PARALLEL)
        std::cout << "\nPerforming stencil operation on chosen device, " 
            << nPasses << " passes.\n"
            << "Depending on chosen device, this may take a while."
            << std::endl;
#if defined(PARALLEL)
        }
#endif // defined(PARALLEL)

#if !defined(PARALLEL)
        std::cout << "At the end of each pass the number of validation\nerrors observed will be printed to the standard output." 
            << std::endl;
#endif // !defined(PARALLEL)


        for( unsigned int pass = 0; pass < nPasses; pass++ )
        {
#if !defined(PARALLEL)
            std::cout << "pass " << pass << ": ";
#endif // !defined(PARALLEL)
            init( data );

            int kernelTimerHandle = Timer::Start();
            (*testStencil)( data, nIters );
            double elapsedTime = Timer::Stop(kernelTimerHandle, "stencil2d");
            double gflops = (nflops / elapsedTime) / 1e9;
            resultDB.AddResult( timerDesc,
                                    experimentDescriptionStr.str(),
                                    "GFLOPS",
                                    gflops );
            if( beVerbose )
            {
                std::cout << "observed result, pass " << pass << ":\n" 
                    << data 
                    << std::endl;
            }

            // validate the result
#if defined(PARALLEL)
            StencilValidater<T>* validater = new MPIStencilValidater<T>;
#else
            StencilValidater<T>* validater = new SerialStencilValidater<T>;            
#endif // defined(PARALLEL)

            validater->ValidateResult( exp,
                            data,
                            valErrThreshold,
                            nValErrsToPrint );
        }
    }
    catch( ... )
    {
        // clean up - abnormal termination
        // wish we didn't have to do this, but C++ exceptions do not 
        // support a try-catch-finally approach
        delete stdStencil;
        delete stdStencilFactory;
        delete testStencil;
        delete testStencilFactory;
        throw;
    }

    // clean up - normal termination
    delete stdStencil;
    delete stdStencilFactory;
    delete testStencil;
    delete testStencilFactory;
}




void
RunBenchmark(OptionParser& opts, ResultDatabase& resultDB )
{
    int device;

#if defined(PARALLEL)
    int cwrank;
    MPI_Comm_rank( MPI_COMM_WORLD, &cwrank );
#endif // defined(PARALLEL)
#if defined(PARALLEL)
    if( cwrank == 0 )
    {
#endif // defined(PARALLEL)
        std::cout << "Running single precision test" << std::endl;
#if defined(PARALLEL)
    }
#endif // defined(PARALLEL)
    //omp_set_num_threads(124);
    DoTest<float>( "SP_Sten2D", resultDB, opts );

    // check if we can run double precision tests
    if( //deviceProps.major == 1) && (deviceProps.minor >= 3)) ||
        //eviceProps.major >= 2))
        1)
    {
#if defined(PARALLEL)
        if( cwrank == 0 )
        {
#endif // defined(PARALLEL)
            std::cout << "DP supported\n" << std::endl;
#if defined(PARALLEL)
        }
#endif // defined(PARALLEL)
	//omp_set_num_threads(93);
        DoTest<double>( "DP_Sten2D", resultDB, opts );
    }
    else
    {
#if defined(PARALLEL)
        if( cwrank == 0 )
        {
#endif // defined(PARALLEL)
            std::cout << "Double precision not supported - skipping" << std::endl;
#if defined(PARALLEL)
        }
#endif // defined(PARALLEL)
        // resultDB requires neg entry for every possible result
        int nPasses = (int)opts.getOptionInt( "passes" );
        for( int p = 0; p < nPasses; p++ )
        {
            resultDB.AddResult( (const char*)"DP_Sten2D", "N/A", "s", FLT_MAX);
        }
    }
	
}


// Adds command line options to given OptionParser
void
addBenchmarkSpecOptions( OptionParser& opts )
{
    opts.addOption("customSize", OPT_VECINT, "0,0", "specify custom problem size");
    opts.addOption( "num-iters", OPT_INT, "1000", "number of stencil iterations" );
    opts.addOption( "weight-center", OPT_FLOAT, "0.25", "center value weight" );
    opts.addOption( "weight-cardinal", OPT_FLOAT, "0.15", "cardinal values weight" );
    opts.addOption( "weight-diagonal", OPT_FLOAT, "0.05", "diagonal values weight" );
    opts.addOption( "seed", OPT_INT, "71594", "random number generator seed" );
    opts.addOption( "val-threshold", OPT_FLOAT, "0.01", "validation error threshold" );
    opts.addOption( "val-print-limit", OPT_INT, "15", "number of validation errors to print" );
    opts.addOption( "haloVal", OPT_FLOAT, "0.0", "value to use for halo data" );

    opts.addOption( "warmupPasses", OPT_INT, "1", "Number of warmup passes to do before starting timings", 'w' );

#if defined(PARALLEL)
    opts.addOption( "msize", OPT_VECINT, "2,2", "MPI 2D grid topology dimensions" );
    opts.addOption( "iters-per-exchange", OPT_INT, "1", "Number of local iterations between MPI boundary exchange operations (also, halo width)" );
#endif // defined(PARALLEL)
}


// validate stencil-independent values
void
CheckOptions( const OptionParser& opts )
{
    // check matrix dimensions - must be 2d, must be positive
    std::vector<long long> arrayDims = opts.getOptionVecInt( "customSize" );
    if( arrayDims.size() != 2 )
    {
        throw InvalidArgValue( "overall size must have two dimensions" );
    }
    if( (arrayDims[0] < 0) || (arrayDims[1] < 0) )
    {
        throw InvalidArgValue( "each size dimension must be positive" );
    }

    // validation error threshold must be positive
    float valThreshold = opts.getOptionFloat( "val-threshold" );
    if( valThreshold <= 0.0f )
    {
        throw InvalidArgValue( "validation threshold must be positive" );
    }

    // number of validation errors to print must be non-negative
    int nErrsToPrint = opts.getOptionInt( "val-print-limit" );
    if( nErrsToPrint < 0 )
    {
        throw InvalidArgValue( "number of validation errors to print must be non-negative" );
    }
	
    int nWarmupPasses = opts.getOptionInt( "warmupPasses" );
    if( nWarmupPasses < 0 )
    {
        throw InvalidArgValue( "number of warmup passes must be non-negative" );
    }
}

