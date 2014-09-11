#if defined(PARALLEL)
#include "mpi.h"
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"
#include "InvalidArgValue.h"
#include "BadCommandLine.h"
#include "Matrix2D.h"
#include "Matrix2D.cpp"
#include "Matrix2DFileSupport.cpp"
#include "InitializeMatrix2D.h"
#include "InitializeMatrix2D.cpp"
#include "ValidateMatrix2D.h"
#include "ValidateMatrix2D.cpp"
#include "Stencil.h"
#include "StencilUtil.cpp"
#include "SerialStencilUtil.cpp"
#include "StencilFactory.cpp"
#include "CommonOpenCLStencilFactory.cpp"
#include "HostStencil.cpp"
#include "OpenCLStencil.cpp"

#if defined(PARALLEL)
#include "ParallelResultDatabase.h"
#include "MPIHostStencilFactory.cpp"
#include "MPIHostStencil.cpp"
#include "MPIOpenCLStencilFactory.cpp"
#include "MPIOpenCLStencil.cpp"
#include "MPIStencilUtil.cpp"
#include "MPI2DGridProgram.cpp"
#else
#include "HostStencilFactory.cpp"
#include "OpenCLStencilFactory.cpp"
#endif // defined(PARALLEL)


// prototypes of auxiliary functions defined in this file
void CheckOptions( const OptionParser& opts );




template<class T>
void
DoTest( std::string testName,
        cl_device_id dev,
        cl_context ctx,
        cl_command_queue queue,
        ResultDatabase& resultDB,
        OptionParser& opts,
        std::string compileFlags )
{
    StencilFactory<T>* stdStencilFactory = NULL;
    Stencil<T>* stdStencil = NULL;
    StencilFactory<T>* testStencilFactory = NULL;
    Stencil<T>* testStencil = NULL;

    try
    {
        // obtain factories for the stencils we use
#if defined(PARALLEL)
        stdStencilFactory = new MPIHostStencilFactory<T>;
        testStencilFactory = new MPIOpenCLStencilFactory<T>( dev, ctx, queue );
#else
        stdStencilFactory = new HostStencilFactory<T>;
        testStencilFactory = new OpenCLStencilFactory<T>( dev, ctx, queue );
#endif // defined(PARALLEL)
        assert( (stdStencilFactory != NULL) && (testStencilFactory != NULL) );

        // do a sanity check on option values
        CheckOptions( opts );
        stdStencilFactory->CheckOptions( opts );
        testStencilFactory->CheckOptions( opts );

        // extract and validate options
        std::vector<long long> arrayDims = opts.getOptionVecInt( "customSize" );
        if( arrayDims.size() != 2 )
        {
            throw InvalidArgValue( "size must have two dimensions" );
        }
        if (arrayDims[0] == 0) // User has not specified a custom size
        {
            int sizeClass = opts.getOptionInt("size");
            arrayDims = StencilFactory<T>::GetStandardProblemSize( sizeClass );
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
#endif // PARALLEL
        float haloVal = (float)opts.getOptionFloat( "haloVal" );

        unsigned int nPasses = (unsigned int)opts.getOptionInt( "passes" );
        unsigned int nWarmupPasses = (unsigned int)opts.getOptionInt( "warmupPasses" );

        // build a description of this experiment
        std::vector<long long> lDims = opts.getOptionVecInt( "lsize" );
        assert( lDims.size() == 2 );
        std::ostringstream experimentDescriptionStr;
        experimentDescriptionStr
            << nIters << ':'
            << arrayDims[0] << 'x' << arrayDims[1] << ':'
            << lDims[0] << 'x' << lDims[1];


        // compute the expected result on the host
        // or read it from a pre-existing file
        std::string matrixFilenameBase = (std::string)opts.getOptionString( "expMatrixFile" );
#if defined(PARALLEL)
        int cwrank;
        MPI_Comm_rank( MPI_COMM_WORLD, &cwrank );
        if( cwrank == 0 )
        {
#endif // defined(PARALLEL)
        if( !matrixFilenameBase.empty() )
        {
            std::cout << "\nReading expected stencil operation result from file for later comparison with OpenCL output\n"
                << std::endl;
        }
        else
        {
            std::cout << "\nPerforming stencil operation on host for later comparison with OpenCL output\n"
                << "Depending on host capabilities, this may take a while."
                << std::endl;
        }
#if defined(PARALLEL)
        }
#endif // defined(PARALLEL)
        Matrix2D<T> expected( arrayDims[0] + 2*haloWidth,
                                arrayDims[1] + 2*haloWidth );
        Initialize<T> init( seed, haloWidth, haloVal );

        bool haveExpectedData = false;
        if( ! matrixFilenameBase.empty() )
        {
            bool readOK = ReadMatrixFromFile( expected, GetMatrixFileName<T>( matrixFilenameBase ) );
            if( readOK )
            {

                if( (expected.GetNumRows() != arrayDims[0] + 2*haloWidth) ||
                    (expected.GetNumColumns() != arrayDims[1] + 2*haloWidth) )
                {
                    std::cerr << "The matrix read from file \'"
                        << GetMatrixFileName<T>( matrixFilenameBase )
                        << "\' does not match the matrix size specified on the command line.\n";
                    expected.Reset( arrayDims[0] + 2*haloWidth, arrayDims[1] + 2*haloWidth );
                }
                else
                {
                    haveExpectedData = true;
                }
            }

            if( !haveExpectedData )
            {
                std::cout << "\nSince we could not read the expected matrix values,\nperforming stencil operation on host for later comparison with OpenCL output.\n"
                    << "Depending on host capabilities, this may take a while."
                    << std::endl;
            }
        }
        if( !haveExpectedData )
        {
            init( expected );
            haveExpectedData = true;
            if( beVerbose )
            {
                std::cout << "initial state:\n" << expected << std::endl;
            }
            stdStencil = stdStencilFactory->BuildStencil( opts );
            (*stdStencil)( expected, nIters );
        }
        if( beVerbose )
        {
            std::cout << "expected result:\n" << expected << std::endl;
        }

        // determine whether we are to save the expected matrix values to a file
        // to speed up future runs
        matrixFilenameBase = (std::string)opts.getOptionString( "saveExpMatrixFile" );
        if( !matrixFilenameBase.empty() )
        {
            SaveMatrixToFile( expected, GetMatrixFileName<T>( matrixFilenameBase ) );
        }
        assert( haveExpectedData );

        // compute the result on the OpenCL device(s)
        Matrix2D<T> data( arrayDims[0] + 2*haloWidth,
                                arrayDims[1] + 2*haloWidth );
        testStencil = testStencilFactory->BuildStencil( opts );

        // Compute the number of floating point operations we will perform.
        //
        // Note: in the truly-parallel case, we count flops for redundant
        // work due to the need for a halo.
        // But we do not add to the count for the local 1-wide halo since
        // we aren't computing new values for those items.
        unsigned long npts = (arrayDims[0] + 2*haloWidth - 2) *
                            (arrayDims[1] + 2*haloWidth - 2);
#if defined(PARALLEL)
        MPIOpenCLStencil<T>* mpiTestStencil = static_cast<MPIOpenCLStencil<T>*>( testStencil );
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
        if( cwrank == 0 )
        {
#endif // defined(PARALLEL)
        std::cout << "\nPerforming stencil operation on chosen OpenCL device, "
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

            int timerHandle = Timer::Start();
            (*testStencil)( data, nIters );
            double elapsedTime = Timer::Stop( timerHandle, "OpenCL stencil" );


            // find and report the computation rate
            double gflops = (nflops / elapsedTime) / 1e9;
            resultDB.AddResult( testName,
                                    experimentDescriptionStr.str(),
                                    "GFLOPS",
                                    gflops );
            if( beVerbose )
            {
                std::cout << "observed result, pass " << (pass + 1) << ":\n"
                    << data
                    << std::endl;
            }

            // validate the result
#if defined(PARALLEL)
            StencilValidater<T>* validater = new MPIStencilValidater<T>;
#else
            StencilValidater<T>* validater = new SerialStencilValidater<T>;
#endif // defined(PARALLEL)

            validater->ValidateResult( expected,
                            data,
                            valErrThreshold,
                            nValErrsToPrint );
        }
    }
    catch( ... )
    {
        // clean up - abnormal termination
        // wish we did not have to repeat this cleanup here, but
        // C++ exceptions do not support a try-catch-finally approach
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
RunBenchmark( cl_device_id dev,
                cl_context ctx,
                cl_command_queue queue,
                ResultDatabase& resultDB,
                OptionParser& op )
{
#if defined(PARALLEL)
    int cwrank;
#endif // defined(PARALLEL)

    // single precision
    DoTest<float>( "SP_Sten2D", dev, ctx, queue, resultDB, op, "-DSINGLE_PRECISION" );

    // double precision - might not be supported
    if( checkExtension( dev, "cl_khr_fp64" ))
    {
#if defined(PARALLEL)
        MPI_Comm_rank( MPI_COMM_WORLD, &cwrank );
        if( cwrank == 0 )
        {
#endif // defined(PARALLEL)
            std::cout << "\nDP supported\n";
#if defined(PARALLEL)
        }
#endif // defined(PARALLEL)
        DoTest<double>( "DP_Sten2D", dev, ctx, queue, resultDB, op, "-DK_DOUBLE_PRECISION" );
    }
    else if( checkExtension( dev, "cl_amd_fp64" ))
    {
#if defined(PARALLEL)
        MPI_Comm_rank( MPI_COMM_WORLD, &cwrank );
        if( cwrank == 0 )
        {
#endif // defined(PARALLEL)
            std::cout << "\nDP supported\n";
#if defined(PARALLEL)
        }
#endif // defined(PARALLEL)
        DoTest<double>( "DP_Sten2D", dev, ctx, queue, resultDB, op, "-DAMD_DOUBLE_PRECISION" );
    }
    else
    {
#if defined(PARALLEL)
        MPI_Comm_rank( MPI_COMM_WORLD, &cwrank );
        if( cwrank == 0 )
        {
#endif // defined(PARALLEL)
            std::cout << "\nDP not supported\n";
#if defined(PARALLEL)
        }
#endif // defined(PARALLEL)

        // resultDB requires an entry for every possible result
        int nPasses = (int)op.getOptionInt( "passes" );
        for( unsigned int p = 0; p < nPasses; p++ )
        {
            resultDB.AddResult( (const char*)"DP_Sten2D",
                                "N/A",
                                "GFLOPS",
                                FLT_MAX );
        }
    }

    std::cout << '\n' << std::endl;
}


// Adds command line options to given OptionParser
void
addBenchmarkSpecOptions( OptionParser& opts )
{
    opts.addOption("customSize", OPT_VECINT, "0,0", "specify custom problem size x,y");
    opts.addOption( "lsize", OPT_VECINT, "8,256", "work-item dimensions" );
    opts.addOption( "num-iters", OPT_INT, "1000", "number of stencil iterations" );
    opts.addOption( "weight-center", OPT_FLOAT, "0.25", "center value weight" );
    opts.addOption( "weight-cardinal", OPT_FLOAT, "0.15", "cardinal values weight" );
    opts.addOption( "weight-diagonal", OPT_FLOAT, "0.05", "diagonal values weight" );
    opts.addOption( "seed", OPT_INT, "71594", "random number generator seed" );
    opts.addOption( "val-threshold", OPT_FLOAT, "0.01", "validation error threshold" );
    opts.addOption( "val-print-limit", OPT_INT, "15", "number of validation errors to print" );
    opts.addOption( "haloVal", OPT_FLOAT, "0.0", "value to use for halo data" );

    opts.addOption( "expMatrixFile", OPT_STRING, "", "Basename for file(s) holding expected matrices" );
    opts.addOption( "saveExpMatrixFile", OPT_STRING, "", "Basename for output file(s) that will hold expected matrices" );

    opts.addOption( "warmupPasses", OPT_INT, "1", "Number of warmup passes to do before starting timings", 'w' );


#if defined(PARALLEL)
    MPI2DGridProgramBase::AddOptions( opts );
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
        throw InvalidArgValue( "size must have two dimensions" );
    }
    if( (arrayDims[0] < 0) || (arrayDims[1] < 0) )
    {
        throw InvalidArgValue( "all size values must be positive" );
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



