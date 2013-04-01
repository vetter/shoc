#include <iostream>
    
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "QTCFuncs.h"

void
addBenchmarkSpecOptions(OptionParser &op)
{
   ;
}


template<class T>
void
RunTest( const std::string& testName,
            ResultDatabase& resultDB,
            OptionParser& opts )
{
    // As of March 2013, few available compilers with OpenACC support
    // support OpenACC from C++ programs.  Instead, we call out to
    // C routines with the OpenACC directives, but we leave the benchmark
    // skeleton in C++ so we can reuse classes like the ResultsDatabase
    // and OptionParser.
    // 
    // Once compilers start supporting C++, the separate C function with
    // OpenACC directives can be inlined into this templatized function.
    //
    // Determine which function we will use, based on type of T.
    // We assume that we will only be called with doubles and floats.
    // Note that our test for type of T is nowhere close to bullet proof -
    // for example, it would recognize T=uint64_t as a 64-bit double.
    // Also note that the signature of our C function has to take the
    // data as a void* since it must handle both types.
    // Likewise, our scan functions return via an argument rather than
    // a return value, so that they can have the correct type for the 
    // output variable.

    void (*qtcfunc)( void );
    if( sizeof(T) == sizeof(double) )
    {
        qtcfunc = DoFloatQTC;
    }
    else if( sizeof(T) == sizeof(float) )
    {
        qtcfunc = DoDoubleQTC;
    }
    else
    {
        // our assumption was wrong - T is not a double or a float
        std::cerr << "unsupported type in RunTest; ignoring" << std::endl;
        return;
    }

    //
    // determine the parameters of the problem we will solve
    int stdPointCounts[4] = { 1, 8, 16, 26 };   // in "kilo-points"
    int numPoints = stdPointCounts[opts.getOptionInt("size") - 1] * 1024;
    float threshold = opts.getOptionFloat("threshold");
    bool verbose = opts.getOptionBool("verbose");

#if READY
    // determine whether to use full or compact storage layout
#else
    // assume we are using full layout
    bool useFullLayout = true;
#endif // READY

    // initialize the input
    float* points = generate_synthetic_data( &dist_source,
                                                &indr_mtrx_host,
                                                &max_degree,
                                                threshold,
                                                numPoints,
                                                useFullLayout );

    // run the benchmark passes

    for( unsigned int pass = 0; pass < nPasses; pass++ )
    {
        double totalTime;       // in seconds
        double clusteringTime;  // in seconds

        // run the benchmark
        (*qtcfunc)( points,
                        numPoints,
                        threshold,
                        &clusteringTime,
                        &totalTime );
        
#if READY
        // verify the result (?)
        // is there any reasonable way to do this?
#endif // READY

        // record the results
        resultDB.AddResult( testName + "_total", sizeStr, "s", totalTime );
        resultDB.AddResult( testName + "_clustering", sizeStr, "s", clusteringTime );
    }

    // clean up
    delete[] points;
}



void
RunBenchmark(ResultDatabase &resultDB, OptionParser &opts)
{
    RunTest<float>( "QTC", resultDB, opts );

#if READY
    // TODO - is there a way in OpenACC to test whether the chosen
    // device supports double precision arithmetic?
    RunTest<double>( "QTC", resultDB, opts );
#endif // READY
}

