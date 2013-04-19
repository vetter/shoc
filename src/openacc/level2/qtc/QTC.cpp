#include <iostream>
#include <sstream>
#include <fstream>
    
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "QTCFuncs.h"
#include "QTC/libdata.h"

void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("threshold", OPT_FLOAT, "1", "cluster diameter threshold");
    op.addOption("compact", OPT_BOOL, "0", "use compact storage distance matrix (default 0)");
    op.addOption("seed", OPT_INT, "-1", "seed for random number generator");
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

    void (*qtcfunc)( T* points,
                    unsigned int numPoints,
                    T threshold,
                    double* clusteringTime,
                    double* totalTime );
    if( sizeof(T) == sizeof(float) )
    {
        qtcfunc = DoFloatQTC;
    }
    else
    {
        // our assumption was wrong - T is not a double or a float
        std::cerr << "unsupported type in RunTest; ignoring" << std::endl;
        return;
    }

    //
    // determine the parameters of the problem we will solve
    int stdPointCounts[4] = { 4, 8, 16, 26 };   // in "kilo-points"
    int numPoints = stdPointCounts[opts.getOptionInt("size") - 1] * 1024;
    float threshold = opts.getOptionFloat("threshold");
    bool verbose = opts.getOptionBool("verbose");
    unsigned int nPasses = (unsigned int)opts.getOptionInt("passes");

    // determine whether to use full or compact storage layout
    // TODO - unlike CUDA, where we can check the size of the memory
    // on the GPU, we don't have a way to check those device properties,
    // so we can't automatically determine whether to use full or compact.
    // Instead, let user tell us with a command line switch.
    bool useCompactLayout = opts.getOptionBool("compact");

    // determine if we were given a seed for the random number generator
    // NOTE: we currently use rand() as a generator in the QTC benchmarks,
    // mainly for convenience and portability.  The quality of the RNG
    // isn't as important for these benchmarks as for, say, a Monte Carlo
    // application.
    int rngSeed = opts.getOptionInt("seed");
    if( rngSeed != -1 )
    {
        // the user provided a seed - use it
        srand( (unsigned int)rngSeed );
    }
    else
    {
        // the user did not provide a seed - don't seed
    }

    // initialize the input
    // NOTE: this function supports float-type data only.
    float* dist_source = NULL;
    int* indr_mtrx = NULL;
    int max_degree;
    float* points = generate_synthetic_data( &dist_source,  // rslt_mtrx
                                                &indr_mtrx,    //indr_mtrx
                                                &max_degree,    // max degree
                                                threshold,
                                                numPoints,
                                                !useCompactLayout );
    std::cout << "point_count: " << numPoints 
        << ", max_degree: " << max_degree
        << ", threshold: " << threshold
        << std::endl;

#if READY
#else
    {
        std::ofstream ofs;
        ofs.open( "bogus.out" );
        for( unsigned int i = 0; i < numPoints; i++ )
        {
            ofs << points[2*i] << '\t' << points[2*i+1] << std::endl;
        }
        ofs.close();
    }
#endif // READY

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
        std::ostringstream sstr;
        sstr << numPoints << "pt";
        resultDB.AddResult( testName + "_clustering", 
                                sstr.str(), 
                                "s", 
                                clusteringTime );
        resultDB.AddResult( testName + "_total", 
                                sstr.str(), 
                                "s", 
                                totalTime );
    }

    // clean up
    delete[] points;
}



void
RunBenchmark(ResultDatabase &resultDB, OptionParser &opts)
{
    RunTest<float>( "QTC", resultDB, opts );

    // no support yet for double precision
    // TODO: is there a way in OpenACC to have it tell us
    // whether the device being used supports double precision?
}


