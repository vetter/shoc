#ifdef PARALLEL
#include <mpi.h>
#include <NodeInfo.h>
#endif
#include <stdlib.h>
#include <string.h>
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Stability.h"

#include "support.h"

#define CHECKS 10
#define ITERS_PER_CHECK 10

#if 0
struct float2 {
    float x;
    float y;
};
#endif


// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  The user is allowed to specify
//   the size of the input data in megabytes.
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Collin McCurdy
// Creation: September 08, 2009
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op) {

    op.addOption("time", OPT_INT, "1",
        "specify running time in miuntes", 't');
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Runs the stablity test. The algorithm for the parallel
//   version of the test, which enables testing of an entire GPU
//   cluster at the same time, is as follows. Each participating node
//   first allocates its data, while node zero additionally determines
//   start and finish times based on a user input parameter. All nodes
//   then enter the outermost loop, copying fresh data from the CPU
//   before entering the core of the test. In the core, each node
//   performs a loop consisting of the forward kernel, a potential
//   check, and then the inverse kernel. After performing a configurable
//   number of forward/inverse iterations, along with a configurable
//   number of checks, each node sends the number of failures it
//   encountered to node zero. Node zero collects and reports the error
//   counts, determines whether the test has run its course, and
//   broadcasts the decision. If the decision is to proceed, each node
//   begins the next iteration of the outer loop, copying fresh data and
//   then performing the kernels and checks of the core loop.
//
// Arguments:
//   resultDB: the benchmark stores its results in this ResultDatabase
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Collin McCurdy
// Creation: September 08, 2009
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser& op)
{
    int mpi_rank, mpi_size, node_rank;
    int i, j;
    float2* source, * result;
    void* work, * chk;

#ifdef PARALLEL
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    NodeInfo NI;
    node_rank = NI.nodeRank();

    cout << "MPI Task " << mpi_rank << " of " << mpi_size
         << " (noderank=" << node_rank << ") starting....\n";
#else
    mpi_rank = 0;
    mpi_size = 1;
    node_rank = 0;
#endif

    // ensure chk buffer alloc succeeds before grabbing the
    // rest of available memory.
    allocDeviceBuffer(&chk, 1);
    unsigned long avail_bytes = findAvailBytes();
    // unsigned long avail_bytes = 1024*1024*1024-1;

    // now determine how much available memory will be used (subject
    // to CUDA's constraint on the maximum block dimension size)
    int blocks = avail_bytes / (512*sizeof(float2));
    int slices = 1;
    while (blocks/slices > 65535) {
        slices *= 2;
    }
    int half_n_ffts = ((blocks/slices)*slices)/2;
    int n_ffts = half_n_ffts * 2;
    fprintf(stderr, "avail_bytes=%ld, blocks=%d, n_ffts=%d\n",
            avail_bytes, blocks, n_ffts);

    int half_n_cmplx = half_n_ffts * 512;
    unsigned long used_bytes = half_n_cmplx * 2 * sizeof(float2);

    cout << mpi_rank << ": testing "
         << used_bytes/((double)1024*1024) << " MBs\n";

    // allocate host memory
    source = (float2*)malloc(used_bytes);
    result = (float2*)malloc(used_bytes);

    // alloc device memory
    allocDeviceBuffer(&work, used_bytes);

    // alloc gather buffer
    int* recvbuf = (int*)malloc(mpi_size*sizeof(int));

    // compute start and finish times
    time_t start = time(NULL);
    time_t finish = start + (time_t)(op.getOptionInt("time")*60);
    struct tm start_tm, finish_tm;
    localtime_r(&start, &start_tm);
    localtime_r(&finish, &finish_tm);
    if (mpi_rank == 0) {
        printf("start = %s", asctime(&start_tm));
        printf("finish = %s", asctime(&finish_tm));
    }

    for (int iter = 0; ; iter++) {
        bool failed = false;
        int errorCount = 0, stop = 0;

        // (re-)init host memory...
        for (i = 0; i < half_n_cmplx; i++) {
            source[i].x = (rand()/(float)RAND_MAX)*2-1;
            source[i].y = (rand()/(float)RAND_MAX)*2-1;
            source[i+half_n_cmplx].x = source[i].x;
            source[i+half_n_cmplx].y = source[i].y;
        }

        // copy to device
        copyToDevice(work, source, used_bytes);
        copyToDevice(chk, &errorCount, 1);

        forward(work, n_ffts);
        if (check(work, chk, half_n_ffts, half_n_cmplx)) {
            fprintf(stderr, "First check failed...");
            failed = true;
        }

        if (!failed) {
            for (i = 1; i <= CHECKS; i++) {
                for (j = 1; j <= ITERS_PER_CHECK; j++) {
                    inverse(work, n_ffts);
                    forward(work, n_ffts);
                }

                if (check(work, chk, half_n_ffts, half_n_cmplx)) {
                    failed = true;
                    break;
                }
            }
        }

        // failing node is responsible for verifying failure, counting
        // errors and reporting count to node 0.
        if (failed) {
            fprintf(stderr, "Failure on node %d, iter %d:", mpi_rank, iter);

            // repeat check on CPU
            copyFromDevice(result, work, used_bytes);
            float2* result2 = result + half_n_cmplx;
            for (j = 0; j < half_n_cmplx; j++) {
                if (result[j].x != result2[j].x ||
                    result[j].y != result2[j].y)
                {
                    errorCount++;
                }
            }
            if (!errorCount) {
                fprintf(stderr, "verification failed!\n");
            }
            else {
                fprintf(stderr, "%d errors\n", errorCount);
            }
        }

#ifdef PARALLEL
        MPI_Gather(&errorCount, 1, MPI_INT, recvbuf, 1, MPI_INT,
                   0, MPI_COMM_WORLD);
#else
        recvbuf[0] = errorCount;
#endif

        // node 0 collects and reports error counts, determines
        // whether test has run its course, and broadcasts decision
        if (mpi_rank == 0) {
            time_t curtime = time(NULL);
            struct tm curtm;
            localtime_r(&curtime, &curtm);
            fprintf(stderr, "iter=%d: %s", iter, asctime(&curtm));

            for (i = 0; i < mpi_size; i++) {
                if (recvbuf[i]) {
                    fprintf(stderr, "--> %d failures on node %d\n", recvbuf[i], i);
                }
            }

            if (curtime > finish) {
                stop = 1;
            }
        }

#ifdef PARALLEL
        MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

        resultDB.AddResult("Check", "", "Failures", errorCount);
        if (stop) break;
    }

    freeDeviceBuffer(work);
    freeDeviceBuffer(chk);

    free(source);
    free(result);
    free(recvbuf);
}
