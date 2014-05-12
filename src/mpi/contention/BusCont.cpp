#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "ResultDatabase.h"
#include "OptionParser.h"

using namespace std;

// ****************************************************************************
// Function: MPITest
//
// Purpose:
//   dumps results from GPU and MPI groups in sequence
//
// Arguments:
//   op: benchmark options
//   resultDB: the parallel results data base
//   numtasks: total tasks that will run the benchmark
//   myrank: my rank
//   mypair: my pair to communicate to
//   newcomm: the context for the ranks
//
// Returns:
//
// Creation: July 08, 2009
//
// Modifications:
//
// ****************************************************************************
void MPITest(OptionParser &op, ResultDatabase &resultDB, int numtasks, int myrank,
                int mypair, MPI_Comm newcomm)
{
    int msgsize;
    int skip=10,i=0,j=0;
    int minmsg_sz = op.getOptionInt("MPIminmsg");
    int maxmsg_sz = op.getOptionInt("MPImaxmsg");
    int iterations = op.getOptionInt("MPIiter");
    char *recvbuf,*sendbuf, *sendptr, *recvptr;
    char sizeStr[256];
    double minlat=0, maxlat=0, avglat=0, latency, t_start, t_end;
    MPI_Status reqstat;
    MPI_Request req;


    recvbuf = (char *) malloc(maxmsg_sz*16);
    sendbuf = (char *) malloc(maxmsg_sz*16);
    if (recvbuf==NULL || sendbuf == NULL)
    {
        printf("\n%d:memory allocation in %s failed",myrank,__FUNCTION__);
        fflush(stdout);
        exit(1);
    }

    for (msgsize = minmsg_sz; msgsize <= maxmsg_sz;
         msgsize = (msgsize ? msgsize * 2 : msgsize + 1))
    {

        MPI_Barrier(newcomm);

        if (myrank < mypair)
        {
            for (i = 0; i < iterations + skip; i++)
            {
                if (i == skip) t_start = MPI_Wtime();
                sendptr = sendbuf+msgsize*((j++)%16);
                recvptr = recvbuf+msgsize*((j++)%16);
                MPI_Send(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm);
                MPI_Recv(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm,
                                &reqstat);
            }
            t_end = MPI_Wtime();
        }
        else if (myrank > mypair)
        {
            for (i = 0; i < iterations + skip; i++)
            {
                if (i == skip) t_start = MPI_Wtime();
                sendptr = sendbuf+msgsize*((j++)%16);
                recvptr = recvbuf+msgsize*((j++)%16);
                MPI_Recv(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm,
                                &reqstat);
                MPI_Send(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm);
            }
            t_end = MPI_Wtime();
        }
        else
        {
            for (i = 0; i < iterations + skip; i++)
            {
                if (i == skip) t_start = MPI_Wtime();
                sendptr = sendbuf+msgsize*((j++)%16);
                recvptr = recvbuf+msgsize*((j++)%16);
                MPI_Irecv(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm,
                                &req);
                MPI_Send(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm);
                MPI_Wait(&req, &reqstat);
            }
            t_end = MPI_Wtime();
        }

        latency = (t_end - t_start) * 1e6 / (2.0 * iterations);
        sprintf(sizeStr, "% 6dkB", msgsize);
        resultDB.AddResult("MPI Latency", sizeStr, "MicroSeconds", latency);


        //MPI_Reduce(&latency,&maxlat,1,MPI_DOUBLE, MPI_MAX, 0, newcomm);
        //MPI_Reduce(&latency,&minlat,1,MPI_DOUBLE, MPI_MIN, 0, newcomm);
        //MPI_Reduce(&latency,&avglat,1,MPI_DOUBLE, MPI_SUM, 0, newcomm);
        //MPI_Comm_size(newcomm,&j);
        //avglat/=j;
        j=0;
        //if (myrank == 0)
        //{
        //    printf("\n%d\t%f\t%f\t%f",msgsize,minlat,avglat,maxlat);
        //    fflush(stdout);
        //}
    }

    free(recvbuf);
    free(sendbuf);
}
