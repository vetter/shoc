#include <string.h>
#include <iomanip>

// When using MPICH and MPICH-derived MPI implementations, there is a
// naming conflict between stdio.h and MPI's C++ binding.
// Since we do not use the C++ MPI binding, we can avoid the ordering
// issue by ignoring the C++ MPI binding headers.
// This #define should be quietly ignored when using other MPI implementations.
#define MPICH_SKIP_MPICXX
#include "mpi.h"

#include <stdio.h>
#include <stdlib.h>
/*begin for sysv*/
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
/*end for sysv*/
#include "NodeInfo.h"
#include "RandomPairs.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include <ParallelResultDatabase.h>
#include <ParallelHelpers.h>

using namespace std;

int GPUDriver(OptionParser &op, ResultDatabase &resultDB);
int GPUSetup(OptionParser &op, int mympirank, int mynoderank);
int GPUCleanup(OptionParser &op);

// ****************************************************************************
// Function: DumpInSequence
//
// Purpose:
//   dumps results from GPU and MPI groups in sequence
//
// Arguments:
//   padb: the parallel results data base
//   mygrprank: group rank
//   mympirank: mpirank
//
// Returns:  nothing
//
// Creation: July 08, 2009
//
// Modifications:
//    Jeremy Meredith, Wed Nov 10 14:20:47 EST 2010
//    Split timing reports into detailed and summary.  For serial code, we
//    report all trial values, and for parallel, skip the per-process vals.
//
//    Jeremy Meredith, Fri Dec  3 16:30:31 EST 2010
//    Use the "passes" argument instead of hardcoding to 4 passes, and
//    increase the default to 10.  Changed the way collection of
//    the summary results worked to extract only the desired results.
//    Added reporting of GPU download latency.
//
// ****************************************************************************
void DumpInSequence(ParallelResultDatabase &padb, int mygrprank, int mympirank)
{
    flush(cout);
    MPI_Barrier(MPI_COMM_WORLD);
    // lets have group1 dump GPU results first
    if (mygrprank==0 && mympirank==0)
        padb.DumpSummary(cout);
    MPI_Barrier(MPI_COMM_WORLD);
    // now let group2 dump the MPI results
    if (mygrprank==0 && mympirank!=0)
        padb.DumpSummary(cout);
    flush(cout);
}

void MPITest(OptionParser &op, ResultDatabase &resultDB, int numtasks, int myrank,
                int mypair, MPI_Comm newcomm);

int main(int argc, char *argv[])
{
    int numdev=0, totalnumdev=0, numtasks, mympirank, dest, source, rc,
        mypair=0, count, tag=2, mynoderank,myclusterrank,nodenprocs;
    int *grp1, *grp2;
    int mygrprank,grpnumtasks;
    MPI_Group orig_group,bmgrp;
    MPI_Comm bmcomm,nlrcomm;
    ResultDatabase resultDB,resultDBWU,resultDB1;
    OptionParser op;
    ParallelResultDatabase pardb, pardb1;
    bool amGPUTask = false;
    volatile unsigned long long *mpidone;
    int i,shmid;

    /* Allocate System V shared memory */

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &mympirank);
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);


    //Add shared options to the parser
    op.addOption("device", OPT_VECINT, "0", "specify device(s) to run on",
		    'd');
    op.addOption("verbose", OPT_BOOL, "", "enable verbose output", 'v');
    op.addOption("quiet", OPT_BOOL, "",
		    "write minimum necessary to standard output", 'q');
    op.addOption("passes", OPT_INT, "10", "specify number of passes", 'z');
    op.addOption("size", OPT_VECINT, "1", "specify problem size", 's');
    op.addOption("time", OPT_INT, "5", "specify running time in miuntes", 't');
    op.addOption("outputFile", OPT_STRING, "output.txt", "specify output file",
       'o');
    op.addOption("infoDevices", OPT_BOOL, "", "show summary info for available devices",
       'i');
    op.addOption("fullInfoDevices", OPT_BOOL, "", "show full info for available devices");
    op.addOption("MPIminmsg", OPT_INT, "0", "specify minimum MPI message size");
    op.addOption("MPImaxmsg", OPT_INT, "16384",
                    "specify maximum MPI message size");
    op.addOption("MPIiter", OPT_INT, "1000",
                    "specify number of MPI benchmark iterations for each size");
    op.addOption("platform", OPT_INT, "0", "specify platform for device selection", 'y');

    if (!op.parse(argc, argv))
    {
        if (mympirank == 0)
            op.usage();
        MPI_Finalize();
        return 0;
    }

    int npasses = op.getOptionInt("passes");

    //our simple mapping
    NodeInfo NI;
    mynoderank = NI.nodeRank();         // rank of my process within the node
    myclusterrank = NI.clusterRank();   // cluster (essentially, node) id
    MPI_Comm smpcomm = NI.getSMPComm();

    if(mynoderank==0){
        shmid = shmget(IPC_PRIVATE,
                 sizeof(unsigned long long),
                 (IPC_CREAT | 0600));
    }

    MPI_Bcast(&shmid, 1, MPI_INT, 0, NI.getSMPComm());

    mpidone = ((volatile unsigned long long*) shmat(shmid, 0, 0));
    if (mynoderank == 0)
        shmctl(shmid, IPC_RMID, 0);
    *mpidone = 0;

    nlrcomm = NI.getNLRComm(); // communcator of all the lowest rank processes
                               // on all the nodes
    int numnodes = NI.numNodes();
    if ( numnodes%2!=0 )
    {
        if(mympirank==0)
            printf("\nThis test needs an even number of nodes\n");
        MPI_Finalize();
	exit(0);
    }
    int nodealr = NI.nodeALR();

    nodenprocs=NI.nodeNprocs();

    // determine how many GPU devices we are to use
    int devsPerNode = op.getOptionVecInt( "device" ).size();
    //cout<<mympirank<<":numgpus="<<devsPerNode<<endl;

    // if there are as many or more devices as the nprocs, only use half of
    // the nproc
    if ( devsPerNode >= nodenprocs ) devsPerNode = nodenprocs/2;

    numdev = (mynoderank == 0) ? devsPerNode : 0;
    MPI_Allreduce(&numdev, &totalnumdev, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
    numdev = devsPerNode;

    // determine whether I am to be a GPU or a comm task
    if( mynoderank < numdev )
    {
        amGPUTask = true;
    }

    //Divide tasks into two distinct groups based upon noderank
    grp1=(int *)calloc(totalnumdev, sizeof(int));
    grp2=(int *)calloc((numtasks-totalnumdev),sizeof(int));
    if (grp1==NULL || grp2==NULL)
    {
        printf("\n%d:calloc failed in %s",mympirank,__FUNCTION__);
        exit(1);
    }


    /*compute the groups*/
    int beginoffset[2]={0,0};
    if(mynoderank == 0)
    {
        int tmp[2];
	tmp[0]=numdev;
	tmp[1]=nodenprocs-numdev;
        if (mympirank ==0)
            MPI_Send(tmp, 2*sizeof(int), MPI_CHAR, 1, 112, nlrcomm);
        else
        {
            MPI_Status reqstat;
	    MPI_Recv(beginoffset, 2*sizeof(int), MPI_CHAR, myclusterrank-1,
			    112, nlrcomm ,&reqstat);
            if (myclusterrank < numnodes-1)
            {
                beginoffset[0]+=numdev;
                beginoffset[1]+=(nodenprocs-numdev);
		MPI_Send(beginoffset,2*sizeof(int), MPI_CHAR, myclusterrank+1,
				112, nlrcomm);
		beginoffset[0]-=numdev;
		beginoffset[1]-=(nodenprocs-numdev);
            }
        }
    }
    MPI_Bcast(beginoffset,2,MPI_INT,0,smpcomm);

    if ( amGPUTask )
    {
        // I am to do GPU work
        grp1[beginoffset[0]+mynoderank]=mympirank;
        grpnumtasks=totalnumdev;
    }
    else
    {
        // I am to do MPI communication work
        grp2[beginoffset[1]+(mynoderank-numdev)]=mympirank;
        grpnumtasks=numtasks-totalnumdev;
    }

    MPI_Allreduce(MPI_IN_PLACE, grp1, totalnumdev, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, grp2, (numtasks-totalnumdev), MPI_INT,
                            MPI_SUM, MPI_COMM_WORLD);

    if ( amGPUTask )
    {
        // I am to do GPU work, so will be part of GPU communicator
        MPI_Group_incl(orig_group, totalnumdev, grp1, &bmgrp);
    }
    else
    {
        // I am to do MPI communication work, so will be part of MPI
        // messaging traffic communicator
        MPI_Group_incl(orig_group, (numtasks-totalnumdev), grp2,
                        &bmgrp);
    }

    MPI_Comm_create(MPI_COMM_WORLD, bmgrp, &bmcomm);
    MPI_Comm_rank(bmcomm, &mygrprank);
    NodeInfo *GRPNI = new NodeInfo(bmcomm);
    int mygrpnoderank=GRPNI->nodeRank();
    int grpnodealr = GRPNI->nodeALR();
    int grpnodenprocs = GRPNI->nodeNprocs();
    MPI_Comm grpnlrcomm = GRPNI->getNLRComm();
    //note that clusterrank and number of nodes don't change for this child
    //group/comm


    //form node-random pairs (see README) among communication tasks
    if( amGPUTask )
    {
        //setup GPU in GPU tasks
        GPUSetup(op, mympirank, mynoderank);
    }
    else
    {
        int * pairlist = new int[numnodes];
        for (i=0;i<numnodes;i++) pairlist[i]=0;

        if ( mygrpnoderank==0 )
        {
            pairlist[myclusterrank]=grpnodealr;
            MPI_Allreduce(MPI_IN_PLACE,pairlist,numnodes,MPI_INT,MPI_SUM,
                          grpnlrcomm);
            mypair = RandomPairs(myclusterrank, numnodes, grpnlrcomm);
            mypair = pairlist[mypair];
        }
        for (i=0;i<numnodes;i++) pairlist[i]=0;
        if ( mygrpnoderank==0 )
            pairlist[myclusterrank]=mypair;
        MPI_Allreduce(MPI_IN_PLACE,pairlist,numnodes,MPI_INT,MPI_SUM,
                      bmcomm);
        mypair = pairlist[myclusterrank]+mygrpnoderank;
    }

    // ensure we are all synchronized before starting test
    MPI_Barrier(MPI_COMM_WORLD);

    //warmup run
    if ( amGPUTask )
    {
        GPUDriver(op, resultDBWU);
    }
    //first, individual runs for device benchmark
    for(i=0;i<npasses;i++){
        if ( amGPUTask )
        {
            GPUDriver(op, resultDB);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //warmup run
    if ( !amGPUTask )
    {
        MPITest(op, resultDBWU, grpnumtasks, mygrprank, mypair, bmcomm);
    }
    //next, individual run for MPI Benchmark
    for(i=0;i<npasses;i++){
        if ( !amGPUTask )
        {
            MPITest(op, resultDB, grpnumtasks, mygrprank, mypair, bmcomm);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //merge and print
    pardb.MergeSerialDatabases(resultDB, bmcomm);
    if (mympirank==0)
        cout<<endl<<"*****************************Sequential GPU and MPI runs****************************"<<endl;
    DumpInSequence(pardb, mygrprank, mympirank);

    // Simultaneous runs for observing impact of contention
    MPI_Barrier(MPI_COMM_WORLD);
    if ( amGPUTask )
    {
        do {
            if (mympirank == 0 ) cout<<".";
            GPUDriver(op, resultDB1);flush(cout);
        } while(*mpidone==0);
        if (mympirank == 0 ) cout<<"*"<<endl;
    }
    else
    {
        for ( i=0;i<npasses;i++ )
        {
            MPITest(op, resultDB1, grpnumtasks, mygrprank, mypair, bmcomm);
        }
        *mpidone=1;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //merge and print
    pardb1.MergeSerialDatabases(resultDB1,bmcomm);
    if (mympirank==0)
        cout<<endl<<"*****************************Simultaneous GPU and MPI runs****************************"<<endl;
    DumpInSequence(pardb1, mygrprank, mympirank);

    //print summary
    if ( !amGPUTask && mygrprank==0)
    {
        vector<ResultDatabase::Result> prelatency  = pardb.GetResultsForTest("MPI Latency(mean)");
        vector<ResultDatabase::Result> postlatency = pardb1.GetResultsForTest("MPI Latency(mean)");
        cout<<endl<<"Summarized Mean(Mean) MPI Baseline Latency vs. Latency with Contention";
        cout<<endl<<"MSG SIZE(B)\t";
        int msgsize=0;
        for (i=0; i<prelatency.size(); i++)
        {
            cout<<msgsize<<"\t";
            msgsize = (msgsize ? msgsize * 2 : msgsize + 1);
        }

        cout << endl <<"BASELATENCY\t";
        for (i=0; i<prelatency.size(); i++)
            cout<<setiosflags(ios::fixed) << setprecision(2)<<prelatency[i].GetMean() << "\t";

        cout << endl <<"CONTLATENCY\t";
        for (i=0; i<postlatency.size(); i++)
            cout<<setiosflags(ios::fixed) << setprecision(2)<<postlatency[i].GetMean() << "\t";
        flush(cout);
        cout<<endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if ( amGPUTask && mympirank==0)
    {
        vector<ResultDatabase::Result> prespeed  = pardb.GetResultsForTest("DownloadSpeed(mean)");
        vector<ResultDatabase::Result> postspeed = pardb1.GetResultsForTest("DownloadSpeed(mean)");
        cout<<endl<<"Summarized Mean(Mean) GPU Baseline Download Speed vs. Download Speed with Contention";
        cout<<endl<<"MSG SIZE(KB)\t";
        int msgsize=1;
        for (i=0; i<prespeed.size(); ++i)
        {
            cout<<msgsize<<"\t";
            msgsize = (msgsize ? msgsize * 2 : msgsize + 1);
        }
        cout << endl <<"BASESPEED\t";
        for (i=0; i<prespeed.size(); ++i)
            cout<<setiosflags(ios::fixed) << setprecision(4)<<prespeed[i].GetMean() << "\t";

        cout << endl <<"CONTSPEED\t";
        for (i=0; i<postspeed.size(); ++i)
            cout<<setiosflags(ios::fixed) << setprecision(4)<<postspeed[i].GetMean() << "\t";
         cout<<endl;
    }

    if ( amGPUTask && mympirank==0)
    {
        vector<ResultDatabase::Result> pregpulat  = pardb.GetResultsForTest("DownloadLatencyEstimate(mean)");
        vector<ResultDatabase::Result> postgpulat = pardb1.GetResultsForTest("DownloadLatencyEstimate(mean)");
        cout<<endl<<"Summarized Mean(Mean) GPU Baseline Download Latency vs. Download Latency with Contention";
        cout<<endl<<"MSG SIZE\t";
        for (i=0; i<pregpulat.size(); ++i)
        {
            cout<<pregpulat[i].atts<<"\t";
        }
        cout << endl <<"BASEGPULAT\t";
        for (i=0; i<pregpulat.size(); ++i)
            cout<<setiosflags(ios::fixed) << setprecision(7)<<pregpulat[i].GetMean() << "\t";

        cout << endl <<"CONTGPULAT\t";
        for (i=0; i<postgpulat.size(); ++i)
            cout<<setiosflags(ios::fixed) << setprecision(7)<<postgpulat[i].GetMean() << "\t";
         cout<<endl;
    }
    //cleanup GPU
    if( amGPUTask )
    {
        GPUCleanup(op);
    }

    MPI_Finalize();

}
