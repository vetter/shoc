#include <string.h>
#include <iomanip>
#include <pthread.h>

// When using MPICH and MPICH-derived MPI implementations, there is a
// naming conflict between stdio.h and MPI's C++ binding.
// Since we do not use the C++ MPI binding, we can avoid the ordering
// issue by ignoring the C++ MPI binding headers.
// This #define should be quietly ignored when using other MPI implementations.
#define MPICH_SKIP_MPICXX
#include "mpi.h"

#include <stdio.h>
#include <stdlib.h>
#include "NodeInfo.h"
#include "RandomPairs.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include <ParallelResultDatabase.h>
#include <ParallelHelpers.h>

using namespace std;

int mympirank;
void GPUDriversim();
void GPUDriverwrmup();
void GPUDriverseq();
ResultDatabase &GPUGetseqrdb();
ResultDatabase &GPUGetsimrdb();
int GPUSetup(OptionParser &op, int mympirank, int mynoderank);
int GPUCleanup();
int numgputimes=1;
volatile unsigned long long mpidone=1;

void MPITest(OptionParser &op, ResultDatabase &resultDB, int numtasks, int myrank,
                int mypair, MPI_Comm newcomm);

void *GPUDriversimwrapper(void *dummy)
{
int i;
    if (mympirank == 0 ) cout<<endl;
    do {
        GPUDriversim();
        //if (mympirank == 0) {cout<<".";flush(cout);}
    }while(mpidone!=1);
    //if (mympirank == 0 ) cout<<endl;
    return (0);
}

pthread_t mychild;
void exitwrapper(const char* err_str, int err_val)
{
    printf("\n%s:%d",err_str,err_val);fflush(stdout);
    exit(1);
}

void spawnongpu ()
{
pthread_attr_t attr;
int rc;
    if((rc=pthread_attr_init(&attr)))
       exitwrapper("pthread attr_init failed",rc);

    rc = pthread_create(&mychild, &attr, GPUDriversimwrapper, (void *)NULL);
    if(rc) exitwrapper("pthread_create failed",rc);

    pthread_attr_destroy(&attr);
}

// Modifications:
//    Jeremy Meredith, Fri Dec  3 16:30:31 EST 2010
//    Use the "passes" argument instead of hardcoding to 4 passes, and
//    increase the default to 10.  Changed the way collection of
//    the summary results worked to extract only the desired results.
//    Added reporting of GPU download latency.
//
int main(int argc, char *argv[])
{
    int numdev=0, totalnumdev=0, numtasks, dest, source, rc,
        mypair=0, count, tag=2, mynoderank,myclusterrank,nodenprocs;
    int *grp1,i;
    int mygrprank,grpnumtasks;
    MPI_Group orig_group,new_group;
    MPI_Comm new_comm,nlrcomm;
    ResultDatabase rdbwupmpi, rdbseqmpi, rdbsimmpi;
    OptionParser op;
    ParallelResultDatabase prdbseqgpu, prdbsimgpu, prdbsimmpi, prdbseqmpi;
    bool amGPUTask = false;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &mympirank);
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

    //Add shared options to the parser
    op.addOption("device", OPT_VECINT, "0", "specify device(s) to run on", 'd');
    op.addOption("verbose", OPT_BOOL, "", "enable verbose output", 'v');
    op.addOption("quiet", OPT_BOOL, "", "write minimum necessary to standard output", 'q');
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
    nlrcomm = NI.getNLRComm();          // communcator of all the lowest rank processes on a node
    MPI_Comm smpcomm = NI.getSMPComm();
    int numnodes = NI.numNodes();

    if ( numnodes%2!=0 )
    {
        if(mympirank==0)
            cout<<"This test needs an even number of nodes"<<endl;
        MPI_Finalize();
        exit(0);
    }
    int nodealr = NI.nodeALR();

    // determine how many GPU devices we are to use
    int devsPerNode = op.getOptionVecInt( "device" ).size();
    nodenprocs = NI.nodeNprocs();
    if ( devsPerNode > nodenprocs)
        devsPerNode = nodenprocs;
    numdev = (mynoderank == 0) ? devsPerNode : 0;
    MPI_Allreduce(&numdev, &totalnumdev, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
    numdev = devsPerNode;

    // determine whether I am to be a GPU or a comm task
    if( mynoderank < numdev )
    {
        amGPUTask = true;
    }

    grp1=(int *)calloc(totalnumdev, sizeof(int));
    if (grp1==NULL)
    {
        printf("\n%d:calloc failed in %s",mympirank,__FUNCTION__);
        exit(1);
    }

    /*compute the groups*/
    int beginoffset=0;
    if(mynoderank == 0)
    {
        if (mympirank ==0)
            MPI_Send(&numdev, sizeof(int), MPI_CHAR, 1, 112, nlrcomm);
        else
        {
            MPI_Status reqstat;
	    MPI_Recv(&beginoffset, sizeof(int), MPI_CHAR, myclusterrank-1,
			    112, nlrcomm ,&reqstat);
	    if (myclusterrank < numnodes-1)
            {
                beginoffset+=numdev;
		MPI_Send(&beginoffset,sizeof(int), MPI_CHAR, myclusterrank+1,
				112, nlrcomm);
		beginoffset-=numdev;
            }
        }
    }
    MPI_Bcast(&beginoffset,1,MPI_INT,0,smpcomm);
    if ( amGPUTask )
    {
        // I am to do GPU work
        grp1[beginoffset+mynoderank]=mympirank;
        grpnumtasks=totalnumdev;
    }

    MPI_Allreduce(MPI_IN_PLACE, grp1, totalnumdev, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
    MPI_Group_incl(orig_group, totalnumdev, grp1, &new_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    if( amGPUTask )
    {
        //setup GPU in GPU tasks
        MPI_Comm_rank(new_comm, &mygrprank);
        GPUSetup(op, mympirank, mynoderank);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //form random pairs among communication tasks
    int * pairlist = new int[numnodes];
    for ( i=0; i<numnodes; i++ ) pairlist[i]=0;

    int *myinplist = new int[nodenprocs-1];
    if ( mynoderank==0 )
    {
        pairlist[myclusterrank]=mympirank;
        MPI_Allreduce(MPI_IN_PLACE,pairlist,numnodes,MPI_INT,MPI_SUM,
                        nlrcomm);
        mypair = RandomPairs(myclusterrank, numnodes, nlrcomm);
	      mypair = pairlist[mypair];

        int *myproclist = new int[nodenprocs-1];
        for (i=0;i<nodenprocs-1;i++)
            myinplist[i]=i+1;
        MPI_Group intragrp,nigrp;
        MPI_Comm_group(smpcomm,&intragrp);
        MPI_Comm_group(MPI_COMM_WORLD,&nigrp);
        MPI_Group_translate_ranks(intragrp,nodenprocs-1,myinplist,nigrp,myproclist);

        if ( mypair<mympirank )
        {
            MPI_Status reqstat;
	    MPI_Send(myproclist, (nodenprocs-1)*sizeof(int), MPI_CHAR, mypair,
			    112, MPI_COMM_WORLD);
	    MPI_Recv(myinplist, (nodenprocs-1)*sizeof(int), MPI_CHAR, mypair,
			    112, MPI_COMM_WORLD,&reqstat);
        }
        else
        {
            MPI_Status reqstat;
	    MPI_Recv(myinplist, (nodenprocs-1)*sizeof(int), MPI_CHAR, mypair,
			    112, MPI_COMM_WORLD,&reqstat);
	    MPI_Send(myproclist, (nodenprocs-1)*sizeof(int), MPI_CHAR, mypair,
			    112, MPI_COMM_WORLD);
        }
    }
    MPI_Bcast(myinplist,(nodenprocs-1),MPI_INT,0,smpcomm);
    if ( mynoderank!=0 )
        mypair = myinplist[mynoderank-1];

    // ensure we are all synchronized before starting test
    MPI_Barrier(MPI_COMM_WORLD);

    //first, individual runs for device benchmark
    if ( amGPUTask ) GPUDriverwrmup();
    if ( amGPUTask )
    {
        for ( i=0; i<npasses; i++ ) GPUDriverseq();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //First, warmup run
    MPITest(op, rdbwupmpi,numtasks,mympirank,mypair,MPI_COMM_WORLD );

    //next, individual run for MPI Benchmark
    for ( i=0; i<npasses; i++ )
        MPITest(op, rdbseqmpi, numtasks,mympirank,mypair,MPI_COMM_WORLD );

    MPI_Barrier(MPI_COMM_WORLD);

    //merge and print results
    if (mympirank==0)
        cout<<endl<<"*****************************Sequential GPU and MPI runs****************************"<<endl;
    if ( amGPUTask )
    {
        ResultDatabase &rdbseqgpu = GPUGetseqrdb();
        prdbseqgpu.MergeSerialDatabases(rdbseqgpu, new_comm);
        if(mygrprank==0)
            prdbseqgpu.DumpSummary(cout);
    }
    flush(cout);
    MPI_Barrier(MPI_COMM_WORLD);
    prdbseqmpi.MergeSerialDatabases(rdbseqmpi,MPI_COMM_WORLD);
    if(mympirank==0)
        prdbseqmpi.DumpSummary(cout);

    // Simultaneous runs for observing impact of contention
    MPI_Barrier(MPI_COMM_WORLD);

    //make sure GPU task keeps running until MPI task is done
    mpidone = 0;

    //first spawnoff GPU tasks
    if ( amGPUTask )
    {
        spawnongpu();
    }

    //run the MPI test
    for ( i=0;i<npasses;i++)
        MPITest(op, rdbsimmpi, numtasks,mympirank,mypair,MPI_COMM_WORLD );
    mpidone=1;

    //waif for MPI test to finish
    MPI_Barrier(MPI_COMM_WORLD);

    //wait for GPU tasks to return
    if ( amGPUTask )
    {
        if((rc=pthread_join(mychild,NULL)))
            exitwrapper("pthread_join: failed",rc);
    }

    //merge and print
    if (mympirank==0)
        cout<<endl<<"*****************************Simultaneous GPU and MPI runs****************************"<<endl;
    if ( amGPUTask )
    {
	ResultDatabase &rdbsimgpu = GPUGetsimrdb();
        prdbsimgpu.MergeSerialDatabases(rdbsimgpu, new_comm);
        if(mygrprank==0)
            prdbsimgpu.DumpSummary(cout);
    }
    flush(cout);
    MPI_Barrier(MPI_COMM_WORLD);
    prdbsimmpi.MergeSerialDatabases(rdbsimmpi,MPI_COMM_WORLD);
    if ( mympirank==0 )
        prdbsimmpi.DumpSummary(cout);
    flush(cout);

    //print summary
#if 1
    if (mympirank == 0)
    {
        vector<ResultDatabase::Result> prelatency  = prdbseqmpi.GetResultsForTest("MPI Latency(mean)");
        vector<ResultDatabase::Result> postlatency = prdbsimmpi.GetResultsForTest("MPI Latency(mean)");
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
        vector<ResultDatabase::Result> prespeed    = prdbseqgpu.GetResultsForTest("DownloadSpeed(mean)");
        vector<ResultDatabase::Result> postspeed   = prdbsimgpu.GetResultsForTest("DownloadSpeed(mean)");
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
        vector<ResultDatabase::Result> pregpulat  = prdbseqgpu.GetResultsForTest("DownloadLatencyEstimate(mean)");
        vector<ResultDatabase::Result> postgpulat = prdbsimgpu.GetResultsForTest("DownloadLatencyEstimate(mean)");
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
#endif
    //cleanup GPU
    if( amGPUTask )
    {
        GPUCleanup();
    }

    MPI_Finalize();
}
