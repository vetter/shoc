#ifndef NODE_INFO_H
#define NODE_INFO_H
#include "config.h"
#include "mpi.h"
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <cassert>

#if HAVE_UNISTD_H
#include <unistd.h>  // for gethostname; TODO make this more portable
#endif // HAVE_UNISTD_H

using namespace std;

#define MAX_HOSTNAME 80
#define HOSTNAME_LEN 64

// ****************************************************************************
// Class:  NodeInfo
//
// Purpose:
//   Provides summary of information on tasks in a node in an MPI communicator.
//
// Programmer:  Vinod Tipparaju
// Creation:    August 17, 2009
//
// ****************************************************************************
class NodeInfo
{
    int nodenprocs;
    int nodealr;
    int clusterrank;
    int numnodes;
    int noderank;
    string nodename;

    MPI_Comm NIComm;
    MPI_Comm LRComm;
    MPI_Comm SMPComm;
    MPI_Group NIGroup;
    MPI_Group LRGroup;

  public:
    NodeInfo(MPI_Comm comm=MPI_COMM_WORLD);
    int nodeNprocs() {return nodenprocs;}
    int numNodes() {return numnodes;}
    int nodeRank() {return noderank;}
    int nodeALR() {return nodealr;}
    int clusterRank() {return clusterrank;}
    string nodeName() {return nodename;}
    MPI_Comm getNLRComm() {return LRComm;}
    MPI_Comm getSMPComm() {return SMPComm;}
    void print();
};


inline
NodeInfo::NodeInfo(MPI_Comm comm)
{
    char **hostnamelist,*name;
    int rank, numprocs;
    int i,j,limit,rc,len,pwidth;
    int *alrlist;  /*store the least rank info*/
    NIComm = comm;
    MPI_Comm_size (comm, &numprocs);
    MPI_Comm_rank (comm, &rank);
    MPI_Comm_group(comm, &NIGroup);

    limit = MAX_HOSTNAME - 1;
    name=(char*)malloc(MAX_HOSTNAME*sizeof(char));
#if HAVE_GETHOSTNAME
    rc=gethostname(name,limit);
#else
#   error "No support (yet) for finding host name on this platform"
#endif
    len = strlen(name)+1;
    *(name+len) = '\0';
    nodename.assign(name);

    /*Rank 0 collects all the node names*/
    if(rank){
        rc = MPI_Send(name,len, MPI_CHAR, 0, 1, comm);
    }
    else{
       /*create an array to store all the host names for parsing*/
       hostnamelist=(char **)malloc(sizeof(char *)*numprocs);
       for(i=0;i<numprocs;i++)
           hostnamelist[i]=(char *)malloc(sizeof(char)*MAX_HOSTNAME);

       /*copy my name in 0'th position*/
       strcpy(hostnamelist[0],name);

       /*get the names from all others*/
       MPI_Status stat;
       for(i=1;i<numprocs;i++){
           rc = MPI_Recv(hostnamelist[i],MAX_HOSTNAME, MPI_CHAR, i, 1, comm,&stat);
       }
    }
    /*allocate and initialize alrlist and */
    alrlist=(int *)calloc(numprocs, sizeof(int));
    int *nodelist = (int *)calloc(numprocs,sizeof(int));
    if(alrlist==NULL||nodelist==NULL)
    {
        cout<<"Malloc failed for SMP/MAP info"<<endl;exit(1);
    }

    /*populate the smp and node info*/
    if(rank == 0){
        numnodes = 0;
        char *b_name = (char *)malloc(sizeof(char)*MAX_HOSTNAME);
        for(i=0;i<numprocs;i++){
        int k=i;
            if(hostnamelist[i][0] != '\0'){
                alrlist[numnodes] = i;
                nodelist[i] = numnodes;
                strcpy(b_name, hostnamelist[i]);
                hostnamelist[i][0] = '\0';
                while(k < numprocs-1){
                    if(!strcmp(b_name,hostnamelist[k+1])){
                        hostnamelist[k+1][0]='\0';
                        nodelist[k+1]=numnodes;
                    }
                    k++;
                }
                numnodes++;
            }
        }
        free(b_name);
    }

    /*broadcast numnodes to everyone*/
    MPI_Bcast(&numnodes, 1, MPI_INT, 0, NIComm);
    MPI_Bcast(nodelist, numprocs, MPI_INT, 0, NIComm);

    int color = nodelist[rank];
    MPI_Group smpgroup;
    MPI_Comm_split(comm,color,rank,&SMPComm);  /*create intranode communicator*/
    MPI_Comm_rank(SMPComm, &noderank);
    MPI_Comm_size(SMPComm, &nodenprocs);
    MPI_Comm_group(SMPComm,&smpgroup);
    color=0;
    MPI_Group_translate_ranks (smpgroup,1,&color,NIGroup,&nodealr);

    /*send the list of ranks for the inter-node communicator*/
    MPI_Bcast(alrlist, numnodes, MPI_INT, 0, NIComm);
    MPI_Group_incl(NIGroup,numnodes,alrlist,&LRGroup);
    MPI_Comm_create(comm,LRGroup,&LRComm);
    if ( noderank == 0 )
        MPI_Comm_rank(LRComm,&clusterrank);
    MPI_Bcast(&clusterrank,1,MPI_INT,0,SMPComm);
    //print();
    MPI_Barrier(comm);

}


inline
void NodeInfo::print()
{
    if (noderank == 0)
    {
        cout << "Printing Node Information" << endl;
        cout << "nodeNprocs = " << nodenprocs<<endl;
        cout << "numNodes   = " << numnodes<<endl;
        cout << "nodeRank   = " << noderank<<endl;
        cout << "clusterRank   = " << clusterrank<<endl;
        cout << "nodeALR    = " << nodealr<<endl;
        cout << "nodeName   = " << nodename<<endl;
        cout << "-------------------------" << endl;
    }
}


#if 0
int main(int argc,char *argv[])
{
    int numtasks, rank, dest, source, rc, count, tag=1, noderank;
    char inmsg, outmsg='x';
    MPI_Init(&argc,&argv);
    NodeInfo NI;
    NI.print();
    MPI_Finalize();
}
#endif

#endif
