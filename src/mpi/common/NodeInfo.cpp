#include "NodeInfo.h"
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <stdlib.h>

using namespace std;

int main(int argc,char *argv[])
{
    int numtasks, rank, dest, source, rc, count, tag=1, noderank;
    char inmsg, outmsg='x';
    MPI_Init(&argc,&argv);
    NodeInfo NI;
    NI.print();
    MPI_Finalize();
}
