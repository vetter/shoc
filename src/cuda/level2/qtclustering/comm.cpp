#include "comm.h"
#include <iostream>

using namespace std;

#if defined(PARALLEL)
MPI_Comm _qtc_mpi_communicator = MPI_COMM_WORLD;
#endif

int comm_get_rank(void){
    int rank=0;
#if defined(PARALLEL)
    MPI_Comm_rank( _qtc_mpi_communicator, &rank );
#endif
    return rank;
}

int comm_get_size(void){
    int node_count=1;
#if defined(PARALLEL)
    MPI_Comm_size( _qtc_mpi_communicator, &node_count );
#endif // defined(PARALLEL)
    return node_count;
}


void comm_broadcast( void *ptr, int cnt, int type, int source){
#if defined(PARALLEL)
    switch(type){
        case COMM_TYPE_INT:
            MPI_Bcast ( ptr, cnt, MPI_INT, source, _qtc_mpi_communicator );
            break;
        case COMM_TYPE_FLOAT:
            MPI_Bcast ( ptr, cnt, MPI_FLOAT, source, _qtc_mpi_communicator );
            break;
        default:
            break;
    }
#endif // defined(PARALLEL)
    return;
}


void comm_barrier(){
#if defined(PARALLEL)
    MPI_Barrier (_qtc_mpi_communicator);
#endif
    return;
}

void comm_find_winner(int *max_card, int *winner_node, int *winner_index, int cwrank, int max_index){
#if defined(PARALLEL)
    int glb_max_card = 0, index = *winner_index;
    // Reduce the cardinalities to see what the highest value is.
    MPI_Allreduce (max_card, &glb_max_card, 1, MPI_INT, MPI_MAX, _qtc_mpi_communicator);

    // If I'm not one of the winners, set my index to max
    if(*max_card != glb_max_card)
        index = max_index;

    MPI_Allreduce (&index, winner_index, 1, MPI_INT, MPI_MIN, _qtc_mpi_communicator);

    *max_card = glb_max_card;

    if( index == *winner_index ){
        *winner_node = cwrank;
    }else{
        *winner_node = -1;
    }

#else
    *winner_node = 0;
#endif // defined(PARALLEL)
    return;
}


void comm_update_communicator(int cwrank, int active_node_count){
#if defined(PARALLEL)
    static int previous_active_node_count = -1;
    int this_node_participates = 1;

    if( -1 == previous_active_node_count ){
        previous_active_node_count = active_node_count;
        return;
    }

    if(active_node_count < previous_active_node_count ){
        if( cwrank >= active_node_count ){
            this_node_participates = 0;
            std::cout << "[" << cwrank << "] Shrinking the communicator and staying out of it." << std::endl;
        }
        MPI_Comm_split(_qtc_mpi_communicator, this_node_participates, cwrank, &_qtc_mpi_communicator);
    }
    previous_active_node_count = active_node_count;
#endif
    return;
}
