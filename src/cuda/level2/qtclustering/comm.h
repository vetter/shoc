#ifndef _COMM_H_
#define _COMM_H_

#if defined(PARALLEL)
#  include "mpi.h"
#endif

#define COMM_TYPE_INT   0
#define COMM_TYPE_FLOAT 1

void comm_update_communicator(int cwrank, int active_node_count);
void comm_find_winner(int *max_card, int *winner_node, int *winner_index, int cwrank, int max_index);
void comm_broadcast( void *ptr, int cnt, int type, int source);
void comm_barrier(void);
int comm_get_size(void);
int comm_get_rank(void);

#endif
