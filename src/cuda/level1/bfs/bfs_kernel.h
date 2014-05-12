#ifndef BFS_KERNEL_H_
#define BFS_KERNEL_H_

#include<cuda.h>

#define get_queue_index(tid) ((tid%NUM_P_PER_MP))
#define get_queue_offset(tid) ((tid%NUM_P_PER_MP)*W_Q_SIZE)


__global__ void BFS_kernel_warp(
        unsigned int *levels,
        unsigned int *edgeArray,
        unsigned int *edgeArrayAux,
        int W_SZ,
        int CHUNK_SZ,
        unsigned int numVertices,
        int curr,
        int *flag);


__global__ void BFS_kernel_one_block(
	volatile unsigned int *frontier,
	unsigned int frontier_len,
	volatile unsigned int *cost,
	volatile int *visited,
	unsigned int *edgeArray,
	unsigned int *edgeArrayAux,
	unsigned int numVertices,
	unsigned int numEdges,
	volatile unsigned int *frontier_length,
    unsigned int num_p_per_mp,
    unsigned int w_q_size);

__global__ void BFS_kernel_SM_block(
	volatile unsigned int *frontier,
	volatile unsigned int *frontier2,
	unsigned int frontier_len,
	volatile unsigned int *cost,
	volatile int *visited,
	unsigned int *edgeArray,
	unsigned int *edgeArrayAux,
	unsigned int numVertices,
	unsigned int numEdges,
	volatile unsigned int *frontier_length,
    unsigned int num_p_per_mp,
    unsigned int w_q_size);


__global__ void BFS_kernel_multi_block(
        volatile unsigned int *frontier,
        volatile unsigned int *frontier2,
        unsigned int frontier_len,
        volatile unsigned int *cost,
        volatile int *visited,
        unsigned int *edgeArray,
        unsigned int *edgeArrayAux,
        unsigned int numVertices,
        unsigned int numEdges,
        volatile unsigned int *frontier_length,
        unsigned int NUM_P_PER_MP,
        unsigned int W_Q_SIZE);


__global__ void Reset_kernel_parameters(unsigned int *frontier_length);

__global__ void Frontier_copy(
	unsigned int *frontier,
	unsigned int *frontier2,
	unsigned int *frontier_length);

__global__ void BFS_kernel_one_block_spill(
    volatile unsigned int *frontier,
    unsigned int frontier_len,
    volatile unsigned int *cost,
    volatile int *visited,
    unsigned int *edgeArray,
    unsigned int *edgeArrayAux,
    unsigned int numVertices,
    unsigned int numEdges,
    volatile unsigned int *frontier_length,
    const unsigned int max_mem);


__global__ void BFS_kernel_SM_block_spill(
    volatile unsigned int *frontier,
    volatile unsigned int *frontier2,
    unsigned int frontier_len,
    volatile unsigned int *cost,
    volatile int *visited,
    unsigned int *edgeArray,
    unsigned int *edgeArrayAux,
    unsigned int numVertices,
    unsigned int numEdges,
    volatile unsigned int *frontier_length,
    const unsigned int max_mem);

__global__ void BFS_kernel_multi_block_spill(
    volatile unsigned int *frontier,
    volatile unsigned int *frontier2,
    unsigned int frontier_len,
    volatile unsigned int *cost,
    volatile int *visited,
    unsigned int *edgeArray,
    unsigned int *edgeArrayAux,
    unsigned int numVertices,
    unsigned int numEdges,
    volatile unsigned int *frontier_length,
    const unsigned int max_mem);



#endif        //BFS_KERNEL_H
