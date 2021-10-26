#ifndef OMP_FOR_MINER_H
#define OMP_FOR_MINER_H

#include "block_header.h"

void omp_for_miner(BlockHeader *block_headers, int num_zeros_target, unsigned int n, unsigned int *success_nonce);
void omp_for_miner_mpi_log(BlockHeader *block_headers, int num_zeros_target, unsigned int n, unsigned int *success_nonce, int my_rank);

#endif

