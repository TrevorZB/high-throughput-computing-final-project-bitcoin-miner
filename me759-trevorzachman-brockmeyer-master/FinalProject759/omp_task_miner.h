#ifndef OMP_TASK_MINER_H
#define OMP_TASK_MINER_H

#include "block_header.h"

void omp_task_miner(BlockHeader &block_header, int num_zeros_target, unsigned int n, unsigned int *success_nonce);

#endif
