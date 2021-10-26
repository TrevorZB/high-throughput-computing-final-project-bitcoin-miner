#ifndef SERIAL_MINER_H
#define SERIAL_MINER_H

#include "block_header.h"

void serial_miner(BlockHeader &block_header, int num_zeros_target, unsigned int n, unsigned int *success_nonce);

#endif