#ifndef MINE_H
#define MINE_H

#include "block_header.h"

bool mine(BlockHeader &block_header, int num_zeros_target, bool log);

#endif