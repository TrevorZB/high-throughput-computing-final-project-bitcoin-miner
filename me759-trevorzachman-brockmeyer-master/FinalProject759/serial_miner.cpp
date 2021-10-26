#include "mine.h"
#include "serial_miner.h"

void serial_miner(BlockHeader &block_header, int num_zeros_target, unsigned int n, unsigned int *success_nonce)
{
    int logs = 25;
    for (unsigned int i = 0; i < n; i++)
    {
        bool log = logs > 0 ? true : false;
        if (mine(block_header, num_zeros_target, log)) 
        {
            (*success_nonce) = i;
        }
        block_header.nonce++;
        logs--;
    }
}
