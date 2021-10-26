#include "omp_for_miner.h"
#include "mine.h"
#include "omp.h"

void omp_for_miner(BlockHeader *block_headers, int num_zeros_target, unsigned int n, unsigned int *success_nonce)
{
    int logs = 25;
    #pragma omp parallel for schedule(static) // load is balanced, can use static here
    for (unsigned int i = 0; i < n; i++)
    {
        bool log = false;
        int tid = omp_get_thread_num();
        if (tid == 0)
        {
            log = logs > 0 ? true : false;
            logs--;
        }
        block_headers[tid].nonce = i;
        if (mine(block_headers[tid], num_zeros_target, log)) 
        {
            (*success_nonce) = i;
        }
    }
}

void omp_for_miner_mpi_log(BlockHeader *block_headers, int num_zeros_target, unsigned int n, unsigned int *success_nonce, int my_rank)
{
    int logs = 25;
    #pragma omp parallel for schedule(static) // load is balanced, can use static here
    for (unsigned int i = 0; i < n; i++)
    {
        bool log = false;
        int tid = omp_get_thread_num();
        if (tid == 0 && my_rank == 0)
        {
            log = logs > 0 ? true : false;
            logs--;
        }
        block_headers[tid].nonce = i;
        if (mine(block_headers[tid], num_zeros_target, log)) 
        {
            (*success_nonce) = i;
        }
    }
}
