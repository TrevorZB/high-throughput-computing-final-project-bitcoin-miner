#include "omp_task_miner.h"
#include "mine.h"
#include "omp.h"

void omp_task_miner(BlockHeader &block_header, int num_zeros_target, unsigned int n, unsigned int *success_nonce)
{
    int logs = 25;
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (unsigned int i = 0; i < n; i++)
            {
                // task spawner handles copying of object and nonce increment
                BlockHeader block_header_copy = block_header;
                block_header_copy.nonce = i;
                #pragma omp task firstprivate(block_header_copy)
                {
                    bool log = false;
                    int tid = omp_get_thread_num();
                    if (tid == 4)
                    {
                        log = logs > 0 ? true : false;
                        logs--;
                    }
                    if (mine(block_header_copy, num_zeros_target, log))
                    {
                        (*success_nonce) = block_header_copy.nonce;
                    }
                }
            }
        }
    }
}
