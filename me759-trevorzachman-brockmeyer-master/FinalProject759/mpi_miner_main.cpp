#include "mpi.h"
#include <omp.h>
#include "block_header.h"
#include "helpers.h"
#include "omp_for_miner.h"
#include <iostream>

int main(int argc, char *argv[])
{
    // omp threads requested per process
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    // timing variables
    double start = 0.0;
    double end = 0.0;
    double time = 0.0;
    bool solved = false;

    // mpi init
    int my_rank, num_p;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);
    MPI_Status status;

    // iteration variables, each process assigned part of work
    unsigned int n = atoi(argv[1]);
    int iterations = n / num_p;
    unsigned int start_nonce = iterations * (my_rank + 1) - (iterations - 1);
    unsigned int success_nonce = 0;

    // start time
    if (my_rank == 0)
    {
        start = MPI_Wtime();
    }

    // create block headers to be hashed
    BlockHeader block_headers[num_threads];
    create_array_block_header_nonce(block_headers, num_threads, start_nonce);

    // create target to be compared against
    std::string target = uncompact_bits(block_headers[0].bits);
    int num_zeros_target = count_zeros(target);

    if (my_rank == 0)
    {
        std::cout << "######################## OpenMPI + OpenMP For Loop Miner #######################" << std::endl;
        std::cout << "                                                                                " << std::endl;
        std::cout << "The OpenMPI + OpenMP For Loop Bitcoin Miner utilizes OpenMPI to spawn multiple  " << std::endl;
        std::cout << "processes across multiple nodes. Each process then utilizes OpenMP to spawn     " << std::endl;
        std::cout << "multiple threads to share the work through the hashing process. This is the     " << std::endl;
        std::cout << "fastest version of the CPU Bitcoin miner.                                       " << std::endl;
        std::cout << "                                                                                " << std::endl;

        std::cout << "Iterations: " << n << std::endl;
        std::cout << "Nodes: " << "1" << std::endl;
        std::cout << "Processes: " << num_p << std::endl;
        std::cout << "Threads per Process: " << num_threads << std::endl;
        std::cout << "Number of zeros needed at front of hash for hash to be small enough: " << num_zeros_target << std::endl;
        std::cout << "Beginning Mining... (Only showing 25 of the calculated hashes)...               " << std::endl;
    }

    // mining function using omp for loop work sharing
    omp_for_miner_mpi_log(block_headers, num_zeros_target, iterations, &success_nonce, my_rank);

    // master process
    if (my_rank == 0)
    {
        // buffer to store mining results from all processes
        int success_nonces[num_p];
        success_nonces[my_rank] = success_nonce;

        // recieve mining results from all processes
        for (int i = 1; i < num_p; i++)
        {
            MPI_Recv(success_nonces+i, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            std::cout << "RANK: " << my_rank << " RECIEVED SUCCESS NONCE: " << success_nonces[i] << " FROM RANK: " << i << std::endl;
        }

        // timing
        end = MPI_Wtime();
        time = end - start;

        std::cout << "Mining finished                                                             " << std::endl;
        std::cout << n << " iterations took: " << time*1000 << "ms" << std::endl;
        std::cout << "Hash Rate: " << (n / time) / 1000000 << "MH/s" << std::endl;    

        // log nonce if correct hash was found
        for (int i = 1; i < num_p; i++)
        {
            if (success_nonces[i] != 0)
            {
                std::cout << "Solution found: solved nonce: " << success_nonce << std::endl;
                solved = true;
            }
        }

        if (!solved)
        {
            std::cout<< "No solution found. Better luck next time!" << std::endl;
        }

        std::cout << "                                                                                " << std::endl;
        std::cout << "################## End of the OpenMPI + OpenMP For Loop Miner ##################" << std::endl;
        std::cout << "                                                                                " << std::endl;
        std::cout << "                                                                                " << std::endl;
    }
    // worker processes
    else
    {
        // send the mining results to the master process
        MPI_Send(&success_nonce, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // cleanup
    MPI_Finalize();
}
