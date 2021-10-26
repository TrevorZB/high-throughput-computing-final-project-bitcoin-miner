#include <omp.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include "omp_for_miner.h"
#include "block_header.h"
#include "helpers.h"
#include "picosha2.h"


using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char *argv[])
{
    // passed in iterations, and verbose logging
    unsigned int n = atoi(argv[1]);

    // success variable
    unsigned int success_nonce = 0;

    // set requested threads
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    // timers
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // start timer
    start = high_resolution_clock::now();

    // create headers to be hashed
    BlockHeader block_headers[num_threads];
    create_array_block_header(block_headers, num_threads);

    // create target to be compared against
    std::string target = uncompact_bits(block_headers[0].bits);
    int num_zeros_target = count_zeros(target);

    std::cout << "############################ OpenMP For Loop Miner #############################" << std::endl;
    std::cout << "                                                                                " << std::endl;
    std::cout << "The OpenMP For Loop Bitcoin miner uses OpenMP for loops to share work during    " << std::endl;
    std::cout << "the hashing process. Schedule used is static due to the balanced work load. This" << std::endl;
    std::cout << "is the second fastest version of the CPU Bitcoin miner.                         " << std::endl;
    std::cout << "                                                                                " << std::endl;

    std::cout << "Iterations: " << n << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Number of zeros needed at front of hash for hash to be small enough: " << num_zeros_target << std::endl;
    std::cout << "Beginning Mining... (Only showing 25 of the calculated hashes)...               " << std::endl;

    // run the omp for loop shared miner
    omp_for_miner(block_headers, num_zeros_target, n, &success_nonce);

    // calc time taken
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << "Mining finished                                                             " << std::endl;
    std::cout << n << " iterations took: " << duration_sec.count() << "ms" << std::endl;
    std::cout << "Hash Rate: " << (n / (duration_sec.count()/1000)) / 1000000 << "MH/s" << std::endl;    

    // log nonce if correct hash was found
    if (success_nonce != 0)
    {
        std::cout << "Solution found: solved nonce: " << success_nonce << std::endl;
    }
    else
    {
        std::cout<< "No solution found. Better luck next time!" << std::endl;
    }

    std::cout << "                                                                                " << std::endl;
    std::cout << "####################### End of the OpenMP For Loop Miner #######################" << std::endl;
    std::cout << "                                                                                " << std::endl;
    std::cout << "                                                                                " << std::endl;
}
