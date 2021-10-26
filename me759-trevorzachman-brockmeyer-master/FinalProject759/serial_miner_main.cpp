#include "block_header.h"
#include "helpers.h"
#include "picosha2.h"
#include "serial_miner.h"
#include <iostream>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char *argv[])
{
    // passed in iterations
    unsigned int n = atoi(argv[1]);

    // success variables
    unsigned int success_nonce;

    // create header to be hashed
    BlockHeader block_header = create_block_header();

    // create target to be compared against
    std::string target = uncompact_bits(block_header.bits);
    int num_zeros_target = count_zeros(target);

    // timers
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    std::cout << "################################# Serial Miner #################################" << std::endl;
    std::cout << "                                                                                " << std::endl;
    std::cout << "The serial Bitcoin miner uses a naive approach of single threaded/single        " << std::endl;
    std::cout << "process mining. This is the fourth fastest of the CPU Bitcoin miners.           " << std::endl;
    std::cout << "                                                                                " << std::endl;

    std::cout << "Iterations: " << n << std::endl;
    std::cout << "Number of zeros needed at front of hash for hash to be small enough: " << num_zeros_target << std::endl;
    std::cout << "Beginning Mining... (Only showing 25 of the calculated hashes)...               " << std::endl;

    // start timer
    start = high_resolution_clock::now();

    // run the serial miner
    serial_miner(block_header, num_zeros_target, n, &success_nonce);

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
    std::cout << "########################### End of the Serial Miner ############################" << std::endl;
    std::cout << "                                                                                " << std::endl;
    std::cout << "                                                                                " << std::endl;
}