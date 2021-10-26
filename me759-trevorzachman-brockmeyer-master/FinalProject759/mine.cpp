#include "mine.h"
#include "sha2.h"
#include "helpers.h"
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

bool mine(BlockHeader &block_header, int num_zeros_target, bool log)
{
    // used to compare to target
    int count_zeros = 0;
    
    // interpret the blockheader struct as byte data
    const unsigned char *byte_array = (unsigned char*)&block_header;

    // buffers to store digests
    unsigned char digest[SHA256_DIGEST_SIZE];
    unsigned char second_digest[SHA256_DIGEST_SIZE];

    // hash, then hash the result of the first hash
    sha256(byte_array, BLOCK_HEADER_SIZE, digest);
    sha256(digest, SHA256_DIGEST_SIZE, second_digest);

    // turn bytes into hexstring
    std::stringstream ss;
    ss << std::hex;
    for (int i = 0; i < SHA256_DIGEST_SIZE; i++)
    {
        ss << std::setw(2) << std::setfill('0') << (int)second_digest[i];
    }
    std::string result = ss.str();

    if (log)
    {
        std::cout << "Calculated hash:" << std::endl;
        std::cout << result << std::endl;
    }

    // count trailing zeros of hexstring
    for (int i = SHA256_DIGEST_SIZE-1; i >= 0; i--)
    {
        if (result[i] == '0')
        {
            count_zeros++;
        }
        else
        {
            break;
        }
    }

    // hash is solved if result is less than target
    return count_zeros >= num_zeros_target;
}
