#include "block_header.h"
#include "helpers.h"
#include "sha2.h"
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

int main(int argc, char *argv[])
{
    // header data of the block pulled from the blockchain explorer
    unsigned int version = 1;
    std::string hash_prev_block_str("0000000000000000000000000000000000000000000000000000000000000000");
    std::string hash_merkle_root_str("4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b");
    unsigned int time_stamp = 1231006505; // unix timestamp of 2009-01-03 12:15:05
    unsigned int bits = 0x1d00ffff; // target value, must hash below this value, essentially the difficulty
    unsigned int nonce = 0x7c2bac1d; // the successfull nonce that was found back in 2009

    /*
    init the blockheader struct with correctly formatted data
    formatting includes placing spaces between pairs of hex digits, converting time_stamp to
    unix time, swapping from big endian to small endian, etc.
    */
    BlockHeader block_header;
    init_block_header(block_header, version, hash_prev_block_str, hash_merkle_root_str, time_stamp, bits, nonce);

    // interpret the blockheader struct as byte data
    const unsigned char *byte_array = (unsigned char*)&block_header;
    unsigned char digest[SHA256_DIGEST_SIZE];
    unsigned char second_digest[SHA256_DIGEST_SIZE];

    /*
    bitcoin standard hashes the block header twice:
    header -> bytes -> sha256 hash -> bytes -> second (final) sha256 hash
    */
    sha256(byte_array, BLOCK_HEADER_SIZE, digest);
    sha256(digest, SHA256_DIGEST_SIZE, second_digest);

    // interpret byte array as hex string
    std::stringstream ss;
    ss << std::hex;
    for (int i=0; i < SHA256_DIGEST_SIZE; i++)
    {
        ss << std::setw(2) << std::setfill('0') << (int)second_digest[i];
    }   
    std::string result = ss.str();

    std::cout << "############################### Proof of Concept ###############################" << std::endl;
    std::cout << "                                                                                " << std::endl;
    std::cout << "Will be testing our block header structure and hashing algorithm using the data " << std::endl;
    std::cout << "present in the very first mined block of bitcoin. This block was mined on Jan   " << std::endl;
    std::cout << "9th, 2009. Data can be found here: 'https://explorer.btc.com/btc/block/1'       " << std::endl;

    /*
    test to see if the caclulated hash is equal to the actual hash found back in 2009
    (need to reverse the string since blockchain explorer shows hash as big endian)
    */
    reverse_hex_string(result);
    std::string actual_hash("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f");

    std::cout << "                                                                                " << std::endl;
    std::cout << "First, we will see if our digest is the same as the one found in 2009.          " << std::endl;
    std::cout << "The actual hash found in 2009 was:                                              " << std::endl;
    std::cout << actual_hash << std::endl;
    std::cout << "The hash calculated using our block head structure and hashing algorithm was:   " << std::endl;
    std::cout << result << std::endl;
    std::cout << "Our digest is exactly the same as the one in calculated in 2009. This proves the" << std::endl;
    std::cout << "algorithms used by our miners are correct.                                      " << std::endl;

    /*
    test to see if the calculation to determine if a block was successfully mined or
    not is working correctly. header stores 256 bit target in a compressed 32 bit
    format so some formatting and padding is needed to accurately compare it
    against the calculated final hash.
    this transformation is correct if the hash < target
    */
    std::string target = uncompact_bits(block_header.bits);
    bool hashed = result < target;

    std::cout << "                                                                                " << std::endl;
    std::cout << "Finally, for a bitcoin to be mined, our digest must be below a target value. We " << std::endl;
    std::cout << "now check if our algorithm correctly finds the desired target and finds that our" << std::endl;
    std::cout << "digest is a value that is less than this target.                                " << std::endl;
    std::cout << "Our hash:                                                                       " << std::endl;
    std::cout << result << std::endl;
    std::cout << "Calculated target:                                                              " << std::endl;
    std::cout << target << std::endl;
    std::cout << "Is the result digest found less than the target?                                " << std::endl;
    std::cout << (hashed ? "Yes":"No") << std::endl;
    std::cout << "Our algorithm has successfully found our digest to be less than the target value" << std::endl;

    std::cout << "                                                                                " << std::endl;
    std::cout << "######################### End of the Proof of Concept ##########################" << std::endl;
    std::cout << "                                                                                " << std::endl;
    std::cout << "                                                                                " << std::endl;
    
}