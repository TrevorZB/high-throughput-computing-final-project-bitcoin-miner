#include "block_header.h"
#include "helpers.h"
#include <algorithm>
#include <ctime>


BlockHeader create_block_header()
{
    BlockHeader block_header;
    unsigned int version = 0x20000000;
    std::string hash_prev_block_str("000000000000000000c8e0f60a4f0c33d0ec762a8c6e933bb032e23417fabf4b");
    std::string hash_merkle_root_str("194497a81c3e45068d1925c9c3389a4298996be9e7607e8ade3f6be95fa37040");
    unsigned int time_stamp = std::time(0);
    unsigned int bits = 0x1d00ffff;
    unsigned int nonce = 0x0;

    init_block_header(block_header, version, hash_prev_block_str, hash_merkle_root_str, time_stamp, bits, nonce);

    return block_header;
}

void create_array_block_header(BlockHeader *block_headers, int size)
{
    unsigned int same_time_stamp = std::time(0);
    for (int i = 0; i < size; i++)
    {
        BlockHeader block_header;
        unsigned int version = 0x20000000;
        std::string hash_prev_block_str("000000000000000000c8e0f60a4f0c33d0ec762a8c6e933bb032e23417fabf4b");
        std::string hash_merkle_root_str("194497a81c3e45068d1925c9c3389a4298996be9e7607e8ade3f6be95fa37040");
        unsigned int time_stamp = same_time_stamp;
        unsigned int bits = 0x1d00ffff;
        unsigned int nonce = 0x0;

        init_block_header(block_header, version, hash_prev_block_str, hash_merkle_root_str, time_stamp, bits, nonce);
        block_headers[i] = block_header;
    }
}

void create_array_block_header_nonce(BlockHeader *block_headers, int size, unsigned int start_nonce)
{
    unsigned int same_time_stamp = std::time(0);
    for (int i = 0; i < size; i++)
    {
        BlockHeader block_header;
        unsigned int version = 0x20000000;
        std::string hash_prev_block_str("000000000000000000c8e0f60a4f0c33d0ec762a8c6e933bb032e23417fabf4b");
        std::string hash_merkle_root_str("194497a81c3e45068d1925c9c3389a4298996be9e7607e8ade3f6be95fa37040");
        unsigned int time_stamp = same_time_stamp;
        unsigned int bits = 0x1d00ffff;
        unsigned int nonce = start_nonce;

        init_block_header(block_header, version, hash_prev_block_str, hash_merkle_root_str, time_stamp, bits, nonce);
        block_headers[i] = block_header;
    }
}

void init_block_header(
    BlockHeader &block_header,
    unsigned int version,
    std::string hash_prev_block_str,
    std::string hash_merkle_root_str,
    unsigned int time_stamp,
    unsigned int bits,
    unsigned int nonce)
{
    add_spaces_to_hex_str(hash_prev_block_str);
    add_spaces_to_hex_str(hash_merkle_root_str);

    std::vector<unsigned char> hash_prev_block = hex_string_to_bytes(hash_prev_block_str);
    std::vector<unsigned char> hash_merkle_root = hex_string_to_bytes(hash_merkle_root_str);

    std::reverse(hash_prev_block.begin(), hash_prev_block.end());
    std::reverse(hash_merkle_root.begin(), hash_merkle_root.end());
    
    block_header.version = version;

    std::copy(hash_prev_block.begin(), hash_prev_block.end(), block_header.hash_prev_block);
    std::copy(hash_merkle_root.begin(), hash_merkle_root.end(), block_header.hash_merkle_root);

    block_header.time = time_stamp;
    block_header.bits = bits;
    block_header.nonce = nonce;
}
