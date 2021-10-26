#include "helpers.h"
#include "sha2.h"
#include <sstream>
#include <iomanip>

std::vector<unsigned char> hex_string_to_bytes(std::string hex_string)
{
    std::istringstream stream(hex_string);
    std::vector<unsigned char> bytes;

    unsigned int byte;
    while (stream >> std::hex >> byte)
    {
        bytes.push_back(byte);
    }

    return bytes;
}

void add_spaces_to_hex_str(std::string &hex_str)
{
    int orig_size = hex_str.size();
    for (int i = 1; i <= orig_size / 2 - 1; i++)
    {
        hex_str.insert(i * 2 + (i - 1), " ");
    }
}

unsigned int get_time(std::string time_stamp)
{
    std::tm t = {};
    std::istringstream stream(time_stamp);

    stream >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");
    std::time_t time = mktime(&t);
    return time;
}

void reverse_hex_string(std::string &hex_string)
{
    std::string temp;
    for (int i = hex_string.size() - 1; i > 0; i -= 2)
    {
        temp.push_back(hex_string[i-1]);
        temp.push_back(hex_string[i]);
    }
    hex_string = temp;
}

std::string uncompact_bits(unsigned int bits)
{
    std::stringstream stream;
    stream << std::hex << bits;
    std::string hex_str(stream.str());

    int size = std::stoi(hex_str.substr(0, 2), 0, 16);
    int padding_num = 64 - size * 2;
    std::string padding;
    for (int i = 0; i < padding_num; i++)
    {
        padding.push_back('0');
    }

    std::string target(padding + hex_str.substr(2, 6));
    for (int i = 6; i < size * 2; i++)
    {
        target.push_back('0');
    }
    
    return target;
}

int count_zeros(std::string target)
{
    int num_zeros = 0;
    for (int i = 0; i < SHA256_DIGEST_SIZE; i++)
    {
        if (target[i] == '0')
        {
            num_zeros++;
        }
        else
        {
            break;
        }
    }
    return num_zeros;
}
