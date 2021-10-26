#ifndef HELPERS_H
#define HELPERS_H

#include <vector>
#include <string>

std::vector<unsigned char> hex_string_to_bytes(std::string hex_string);
void add_spaces_to_hex_str(std::string &hex_str);
unsigned int get_time(std::string time_stamp);
void reverse_hex_string(std::string &hex_string);
std::string uncompact_bits(unsigned int bits);
int count_zeros(std::string target);

#endif