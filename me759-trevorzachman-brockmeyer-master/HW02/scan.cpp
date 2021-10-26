#include "scan.h"

void scan(const float *arr, float *output, std::size_t n)
{
    float sum = 0.0;
    for (std::size_t i = 0; i < n; i++)
    {
        sum += arr[i];
        output[i] = sum;
    }
}