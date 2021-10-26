#include "montecarlo.h"
#include <cmath>

int montecarlo(const size_t n, const float *x, const float *y, const float radius)
{
    int count = 0;
    #pragma omp parallel for simd reduction(+:count)
    for (size_t i = 0; i < n; i++)
    {
        if (sqrt(pow(x[i], 2) + pow(y[i], 2)) < radius)
        {
            count++;
        }
    }
    return count;
}