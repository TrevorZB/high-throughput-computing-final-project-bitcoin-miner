#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "count.cuh"

void randomize_vector_int(thrust::host_vector<int> &h_vec, int start, int stop)
{
    // randomize the seed, create distribution
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);

    // fill array with random values
    std::uniform_int_distribution<int> dist(start, stop);
    for (thrust::host_vector<int>::iterator i = h_vec.begin(); i != h_vec.end(); i++)
    {
        *i = dist(gen);
    }
}

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    thrust::host_vector<int> h_vec(n);
    randomize_vector_int(h_vec, 0, 500);

    thrust::device_vector<int> d_vec = h_vec;
    thrust::device_vector<int> values(n);
    thrust::device_vector<int> counts(n);

    // timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    count(d_vec, values, counts);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calc time in ms
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    thrust::host_vector<int> h_values = values;
    thrust::host_vector<int> h_counts = counts;

    int *values_end = thrust::raw_pointer_cast(&h_values[h_values.size() - 1]);
    int *counts_end = thrust::raw_pointer_cast(&h_counts[h_counts.size() - 1]);

    printf("%d\n", *values_end);
    printf("%d\n", *counts_end);
    printf("%f\n", ms);
}
