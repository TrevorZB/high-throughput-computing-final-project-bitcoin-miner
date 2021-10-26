#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>


void randomize_vector_float(thrust::host_vector<float> &h_vec, float start, float stop)
{
    // randomize the seed, create distribution
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);

    // fill array with random values
    std::uniform_real_distribution<float> dist(start, stop);
    for (thrust::host_vector<float>::iterator i = h_vec.begin(); i != h_vec.end(); i++)
    {
        *i = dist(gen);
    }
}


int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    thrust::host_vector<float> h_vec(n);
    randomize_vector_float(h_vec, -1.0, 1.0);

    thrust::device_vector<float> d_vec = h_vec;

    // timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    float result = thrust::reduce(d_vec.begin(), d_vec.end());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calc time in ms
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("%f\n", result);
    printf("%f\n", ms);
}
