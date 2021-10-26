#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>
#include "reduce.cuh"

void randomize_array(float *a, int size, float start, float stop)
{
    // randomize the seed, create distribution
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(start, stop);

    for (int i = 0; i < size; i++)
    {
        a[i] = dist(gen);
    }
}

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    int threads_per_block = atoi(argv[2]);
    int blocks_per_grid = n / (threads_per_block * 2);

    float *input_host = new float[n];
    float *input_dev;
    float *output_dev;

    // randomize input array, pass to device
    cudaMalloc((void **)&input_dev, sizeof(float) * n);
    randomize_array(input_host, n, -1.0, 1.0);
    cudaMemcpy(input_dev, input_host, sizeof(float) * n, cudaMemcpyHostToDevice);

    // create output array on device
    cudaMalloc((void **)&output_dev, sizeof(float) * blocks_per_grid);

    // timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    reduce(&input_dev, &output_dev, n, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calc time in ms
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // copy sum to first element of host input array
    cudaMemcpy(input_host, input_dev, sizeof(float), cudaMemcpyDeviceToHost);

    // prints for assignment
    printf("%f\n", input_host[0]);
    printf("%f\n", ms);

    // free all memory on host and device
    delete[] input_host;
    cudaFree(input_dev);
    cudaFree(output_dev);
}