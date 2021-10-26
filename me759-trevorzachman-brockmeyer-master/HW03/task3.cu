#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>
#include "vscale.cuh"

#include <iostream>

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
    int threads_per_block = 512;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    float *a_host = new float[n];
    float *a_dev;
    float *b_host = new float[n];
    float *b_dev;

    // randomize array a, pass to device
    cudaMalloc((void **)&a_dev, sizeof(float) * n);
    randomize_array(a_host, n, -10.0, 10.0);
    cudaMemcpy(a_dev, a_host, sizeof(float) * n, cudaMemcpyHostToDevice);
    
    // randomize array b, pass to device
    cudaMalloc((void **)&b_dev, sizeof(float) * n);
    randomize_array(b_host, n, 0.0, 1.0);
    cudaMemcpy(b_dev, b_host, sizeof(float) * n, cudaMemcpyHostToDevice);

    // timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vscale<<<blocks_per_grid, threads_per_block>>>(a_dev, b_dev, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calc time in ms
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(b_host, b_dev, sizeof(float) * n, cudaMemcpyDeviceToHost);

    printf("%f\n", ms);
    printf("%f\n", b_host[0]);
    printf("%f\n", b_host[n - 1]);

    delete[] a_host;
    delete[] b_host;
    cudaFree(a_dev);
    cudaFree(b_dev);
}