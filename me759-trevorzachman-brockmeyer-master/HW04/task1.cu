#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>
#include "matmul.cuh"

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
    int size = n * n;
    int threads_per_block = atoi(argv[2]);

    float *a_host = new float[size];
    float *a_dev;
    float *b_host = new float[size];
    float *b_dev;
    float *c_host = new float[size];
    float *c_dev;

    // randomize matrix a, pass to device
    cudaMalloc((void **)&a_dev, sizeof(float) * size);
    randomize_array(a_host, size, -1.0, 1.0);
    cudaMemcpy(a_dev, a_host, sizeof(float) * size, cudaMemcpyHostToDevice);
    
    // randomize matrix b, pass to device
    cudaMalloc((void **)&b_dev, sizeof(float) * size);
    randomize_array(b_host, size, -1.0, 1.0);
    cudaMemcpy(b_dev, b_host, sizeof(float) * size, cudaMemcpyHostToDevice);

    // create matrix c on the device
    cudaMalloc((void **)&c_dev, sizeof(float) * size);

    // timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matmul(a_dev, b_dev, c_dev, n, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calc time in ms
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // copy c_dev to c_host
    cudaMemcpy(c_host, c_dev, sizeof(float) * size, cudaMemcpyDeviceToHost);

    printf("%f\n", c_host[size - 1]);
    printf("%f\n", ms);

    // free all memory on host and device
    delete[] a_host;
    delete[] b_host;
    delete[] c_host;
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
}