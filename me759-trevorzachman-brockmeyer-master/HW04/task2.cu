#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>
#include "stencil.cuh"


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
    int R = atoi(argv[2]);
    int threads_per_block = atoi(argv[3]);

    float *image_host = new float[n];
    float *image_dev;
    float *mask_host = new float[n];
    float *mask_dev;
    float *output_host = new float[n];
    float *output_dev;

    // randomize array image, pass to device
    cudaMalloc((void **)&image_dev, sizeof(float) * n);
    randomize_array(image_host, n, -1.0, 1.0);
    cudaMemcpy(image_dev, image_host, sizeof(float) * n, cudaMemcpyHostToDevice);

    // randomize array mask, pass to device
    cudaMalloc((void **)&mask_dev, sizeof(float) * n);
    randomize_array(mask_host, n, -1.0, 1.0);
    cudaMemcpy(mask_dev, mask_host, sizeof(float) * n, cudaMemcpyHostToDevice);

    // create output array on the device
    cudaMalloc((void **)&output_dev, sizeof(float) * n);

    // timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    stencil(image_dev, mask_dev, output_dev, n, R, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calc time in ms
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // copy output_dev to output_host
    cudaMemcpy(output_host, output_dev, sizeof(float) * n, cudaMemcpyDeviceToHost);

    printf("%f\n", output_host[n - 1]);
    printf("%f\n", ms);

    // free all memory on host and device
    delete[] image_host;
    delete[] mask_host;
    delete[] output_host;
    cudaFree(image_dev);
    cudaFree(mask_dev);
    cudaFree(output_dev);
}