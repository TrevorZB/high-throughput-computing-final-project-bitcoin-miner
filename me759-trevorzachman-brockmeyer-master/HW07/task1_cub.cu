#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>


using namespace cub;
CachingDeviceAllocator  g_allocator(true);

void randomize_array_float(float *a, int size, float start, float stop)
{
    // randomize the seed, create distribution
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);

    // fill array with random values
    std::uniform_real_distribution<float> dist(start, stop);
    for (int i = 0; i < size; i++)
    {
        a[i] = dist(gen);
    }
}


int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    float *h_in = new float[n];
    randomize_array_float(h_in, n, -1.0, 1.0);

    float *d_in = NULL;
    g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * n);
    cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice);

    float *d_sum = NULL;
    g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1);

    void *d_temp_storage = NULL;
    size_t d_temp_storage_bytes = 0;
    DeviceReduce::Sum(d_temp_storage, d_temp_storage_bytes, d_in, d_sum, n);
    g_allocator.DeviceAllocate(&d_temp_storage, d_temp_storage_bytes);

    // timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    DeviceReduce::Sum(d_temp_storage, d_temp_storage_bytes, d_in, d_sum, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost);

    // calc time in ms
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("%f\n", gpu_sum);
    printf("%f\n", ms);

    if (d_in)
    {
        g_allocator.DeviceFree(d_in);
    }
    if (d_sum)
    {
        g_allocator.DeviceFree(d_sum);
    }
    if (d_temp_storage)
    {
        g_allocator.DeviceFree(d_temp_storage);
    }
    if (h_in)
    {
        delete[] h_in;
    }
}
