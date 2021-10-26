#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>
#include "matmul.cuh"


void randomize_array_int(int *a, int size, int start, int stop)
{
    // randomize the seed, create distribution
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);

    // fill array with random values
    std::uniform_int_distribution<int> dist(start, stop);
    for (int i = 0; i < size; i++)
    {
        a[i] = dist(gen);
    }
}

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

void randomize_array_double(double *a, int size, double start, double stop)
{
    // randomize the seed, create distribution
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);

    // fill array with random values
    std::uniform_real_distribution<double> dist(start, stop);
    for (int i = 0; i < size; i++)
    {
        a[i] = dist(gen);
    }
}

void run_matmul_1(int n, int block_dim)
{
    int size = n * n;
    int *a_host = new int[size];
    int *a_dev;
    int *b_host = new int[size];
    int *b_dev;
    int *c_host = new int[size];
    int *c_dev;

    // randomize matrix a, pass to device
    cudaMalloc((void **)&a_dev, sizeof(int) * size);
    randomize_array_int(a_host, size, 1, 10);
    cudaMemcpy(a_dev, a_host, sizeof(int) * size, cudaMemcpyHostToDevice);
    
    // randomize matrix b, pass to device
    cudaMalloc((void **)&b_dev, sizeof(int) * size);
    randomize_array_int(b_host, size, 1, 10);
    cudaMemcpy(b_dev, b_host, sizeof(int) * size, cudaMemcpyHostToDevice);

    // create matrix c on the device
    cudaMalloc((void **)&c_dev, sizeof(int) * size);

    // timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_1(a_dev, b_dev, c_dev, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calc time in ms
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // copy c_dev to c_host
    cudaMemcpy(c_host, c_dev, sizeof(int) * size, cudaMemcpyDeviceToHost);

    printf("%d\n", c_host[0]);
    printf("%d\n", c_host[size - 1]);
    printf("%f\n", ms);

    // free all memory on host and device
    delete[] a_host;
    delete[] b_host;
    delete[] c_host;
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
}

void run_matmul_2(int n, int block_dim)
{
    int size = n * n;
    float *a_host = new float[size];
    float *a_dev;
    float *b_host = new float[size];
    float *b_dev;
    float *c_host = new float[size];
    float *c_dev;

    // randomize matrix a, pass to device
    cudaMalloc((void **)&a_dev, sizeof(float) * size);
    randomize_array_float(a_host, size, 1.0, 10.0);
    cudaMemcpy(a_dev, a_host, sizeof(float) * size, cudaMemcpyHostToDevice);
    
    // randomize matrix b, pass to device
    cudaMalloc((void **)&b_dev, sizeof(float) * size);
    randomize_array_float(b_host, size, 1.0, 10.0);
    cudaMemcpy(b_dev, b_host, sizeof(float) * size, cudaMemcpyHostToDevice);

    // create matrix c on the device
    cudaMalloc((void **)&c_dev, sizeof(float) * size);

    // timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_2(a_dev, b_dev, c_dev, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calc time in ms
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // copy c_dev to c_host
    cudaMemcpy(c_host, c_dev, sizeof(float) * size, cudaMemcpyDeviceToHost);

    printf("%f\n", c_host[0]);
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

void run_matmul_3(int n, int block_dim)
{
    int size = n * n;
    double *a_host = new double[size];
    double *a_dev;
    double *b_host = new double[size];
    double *b_dev;
    double *c_host = new double[size];
    double *c_dev;

    // randomize matrix a, pass to device
    cudaMalloc((void **)&a_dev, sizeof(double) * size);
    randomize_array_double(a_host, size, 1.0, 10.0);
    cudaMemcpy(a_dev, a_host, sizeof(double) * size, cudaMemcpyHostToDevice);
    
    // randomize matrix b, pass to device
    cudaMalloc((void **)&b_dev, sizeof(double) * size);
    randomize_array_double(b_host, size, 1.0, 10.0);
    cudaMemcpy(b_dev, b_host, sizeof(double) * size, cudaMemcpyHostToDevice);

    // create matrix c on the device
    cudaMalloc((void **)&c_dev, sizeof(double) * size);

    // timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_3(a_dev, b_dev, c_dev, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calc time in ms
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // copy c_dev to c_host
    cudaMemcpy(c_host, c_dev, sizeof(double) * size, cudaMemcpyDeviceToHost);

    printf("%f\n", c_host[0]);
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

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    int block_dim = atoi(argv[2]);

    run_matmul_1(n, block_dim);
    run_matmul_2(n, block_dim);
    run_matmul_3(n, block_dim);
}