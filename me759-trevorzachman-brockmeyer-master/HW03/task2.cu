#include <cuda.h>
#include <stdio.h>
#include <chrono>
#include <random>

__global__ void calc_kernel(int a, int *dA, int dim) 
{
    int t_x = threadIdx.x;
    int b_x = blockIdx.x;
    dA[(dim * b_x) + t_x] = a * t_x + b_x;
}

int random_int()
{
    // randomize the seed, create distribution
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> int_dist(1, 100);

    // return random int between 1 and 100
    return int_dist(gen);
}

int main() {
    const int a = random_int();
    const int n = 16;
    const int blocks = 2;
    const int threads_per_block = 8;

    int hA[n], *dA;
    cudaMalloc((void **)&dA, sizeof(int) * n);

    calc_kernel<<<blocks, threads_per_block>>>(a, dA, n / 2);

    cudaMemcpy(&hA, dA, sizeof(int) * n, cudaMemcpyDeviceToHost);

    for (int i : hA)
    {
        printf("%d ", i);
    }
    printf("\n");

    cudaFree(dA);
}
