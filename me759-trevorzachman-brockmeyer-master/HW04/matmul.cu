#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n)
{
    int thread = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread < n * n)
    {
        int row = thread / n;
        int col = thread % n;
        float value = 0.0;
        
        for (int i = 0; i < n; i++)
        {
            value += A[row * n + i] * B[col + i * n];
        }
        C[thread] = value;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block)
{
    int blocks_per_grid = (n * n + threads_per_block - 1) / threads_per_block;
    matmul_kernel<<<blocks_per_grid, threads_per_block>>>(A, B, C, n);
    cudaDeviceSynchronize();
}
