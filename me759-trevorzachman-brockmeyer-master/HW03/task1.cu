#include <cuda.h>
#include <stdio.h>

__global__ void fact_kernel()
{
    int n = threadIdx.x;
    if (n != 0)
    {
        int fact = 1;
        for (int i = 1; i <= n; i++)
        {
            fact *= i;
        }
        printf("%d!=%d\n", n, fact);
    }
}

int main()
{
    const int num_threads = 9;
    fact_kernel<<<1, num_threads>>>();
    cudaDeviceSynchronize();
}
