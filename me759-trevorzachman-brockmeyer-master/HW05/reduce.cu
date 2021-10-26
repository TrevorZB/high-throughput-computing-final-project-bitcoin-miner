#include "reduce.cuh"


__global__ void reduce_kernel(float* g_idata, float* g_odata, unsigned int n)
{
    extern __shared__ float s[];
    int bid = blockIdx.x;
    int threads_per_block = blockDim.x;
    int tid = threadIdx.x;

    // shared memory as input pointer
    float *s_input = &s[0];

    int g_input_index = tid + threads_per_block * bid * 2; // global input index
    int sec_g_input_index = g_input_index + threads_per_block; // second global input index

    // only pull into shared memory if within bounds of the inputted array
    if (g_input_index < n && sec_g_input_index < n)
    {
        s_input[tid] = g_idata[g_input_index] + g_idata[g_input_index + threads_per_block];
    } else if (g_input_index < n)
    {
        s_input[tid] = g_idata[g_input_index];
    } else 
    {
        s_input[tid] = 0.0;
    }

    // wait for all threads to finish bringing in data
    __syncthreads();

    // sum the elements in the shared memory input array
    for (int i = threads_per_block / 2; i > 0; i /= 2)
    {
        if (tid < i)
        {
            s_input[tid] += s_input[i + tid];
        }
        __syncthreads(); // since doing in place, have to wait for each round of threads to finish
    }

    // write shared summation to global output array
    if (tid == 0)
    {
        g_odata[bid] = s_input[0];
    }
}


__host__ void reduce(float** input, float** output, unsigned int N, unsigned int threads_per_block)
{
    float *temp;
    int blocks_per_grid = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);

    // first call of the kernel, rest of calls will be in the while loop
    reduce_kernel<<<blocks_per_grid, threads_per_block, sizeof(float) * threads_per_block>>>(*input, *output, N);

    // iterates until the last run only used 1 block, means we are done
    while (blocks_per_grid != 1)
    {
        N = blocks_per_grid; // new size of input
        blocks_per_grid = (N + threads_per_block * 2 - 1) / (threads_per_block * 2); // new number of blocks needed

        // flip input and output pointers
        temp = *input;
        *input = *output;
        *output = temp;

        // call kernel with new values
        reduce_kernel<<<blocks_per_grid, threads_per_block, sizeof(float) * threads_per_block>>>(*input, *output, N);
    }
    // one last flip so the input pointer points to the ending summation at the 0th index
    temp = *input;
    *input = *output;
    *output = temp;

    cudaDeviceSynchronize();
}