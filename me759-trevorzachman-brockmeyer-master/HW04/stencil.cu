#include "stencil.cuh"


__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R)
{
    extern __shared__ float s[];
    int block_x = blockIdx.x;
    int thread_x = threadIdx.x;
    int block_dim = blockDim.x;

    int thread = thread_x + block_dim * block_x;
    int s_mask_size = 2 * R + 1;
    int s_image_size = 2 * R + block_dim;

    // shared memory broken into mask, image, and output arrays
    float *s_mask = &s[0];
    float *s_image = &s[s_mask_size];
    float *s_output = &s[s_mask_size + s_image_size];

    // threads with id lower than the mask size
    // bring over their mask index from global to shared memory
    if (thread_x < s_mask_size)
    {
        s_mask[thread_x] = mask[thread_x];
    }

    // calculates how many shared memory image indexes
    // this thread must bring over from global memory
    int work = (2 * R + 2 * block_dim - 1) / block_dim;

    // starting location in global memory of the image
    // for this block
    int global_loc;
    if (block_x == 0)
    {
        global_loc = 0;
    } else
    {
        global_loc = block_x * block_dim - R;
    }

    // brings over this thread's designated image indexes
    // from global memory to shared memory
    for (int i = 0; i < work; i++)
    {
        int s_i_index = thread_x * work + i; // index in the shared memory image
        int i_index = s_i_index + global_loc; // index in the global memory image
        if (s_i_index < s_image_size && i_index < n)
        {
            s_image[s_i_index] = image[i_index];
        }
    }

    // wait for threads in this block to finish bringing
    // over the mask and image values
    __syncthreads();

    // only threads within image index do the convolution
    if (thread < n)
    {
        // calculate the convolution summation
        float out = 0.0;
        for (int j = -R; j <= (int)R; j++)
        {
            float image_val;
            int i_index = thread + j;
            if (i_index < 0 || i_index > n - 1)
            {
                image_val = 1.0; // out of bounds, default value
            } else
            {
                int s_i_index = i_index - global_loc; // change to shared memory index
                image_val = s_image[s_i_index]; // grab from shared memory
            }
            out += image_val * s_mask[j + R];
        }
        s_output[thread_x] = out;

        // write shared memory output to global memory output
        output[thread] = s_output[thread_x];
    }
}


__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block)
{
    int shared_mem = 4 * R + 2 * threads_per_block + 1;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    stencil_kernel<<<blocks_per_grid, threads_per_block, sizeof(float) * shared_mem>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();
}