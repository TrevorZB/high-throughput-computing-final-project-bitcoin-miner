template <typename TYPE>
__global__ void matmul_kernel(const TYPE* A, const TYPE* B, TYPE* C, unsigned int n)
{
    extern __shared__ char s[];
    TYPE* shmem = reinterpret_cast<TYPE*>(s);

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int block_dim = blockDim.x;
    TYPE c_val = 0;

    // split shared memory into A and B pointers
    TYPE *s_a = &shmem[0];
    TYPE *s_b = &shmem[block_dim * 2];

    // global index into A and B for this block
    int a_index = block_dim * n * block_y;
    int b_index = block_dim * block_x;

    // loop to pass block along the row/columns of A/B
    int end = (n + block_dim - 1) / block_dim;
    for (int j = 0; j < end; j++)
    {
        // indexes into shared and global memory for this thread
        int s_a_index = thread_y * block_dim + thread_x;
        int s_b_index = s_a_index;
        int g_a_index = n * thread_y + a_index + thread_x;
        int g_b_index = n * thread_y + b_index + thread_x;

        // grab global memory and put into shared memory
        // makes sure within index bounds
        if (g_a_index < n*n)
        {
            s_a[s_a_index] = A[g_a_index];
        } else
        {
            s_a[s_a_index] = 0;
        }
        if (g_b_index < n * n)
        {
            s_b[s_b_index] = B[g_b_index];
        } else
        {
            s_b[s_b_index] = 0;
        }

        // wait for threads to bring in correct parts of A and B
        __syncthreads();

        // add to summation the row and column for this thread
        for (int i = 0; i < block_dim; i++)
        {
            c_val += s_a[thread_y * block_dim + i] * s_b[block_dim * i + thread_x];
        }

        // wait for threads to finish calculations, then next iteration begins
        __syncthreads();

        // move block over and down along A and B
        a_index += block_dim; 
        b_index += n * block_dim;
    }

    // check if dead thread
    if (block_x * block_dim + thread_x < n)
    {
        // calc index of global C matrix
        int g_c_index = n * block_dim * block_y + block_dim * block_x;
        g_c_index += (n * thread_y + thread_x);

        // check if in boundaries of matrix
        if (g_c_index < n * n)
        {
            // assign to global C matrix
            C[g_c_index] = c_val;
        }
    }
}