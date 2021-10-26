__global__ void vscale(const float *a, float *b, unsigned int n)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n)
    {
        b[index] *= a[index];
    }
}