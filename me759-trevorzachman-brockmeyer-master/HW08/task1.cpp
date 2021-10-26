#include "matmul.h"
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

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
    int t = atoi(argv[2]);

    float *A = new float[n * n]();
    float *B = new float[n * n]();
    float *C = new float[n * n]();

    randomize_array_float(A, n * n, 1.0, 10.0);
    randomize_array_float(B, n * n, 1.0, 10.0);

    omp_set_num_threads(t);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    start = high_resolution_clock::now();
    mmul(A, B, C, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    printf("%f\n", C[0]);
    printf("%f\n", C[n * n - 1]);
    printf("%f\n", duration_sec.count());

    delete[] A;
    delete[] B;
    delete[] C;
}
