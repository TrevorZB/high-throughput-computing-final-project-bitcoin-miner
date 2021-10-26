#include "cluster.h"
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>
#include <algorithm>

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

void init_centers(float *centers, int size, float n)
{
    for (int i = 0; i < size; i++)
    {
        centers[i] = ((2.0f * float(i) + 1.0f) * n) / (2.0f * float(size));
    }
}

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);

    float *arr = new float[n]();
    float *centers = new float[t]();
    float *dists = new float[t]();

    randomize_array_float(arr, n, 0.0, float(n));
    std::sort(arr, arr + n);
    init_centers(centers, t, float(n));

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    start = high_resolution_clock::now();
    cluster(n, t, arr, centers, dists);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    float *max = std::max_element(dists, dists + t);
    int max_tid = max - dists;

    printf("%f\n", *max);
    printf("%d\n", max_tid);
    printf("%f\n", duration_sec.count());

    delete[] arr;
    delete[] centers;
    delete[] dists;
}