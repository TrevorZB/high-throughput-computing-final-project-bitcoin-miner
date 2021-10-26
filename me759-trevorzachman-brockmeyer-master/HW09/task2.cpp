#include "montecarlo.h"
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

int main(int argc, char *argv[])
{
    int incircle;
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);

    float *x = new float[n]();
    float *y = new float[n]();

    randomize_array_float(x, n, -1.0, 1.0);
    randomize_array_float(y, n, -1.0, 1.0);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    omp_set_num_threads(t);

    start = high_resolution_clock::now();
    incircle = montecarlo(n, x, y, 1.0);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    float pi = (4.0f * float(incircle)) / float(n);

    printf("%f\n", pi);
    printf("%f\n", duration_sec.count());

    delete[] x;
    delete[] y;
}