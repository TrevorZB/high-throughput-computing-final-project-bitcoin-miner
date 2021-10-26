#include "msort.h"
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

void randomize_array_int(int *a, int size, int start, int stop)
{
    // randomize the seed, create distribution
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);

    // fill array with random values
    std::uniform_int_distribution<int> dist(start, stop);
    for (int i = 0; i < size; i++)
    {
        a[i] = dist(gen);
    }
}

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);
    int ts = atoi(argv[3]);

    int *arr = new int[n]();
    omp_set_num_threads(t);

    randomize_array_int(arr, n, -1000, 1000);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    start = high_resolution_clock::now();
    msort(arr, n, ts);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    printf("%d\n", arr[0]);
    printf("%d\n", arr[n - 1]);
    printf("%f\n", duration_sec.count());

    delete[] arr;
}
