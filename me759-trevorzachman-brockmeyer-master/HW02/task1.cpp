#include <random>
#include <chrono>
#include <iostream>

#include "scan.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::cout;

int main(int argc, char *argv[])
{
    std::size_t n;
    sscanf(argv[1], "%zu", &n);
    float *list = new float[n];

    // randomize the seed, create distribution
    std::default_random_engine gen{static_cast<long unsigned int>(time(0))};
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    // initialize array with random floats between -1.0 and 1.0
    for (std::size_t i = 0; i < n; i++)
    {
        list[i] = dist(gen);
    }

    float *output = new float[n];
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    start = high_resolution_clock::now();
    scan(list, output, n);
    end = high_resolution_clock::now();

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    cout << duration_sec.count() << "\n";
    cout << output[0] << "\n";
    cout << output[n - 1] << "\n";

    delete[] list;
    delete[] output;
}