#include "convolution.h"
#include <iostream>
#include <stdlib.h>
#include <random>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char *argv[])
{
    std::size_t n, m;
    sscanf(argv[1], "%zu", &n);
    sscanf(argv[2], "%zu", &m);
    
    float *image = new float[n * n];
    float *mask = new float[m * m];

    // randomize the seed, create separate distributions for image/mask
    std::default_random_engine gen{static_cast<long unsigned int>(time(0))};
    std::uniform_real_distribution<float> image_dist(-10.0, 10.0);
    std::uniform_real_distribution<float> mask_dist(-1.0, 1.0);

    // initialize image and mask arrays with random values in specified ranges
    for (std::size_t i = 0; i < (n * n); i++)
    {
        image[i] = image_dist(gen);
    }
    for (std::size_t i = 0; i < (m * m); i++)
    {
        mask[i] = mask_dist(gen);
    }

    float *output = new float[n * n];
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    start = high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    end = high_resolution_clock::now();

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << duration_sec.count() << "\n";
    std::cout << output[0] << "\n";
    std::cout << output[(n * n) - 1] << "\n";

    delete[] image;
    delete[] mask;
    delete[] output;
}