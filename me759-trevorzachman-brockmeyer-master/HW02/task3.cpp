#include "matmul.h"
#include <iostream>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <vector>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

// class used to store testing functionality
class Testing
{
private:
    // stores the array-based functions to time test
    std::vector<void (*)(const double* A,
                        const double* B,
                        double* C,
                        const unsigned int n)> functions;

    // stores the single vector-based function to time test
    void (*mmul4_func)(const std::vector<double>& A,
                       const std::vector<double>& B,
                       double* C,
                       const unsigned int n) = &mmul4;

    // variables used to time test the matmul functions
    unsigned int n;
    const double *a;
    const double *b;
    double *c;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

public:
    Testing(const double* A, const double* B, unsigned int N)
    {
        // constructor, stores function pointers and variables/arrays
        functions.push_back(&mmul1);
        functions.push_back(&mmul2);
        functions.push_back(&mmul3);
        a = A;
        b = B;
        n = N;
    }

    void time_test()
    {
        for (auto func : functions)
        {
            c = new double[n * n]();

            start = high_resolution_clock::now();
            (*func)(a, b, c, n);
            end = high_resolution_clock::now();
            duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

            std::cout << duration_sec.count() << "\n";
            std::cout << c[(n * n) - 1] << "\n";
            delete[] c;
        }
    }

    void time_vec_test()
    {
        // creates vector versions of the a and b arrays
        std::vector<double> a_vec(a, a + (n * n));
        std::vector<double> b_vec(b, b + (n * n));
        c = new double[n * n]();

        start = high_resolution_clock::now();
        (*mmul4_func)(a_vec, b_vec, c, n);
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

        std::cout << duration_sec.count() << "\n";
        std::cout << c[(n * n) - 1] << "\n";
        delete[] c;
    }
};



int main()
{
    unsigned int N = 1024;
    std::cout << N << "\n";

    double *A = new double[N * N]();
    double *B = new double[N * N]();

    // randomize the seed, create distribution
    std::default_random_engine gen{static_cast<long unsigned int>(time(0))};
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    // initialize arrays with random doubles between 1.0 and 10.0
    for (std::size_t i = 0; i < (N * N); i++)
    {
        A[i] = dist(gen);
    }
    for (std::size_t i = 0; i < (N * N); i++)
    {
        B[i] = dist(gen);
    }

    // creating testing instance and run the time tests
    Testing test(A, B, N);
    test.time_test();
    test.time_vec_test();

    delete[] A;
    delete[] B;
}