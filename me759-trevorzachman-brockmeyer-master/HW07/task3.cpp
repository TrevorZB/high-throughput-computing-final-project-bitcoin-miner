#include <omp.h>
#include <stdio.h>


void print_factorial(int i)
{
    int factorial = 1;
    for (int j = 1; j <= i; j++)
    {
        factorial *= j;
    }
    printf("%d!=%d\n", i, factorial);
}


int main()
{
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("Number of threads: %d\n", omp_get_num_threads());
        }
        printf("I am thread No. %d\n", omp_get_thread_num());
        #pragma omp barrier
        #pragma omp for
        for (int i = 1; i < 9; i++)
        {
            print_factorial(i);
        }
    }
}
