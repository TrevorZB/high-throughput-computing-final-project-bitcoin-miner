#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>

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
    int my_rank;
    int num_p;
    int partner;
    int tag = 0;
    MPI_Status status;
    double start;
    double end;

    int n = atoi(argv[1]);
    float *send_message = new float[n]();
    float *recieve_message = new float[n]();
    double time[1];
    randomize_array_float(send_message, n, -1.0, 1.0);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

    partner = my_rank == 0 ? 1 : 0;

    if (my_rank == 0)
    {
        start = MPI_Wtime();
        MPI_Send(send_message, n, MPI_FLOAT, partner, tag, MPI_COMM_WORLD);
        MPI_Recv(recieve_message, n, MPI_FLOAT, partner, tag, MPI_COMM_WORLD, &status);
        end = MPI_Wtime();
        MPI_Recv(time, 1, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD, &status);
        printf("%f\n", (end - start + time[0]) * 1000.0);
    } else if (my_rank == 1)
    {
        start = MPI_Wtime();
        MPI_Recv(recieve_message, n, MPI_FLOAT, partner, tag, MPI_COMM_WORLD, &status);
        MPI_Send(send_message, n, MPI_FLOAT, partner, tag, MPI_COMM_WORLD);
        end = MPI_Wtime();
        time[0] = end - start;
        MPI_Send(time, 1, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD);
    }

    delete[] send_message;
    delete[] recieve_message;
}