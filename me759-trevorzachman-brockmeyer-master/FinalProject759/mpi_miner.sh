#!/usr/bin/env bash

#SBATCH --partition=wacc

#SBATCH --job-name=mpi_miner

#SBATCH --output=outputs/mpi_miner.out

#SBATCH --error=outputs/mpi_miner.err

#SBATCH --nodes=1

#SBATCH --tasks-per-node=24

#SBATCH --cpus-per-task=2

module load openmpi

n=$((2**20))

mpirun -x OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK mpi_miner $n

sleep 20

make clean > /dev/null

cat outputs/proof_of_concept.out outputs/serial_miner.out outputs/omp_task_miner.out outputs/omp_for_miner.out outputs/mpi_miner.out > demo.out
