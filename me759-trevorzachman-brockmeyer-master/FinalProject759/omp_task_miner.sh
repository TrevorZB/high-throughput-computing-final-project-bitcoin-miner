#!/usr/bin/env bash

#SBATCH -p wacc

#SBATCH --job-name=omp_task_miner

#SBATCH --output=outputs/omp_task_miner.out

#SBATCH --error=outputs/omp_task_miner.err

#SBATCH --cpus-per-task=18

n=$((2**20))
./omp_task_miner $n
