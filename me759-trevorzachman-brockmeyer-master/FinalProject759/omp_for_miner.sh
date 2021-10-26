#!/usr/bin/env bash

#SBATCH -p wacc

#SBATCH --job-name=omp_for_miner

#SBATCH --output=outputs/omp_for_miner.out

#SBATCH --error=outputs/omp_for_miner.err

#SBATCH --cpus-per-task=18

n=$((2**20))
./omp_for_miner $n

