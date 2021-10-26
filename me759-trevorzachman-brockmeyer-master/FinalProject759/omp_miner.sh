#!/usr/bin/env bash

#SBATCH -p wacc

#SBATCH --job-name=omp_miner

#SBATCH --output=omp_miner.out

#SBATCH --error=omp_miner.err

#SBATCH --cpus-per-task=48

n=$((2**25))
./omp_miner $n 48 1 1

