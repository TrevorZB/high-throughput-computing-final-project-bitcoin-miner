#!/usr/bin/env bash

#SBATCH -p wacc

#SBATCH --job-name=demo

#SBATCH --output=outputs/out.out

#SBATCH --error=outputs/err.err

module load openmpi
make all

sbatch proof_of_concept.sh
sbatch serial_miner.sh
sbatch omp_task_miner.sh
sbatch omp_for_miner.sh
sbatch mpi_miner.sh
