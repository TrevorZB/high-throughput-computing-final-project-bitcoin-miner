#!/usr/bin/env bash

#SBATCH -p wacc

#SBATCH --job-name=FirstSlurm

#SBATCH --output=FirstSlurm.out

#SBATCH --error=FirstSlurm.err

#SBATCH --time=0-00:01:00

#SBATCH --cpus-per-task=2

echo $HOSTNAME
