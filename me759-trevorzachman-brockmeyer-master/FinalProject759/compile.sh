#!/usr/bin/env bash

#SBATCH -p wacc

#SBATCH --job-name=compile

#SBATCH --output=compile.out

#SBATCH --error=compile.err

#SBATCH --time=0-00:01:00

make all
