#!/usr/bin/env bash

#SBATCH -p wacc

#SBATCH --job-name=serial_miner

#SBATCH --output=outputs/serial_miner.out

#SBATCH --error=outputs/serial_miner.err

n=$((2**20))
./serial_miner $n
