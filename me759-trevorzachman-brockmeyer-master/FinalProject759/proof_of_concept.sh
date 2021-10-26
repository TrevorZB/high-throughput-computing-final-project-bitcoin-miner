#!/usr/bin/env bash

#SBATCH -p wacc

#SBATCH --job-name=proof_of_concept

#SBATCH --output=outputs/proof_of_concept.out

#SBATCH --error=outputs/proof_of_concept.err

./proof_of_concept
