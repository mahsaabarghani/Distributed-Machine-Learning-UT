#!/bin/bash
#SBATCH --job-name=sqrt_e
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --partition=partition
#SBATCH --output=sqrt_e.out
echo 'jobs are starter ...'
srun --mpi=pmix_v4 python sqrt_e.py
