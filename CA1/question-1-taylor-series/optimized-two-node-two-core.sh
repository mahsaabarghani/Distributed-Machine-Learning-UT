#!/bin/bash
#SBATCH --job-name=sqrt_d_2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --partition=partition
#SBATCH --output=sqrt_d_2.out
echo 'jobs are starter ...'
srun --mpi=pmix_v4 python sqrt_d_2.py
