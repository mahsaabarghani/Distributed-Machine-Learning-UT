#!/bin/bash
#SBATCH --job-name=LogReg_a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=partition
#SBATCH --output=LogReg_a.out
echo 'jobs are starter ...'
srun --mpi=pmix_v4 python LogReg_a.py
