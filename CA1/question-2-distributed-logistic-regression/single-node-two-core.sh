#!/bin/bash
#SBATCH --job-name=LogReg_b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=partition
#SBATCH --output=LogReg_b.out
echo 'jobs are starter ...'
srun --mpi=pmix_v4 python LogReg_b.py
