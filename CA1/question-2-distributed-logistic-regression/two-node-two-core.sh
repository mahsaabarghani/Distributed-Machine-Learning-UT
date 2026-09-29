#!/bin/bash
#SBATCH --job-name=LogReg_c
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --partition=partition
#SBATCH --output=LogReg_c.out
echo 'jobs are starter ...'
srun --mpi=pmix_v4 python LogReg_c.py
