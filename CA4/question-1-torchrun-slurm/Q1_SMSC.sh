#!/bin/bash
#SBATCH --job-name=Q1_SMSC
#SBATCH --output=Q1_SMSC.log
#SBATCH --error=error_Q1_SMSC.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=partition
#SBATCH --mem-per-cpu=1000
source /home/shared_files/pytorch_venv/bin/activate

echo '----------------------------------------'
echo 'Nodelist: ' $SLURM_JOB_NODELIST
echo "Number of nodes: " $SLURM_JOB_NUM_NODES
echo "Ntasks per node: " $SLURM_NTASKS
echo '----------------------------------------'

export MASTER_PORT=2048
export RENDEZVOUS_ID=$RANDOM
export WORLD_SIZE=2

export MASTER_ADDR=$(hostname -I | awk '{print $1}')
echo "MASTER_ADDR:MASTER_PORT=$MASTER_ADDR:$MASTER_PORT"
echo '----------------------------------------'

export OMP_NUM_THREADS=1

srun torchrun --nnodes=1 --nproc-per-node=1 --rdzv-id=$RENDEZVOUS_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT Q1_SMSC.py