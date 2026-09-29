#!/bin/bash
#SBATCH --job-name=Q2_SMMC
#SBATCH --output=Q2_SMMC.log
#SBATCH --error=error_Q2_SMMC.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --partition=partition
#SBATCH --mem-per-cpu=1000

echo '----------------------------------------'
echo 'Nodelist: ' $SLURM_JOB_NODELIST
echo "Number of nodes: " $SLURM_JOB_NUM_NODES
echo "Ntasks per node: " $SLURM_NTASKS
echo '----------------------------------------'

export MASTER_PORT=2958
export RENDEZVOUS_ID=$RANDOM
export WORLD_SIZE=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR:MASTER_PORT=$MASTER_ADDR:$MASTER_PORT"
echo '----------------------------------------'

export OMP_NUM_THREADS=1
source /home/shared_files/pytorch_venv/bin/activate

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  Q2_SMMC.py
