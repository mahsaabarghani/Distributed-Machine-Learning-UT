#!/bin/bash
#SBATCH --job-name=Q2_SMSC
#SBATCH --output=Q2_SMSC.log
#SBATCH --error=error_Q2_SMSC.log
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=partition
#SBATCH --mem-per-cpu=1000

echo '----------------------------------------'
echo 'Nodelist: ' $SLURM_JOB_NODELIST
echo "Number of nodes: " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:" $SLURM_NTASKS
echo '----------------------------------------'
echo "SLURM_NTASKS:" $SLURM_NTASKS
echo "SLURM_NTASKS_PER_NODE:" $SLURM_NTASKS_PER_NODE
echo "SLURM_JOB_NUM_NODES:" $SLURM_JOB_NUM_NODES

export MASTER_PORT=5048
export WORLD_SIZE=1
export MASTER_ADDR=$(hostname)

export OMP_NUM_THREADS=1

source /home/shared_files/pytorch_venv/bin/activate

srun accelerate launch \
  --mixed_precision=no \
  --dynamo_backend=no \
  --num_processes=1 \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  Q2_SMSC.py
