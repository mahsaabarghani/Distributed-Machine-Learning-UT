#!/bin/bash
#SBATCH --job-name=Q2_MMMC
#SBATCH --output=Q2_MMMC.log
#SBATCH --error=error_Q2_MMMC.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --nodes=2
#SBATCH --partition=partition
#SBATCH --mem-per-cpu=1000

source /home/shared_files/pytorch_venv/bin/activate

echo '----------------------------------------'
echo 'Running on two machines with one core each'
echo 'Nodelist: ' $SLURM_JOB_NODELIST
echo "Number of nodes: " $SLURM_JOB_NUM_NODES
echo "Ntasks per node: " $SLURM_NTASKS
echo '----------------------------------------'

sleep 10 
export MASTER_PORT=$(shuf -i 2000-65000 -n 1)
export RENDEZVOUS_ID=$RANDOM
export WORLD_SIZE=4

export MASTER_ADDR=$(hostname -I | awk '{print $1}')
echo "MASTER_ADDR:MASTER_PORT=$MASTER_ADDR:$MASTER_PORT"
echo '----------------------------------------'

export OMP_NUM_THREADS=1
source /home/shared_files/pytorch_venv/bin/activate

srun accelerate launch \
  --num_processes=2 \
  --num_machines=1 \
  --machine_rank={0,1} \
  --main_process_ip=172.18.32.200 \
  --mixed_precision=no \
  --dynamo_backend=no \
  --multi_gpu \
  Q2_MMMC.py
