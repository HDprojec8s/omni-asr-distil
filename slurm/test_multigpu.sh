#!/bin/bash
#SBATCH --job-name=test-mgpu
#SBATCH --output=logs/test-mgpu_%j.out
#SBATCH --error=logs/test-mgpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=p4
#SBATCH --time=0:05:00
#SBATCH --qos=gpuultimate

cd /nfs1/scratch/students/witzlch88229/projects/omni-asr-distil || exit 1
mkdir -p logs
source .venv/bin/activate

srun python -c "
import torch, torch.distributed as dist, os

# Map SLURM env vars to PyTorch distributed (same as fairseq2's SlurmHandler)
os.environ['RANK'] = os.environ['SLURM_PROCID']
os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
os.environ['LOCAL_WORLD_SIZE'] = os.environ['SLURM_NTASKS_PER_NODE']
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '29500')

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

dist.init_process_group('nccl')
r = dist.get_rank()
t = torch.ones(1, device=f'cuda:{local_rank}') * r
dist.all_reduce(t)
print(f'Rank {r}: all_reduce = {t.item()}')
dist.destroy_process_group()
"
