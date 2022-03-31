#!/bin/bash
#
#SBATCH --job-name=cat2sph-attn_net-10000epoch-single_block
#SBATCH --output="log/osc/attn_net_single_block_job%j.out"
#SBATCH --signal=USR1@20
#
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1
#SBATCH --time=3:49:00
#
#SBATCH --account=PAS0027

# find project dir
cd $HOME/project/scivis-kit/pycurvis

# prep software
module load cuda/10.2.89
module load miniconda3
source activate vis

# execute job
srun python train.py --nncfg=nn_single_block_attn.yaml --resume
# srun python train_block.py --nncfg=nn_block_half.yaml --resume
# python test.py --testcfg=cfg/nntest_curv.yaml
# python mem.py