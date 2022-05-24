#!/bin/bash
#
#SBATCH --job-name=vls_mlp_md-1000epoch
#SBATCH --output="log/osc/vls_mlp_md%j.out"
#SBATCH --signal=USR1@20
#
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --time=59
#
#SBATCH --account=PAS0027

# find project dir
cd $HOME/project/scivis-kit/pycurvis

# prep software
module load cuda/10.2.89
module load miniconda3
source activate vis

# execute job
srun python train.py --nncfg=nn_vls_md.yaml --resume
# srun python train_block.py --nncfg=nn_block_half.yaml --resume
# python test.py --testcfg=cfg/nntest_curv.yaml
# python mem.py