#!/bin/bash
#
#SBATCH --job-name=vls_phys2comp_res
#SBATCH --output="log/osc/aug%j.out"
#SBATCH --signal=USR1@20
#
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1
#SBATCH --time=14:59:00
#
#SBATCH --account=PAS0027

# find project dir
cd $HOME/project/scivis-kit/pycurvis

# prep software
module load cuda/10.2.89
module load miniconda3
source activate vis

# execute job
srun python train.py --nncfg=vls_siren_aug.yaml
# srun python train_block.py --nncfg=nn_block_half.yaml --resume
# python test.py --testcfg=cfg/nntest_curv.yaml
# python mem.py