#!/bin/bash

#SBATCH -A hpc_default
#SBATCH --exclude=tnode[01-17]
#SBATCH --exclude=gnode14
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

MODEL_PATH=$1
MODEL_BASE=$2
CHANGE_SPAWN=$3
SPAWN_TRAIN_DISTRIBUTION=$4
RUN_NUMBER=$5
echo "*********************************************************************"
echo "RUN model ${MODEL_PATH} with base ${MODEL_BASE} and run number ${RUN_NUMBER}"
echo "CHANGE SPAWN=${CHANGE_SPAWN} SPAWN_TRAIN_DISTRIBUTION=${SPAWN_TRAIN_DISTRIBUTION}"
echo "*********************************************************************"

if [ "$RUN_NUMBER" -ne 1 ]; then
    SAVE=False
fi

srun torchrun --standalone --nnodes 1 --nproc-per-node 1 eval_libero.py \
    --model_path ${MODEL_PATH} \
    --model_base ${MODEL_BASE} \
    --change_spawn ${CHANGE_SPAWN} \
    --spawn_train_distribution ${SPAWN_TRAIN_DISTRIBUTION} \
    --run_number ${RUN_NUMBER} \
    --debug True
