#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=tinyvla_54000_task_comp_l1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=2          # Array index = seed (0, 1, 2 for multi-seed)
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/eval_tinyvla_54000_task_comp_l1_seed_%a_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/eval_tinyvla_54000_task_comp_l1_seed_%a_%j.err

# ==========================================
# TinyVLA - Task Composition L1 Evaluation
# ==========================================
# Tests task-level generalization: the model must apply known
# primitives (pick-place, open drawer, etc.) to new object/target
# combinations never seen during training.
# ==========================================

SEED=$SLURM_ARRAY_TASK_ID
ID_NOTE="tinyvla_54000_task_comp_l1_seed_${SEED}"

# Model configuration
MODEL_PATH="/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/post_processed_tiny_vla_llava_pythia_lora_libero_goal_no_noops_lora_r_64_processed/checkpoint-54000"
MODEL_BASE="/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/parte2_llava_pythia_libero_goal_no_noops_64/1.3B"

# Directories
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA/test/libero_test"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
TINYVLA_ROOT="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA"
OUTPUT_DIR="/mnt/beegfs/a.cardamone7/outputs"

echo "=========================================="
echo "TinyVLA Task Composition L1 Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID (Seed): $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Model: $MODEL_PATH"
echo "Model Base: $MODEL_BASE"
echo "Start time: $(date)"
echo ""

# ==========================================
# Environment Setup
# ==========================================

export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:${TINYVLA_ROOT}:${WORK_DIR}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

# ==========================================
# Activate Conda Environment
# ==========================================

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate tinyvla_libero

echo "Working directory: ${WORK_DIR}"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# ==========================================
# Run Evaluation
# ==========================================

cd ${WORK_DIR}

srun python run_libero_eval_task_comp.py \
    --model_path ${MODEL_PATH} \
    --model_base ${MODEL_BASE} \
    --model_family tiny_vla \
    --task_suite_name libero_goal \
    --num_trials_per_task 50 \
    --num_steps_wait 10 \
    --env_img_res 256 \
    --seed ${SEED} \
    --run_number ${SEED} \
    --run_id_note ${ID_NOTE} \
    --local_log_dir ${OUTPUT_DIR}/logs \
    --summary_file ${OUTPUT_DIR}/logs/summary/task_comp_l1/tinyvla_seed${SEED}.json \
    --use_wandb False \
    --debug False

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task Composition L1 evaluation completed successfully!"
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
fi
echo "Seed: $SEED"
echo "Finish time: $(date)"
echo "=========================================="

exit $EXIT_CODE
