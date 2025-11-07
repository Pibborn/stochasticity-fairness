#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A m2_datamining
#SBATCH -p m2_gpu
#SBATCH -J DecEntr
#SBATCH --mem=16G
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1      # Reserve 1 GPUs
#SBATCH --mem 64G
#========[ + + + + Environment + + + + ]========#
module load lang/R/4.2.0-foss-2021b
module load lang/Python/3.9.6-GCCcore-11.2.0
module unload lang/SciPy-bundle/2021.10-foss-2021b

#========[ + + + + Job Steps + + + + ]========#
datasets=("adult" "compas" "german" "banks" "folktables_AK" "folktables_HI")
models=("BinaryMI" "ICVAE")

num_datatsets=${#datasets[@]}
num_models=${#models[@]}

dataset_index=$(( SLURM_ARRAY_TASK_ID / num_models))
model_index=$(( SLURM_ARRAY_TASK_ID % num_models ))

dataset=${datasets[$dataset_index]}
model=${models[$model_index]}

source  ../venv/bin/activate
srun python3 decision_entropy_full_experiment.py --dataset=$dataset --model=$model --path=results
deactivate