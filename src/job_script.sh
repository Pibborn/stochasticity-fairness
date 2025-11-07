#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A m2_datamining
#SBATCH -p smp
#SBATCH -J automlS_DNN_banks
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5

#========[ + + + + Environment + + + + ]========#
module load lang/R/4.2.0-foss-2021b
module load lang/Python/3.9.6-GCCcore-11.2.0
module unload lang/SciPy-bundle/2021.10-foss-2021b

#========[ + + + + Job Steps + + + + ]========#
source  ../venv/bin/activate
srun python randomness_experiments_scores.py --dataset=compas --model=BinaryMI
deactivate