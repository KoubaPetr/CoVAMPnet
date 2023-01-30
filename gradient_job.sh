#!/bin/bash
#SBATCH --job-name=gradients
#SBATCH --output=gradients_%a.log
#SBATCH --time=1-12:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=node-12,dgx-2,dgx-5

module purge
module load cuDNN/7.6.4.38-gcccuda-2019b
module load CUDA/10.1.243-GCC-8.3.0
module load Anaconda3/5.0.1
source /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh

conda activate tf

python gradient_job.py --num_frames 5 --job_no ${SLURM_ARRAY_TASK_ID}
