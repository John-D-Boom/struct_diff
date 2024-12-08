#!/bin/bash
#SBATCH --job-name=gen_esm_tokens
#SBATCH --output=result-%j.out
#SBATCH --error=result-%j.err
#SBATCH --time=07:00:00
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --account pmg


###SETUP ENVIRONMENT####
module load cuda12.2/
module load mamba
mamba init
source /burg/home/jb5005/.bashrc

# Activate/create the 'struct_diff' environment
if mamba env list | grep -q 'struct_diff'; then
    echo "Environment 'struct_diff' found. Activating it..."
    mamba activate struct_diff
else
    echo "Environment 'struct_diff' not found. Creating it..."
    mamba create -n struct_diff
    mamba activate struct_diff
fi

#Try to install packages to ensure that everything is up to date
#TODO: add version control to everything to stabilize environment

mamba install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia --yes
pip install esm wandb
mamba install ipykernel ipywidgets lightning biotite --yes

python generate_esm_tokens.py

tar -cf /pmglocal/jb5005/small_struct_diff_files.tar -C /pmglocal/jb5005/struct_diff_data/ .

rsync -a /pmglocal/jb5005/small_struct_diff_files.tar /manitou/pmg/users/jb5005/struct_diff_data
