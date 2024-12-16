#!/bin/bash
#SBATCH --job-name=scale=0_06
#SBATCH --output=result-%j.out
#SBATCH --error=result-%j.err
#SBATCH --time=48:00:00
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
mamba install ipykernel ipywidgets lightning biotite matplotlib seaborn --yes

#Data Transfer from Manitou

#First initalize directory in pmglocal
# Define the folder path
FOLDER_PATH="/pmglocal/jb5005/struct_diff_data"

# Check if the folder exists
if [ ! -d "$FOLDER_PATH" ]; then
  # Create the folder, including parent directories if needed
  mkdir -p "$FOLDER_PATH"
  echo "Folder created: $FOLDER_PATH"
else
  echo "Folder already exists: $FOLDER_PATH"
fi

#Transfer and unpack data
rsync -a  /manitou/pmg/users/jb5005/struct_diff_data/struct_token_comp.tar.gz /pmglocal/jb5005/struct_diff_data
tar -xf /pmglocal/jb5005/struct_diff_data/struct_token_comp.tar.gz -C /pmglocal/jb5005/struct_diff_data/


#Run File
python train.py
