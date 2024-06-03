#!/bin/bash

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Conda and try again."
    exit
fi

# Create a new Conda environment
echo "Creating a new Conda environment named 'rl_env'..."
conda create -n rl_env python=3.8 -y

# Activate the new environment
echo "Activating the 'rl_env' environment..."
source activate rl_env

# Install PyTorch
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio -c pytorch

# Install OpenAI Gym
echo "Installing OpenAI Gym..."
pip install gym

echo "Setup complete. To activate the environment, run 'source activate rl_env'."
