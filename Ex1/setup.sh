#!/bin/bash

# Function to install Miniconda
install_miniconda() {
    echo "Conda could not be found. Installing Miniconda..."
    # Download Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest.sh
    # Run the installer
    bash Miniconda3-latest.sh -b -p $HOME/miniconda
    # Initialize conda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
    # Remove installer
    rm Miniconda3-latest.sh
    echo "Miniconda installed successfully."
}

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    install_miniconda
else
    echo "Conda is already installed."
fi

# Create a new Conda environment
echo "Creating a new Conda environment named 'rl_env'..."
conda create -n rl_env python=3.8 -y

# Activate the new environment
echo "Activating the 'rl_env' environment..."
source activate rl_env

# Install PyTorch
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio -c pytorch -y

# Install OpenAI Gym
echo "Installing OpenAI Gym..."
pip install gym

echo "Setup complete. To activate the environment, run 'source activate rl_env'."
