#!/bin/bash

# Function to install Miniconda
install_miniconda() {
    echo "Conda could not be found. Installing Miniconda..."

    # Determine the platform and download the appropriate installer
    if [[ "$OSTYPE" == "darwin"* ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        INSTALLER="Miniconda3-latest-MacOSX.sh"
        wget $MINICONDA_URL -O $INSTALLER
        bash $INSTALLER -b -p $HOME/miniconda
    elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        INSTALLER="Miniconda3-latest-Windows.exe"
        curl -O $MINICONDA_URL -o $INSTALLER
        start /wait "" $INSTALLER /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D=%UserProfile%\miniconda
    else
        echo "Unsupported platform: $OSTYPE"
        exit 1
    fi

    # Initialize conda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)" || eval "$(%UserProfile%\\miniconda\\Scripts\\conda shell.bash hook)"
    conda init

    # Remove installer
    rm $INSTALLER
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
