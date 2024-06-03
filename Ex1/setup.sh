#!/bin/bash

# Function to print error messages in red
print_error() {
    echo -e "\033[31mERROR: $1\033[0m"
}

# Function to install Miniconda
install_miniconda() {
    echo "Conda could not be found. Installing Miniconda..."

    # Determine the platform and download the appropriate installer
    if [[ "$OSTYPE" == "darwin"* ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        INSTALLER="Miniconda3-latest-MacOSX.sh"
        wget $MINICONDA_URL -O $INSTALLER || { print_error "Failed to download Miniconda installer for macOS"; exit 1; }
        bash $INSTALLER -b -p $HOME/miniconda || { print_error "Failed to install Miniconda on macOS"; exit 1; }
    elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        INSTALLER="Miniconda3-latest-Windows.exe"
        curl -O $MINICONDA_URL -o $INSTALLER || { print_error "Failed to download Miniconda installer for Windows"; exit 1; }
        start /wait "" $INSTALLER /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D=%UserProfile%\miniconda || { print_error "Failed to install Miniconda on Windows"; exit 1; }
    else
        print_error "Unsupported platform: $OSTYPE"
        exit 1
    fi

    # Initialize conda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)" || eval "$(%UserProfile%\\miniconda\\Scripts\\conda shell.bash hook)" || { print_error "Failed to initialize Conda"; exit 1; }
    conda init || { print_error "Failed to initialize Conda"; exit 1; }

    # Remove installer
    rm $INSTALLER || { print_error "Failed to remove Miniconda installer"; exit 1; }
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
conda create -n rl_env python=3.8 -y || { print_error "Failed to create Conda environment"; exit 1; }

# Activate the new environment
echo "Activating the 'rl_env' environment..."
source activate rl_env || { print_error "Failed to activate Conda environment"; exit 1; }

# Install PyTorch
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio -c pytorch -y || { print_error "Failed to install PyTorch"; exit 1; }

# Install OpenAI Gym
echo "Installing OpenAI Gym..."
pip install gym || { print_error "Failed to install OpenAI Gym"; exit 1; }

echo "Setup complete. To activate the environment, run 'source activate rl_env'."
