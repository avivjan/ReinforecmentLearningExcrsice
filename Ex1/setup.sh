#!/bin/bash

# Function to print error messages in red
print_error() {
    echo -e "\033[31mERROR: $1\033[0m"
}

# Function to install Anaconda
install_anaconda() {
    echo "Conda could not be found. Installing Anaconda..."

    # Determine the platform and download the appropriate installer
    if [[ "$OSTYPE" == "darwin"* ]]; then
        ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-latest-MacOSX-x86_64.sh"
        INSTALLER="Anaconda3-latest-MacOSX.sh"
        wget $ANACONDA_URL -O $INSTALLER || { print_error "Failed to download Anaconda installer for macOS"; exit 1; }
        bash $INSTALLER -b -p $HOME/anaconda3 || { print_error "Failed to install Anaconda on macOS"; exit 1; }
    elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-latest-Windows-x86_64.exe"
        INSTALLER="Anaconda3-latest-Windows.exe"
        curl -O $ANACONDA_URL -o $INSTALLER || { print_error "Failed to download Anaconda installer for Windows"; exit 1; }
        start /wait "" $INSTALLER /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D=%UserProfile%\anaconda3 || { print_error "Failed to install Anaconda on Windows"; exit 1; }
    else
        print_error "Unsupported platform: $OSTYPE"
        exit 1
    fi

    # Initialize conda
    eval "$($HOME/anaconda3/bin/conda shell.bash hook)" || eval "$(%UserProfile%\\anaconda3\\Scripts\\conda shell.bash hook)" || { print_error "Failed to initialize Conda"; exit 1; }
    conda init || { print_error "Failed to initialize Conda"; exit 1; }

    # Remove installer
    rm $INSTALLER || { print_error "Failed to remove Anaconda installer"; exit 1; }
    echo "Anaconda installed successfully."
}

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    install_anaconda
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

# Verification phase
echo "Verifying the installations..."

# Verify Conda activation
source activate rl_env || { print_error "Conda environment activation failed"; exit 1; }
echo -e "\033[32mConda environment activated successfully.\033[0m"

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)" &> /dev/null
if [ $? -ne 0 ]; then
    print_error "PyTorch verification failed"
    exit 1
else
    echo -e "\033[32mPyTorch installed successfully.\033[0m"
fi

# Verify OpenAI Gym installation
python -c "import gym; print(gym.__version__)" &> /dev/null
if [ $? -ne 0 ]; then
    print_error "OpenAI Gym verification failed"
    exit 1
else
    echo -e "\033[32mOpenAI Gym installed successfully.\033[0m"
fi

echo -e "\033[32mAll installations verified successfully.\033[0m"
