#!/bin/bash

RED="\e[31m"
YELLOW='\033[1;33m'
GREEN="\e[32m"
DEFAULT="\e[39m"

error () {
    echo -e "${RED}Error: ${DEFAULT}$1"
    exit 1
}

warning () {
    echo -e "${YELLOW}$1${DEFAULT}"
}

info () {
    echo -e "${GREEN}$1${DEFAULT}"
}

checkStatus () {
    [ "$1" == "0" ] || error "$2"
}

install_script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# install system dependencies
dependencies=(
cmake=3.22.1-1ubuntu1.22.04.2
python3-pyaudio=0.2.11-1.3ubuntu1
portaudio19-dev=19.6.0-1.1
espeak-ng=1.50+dfsg-10
)

# install pkgs in $dependencies
for dep in ${dependencies[@]}; do
    info "Installing ${dep}"
    sudo apt-get update
    sudo apt-get -y install ${dep}
    checkStatus $? "failed to install ${dep}"
    info "Successfully installed ${dep}"
done

# only attempt to install GPU-related items if not on an RT kernel
kernel_info=$(uname -v)
if [[ $kernel_info == *"PREEMPT_RT"* ]]; then
    info "Realtime kernel detected, skipping GPU-related installations"
    driver_installed=0

    # update conda environment for CPU-only systems
    info "Updating brand-speech conda env for CPU-only systems"
    conda env update --file $install_script_dir/environment_cpu.yaml --prune
    checkStatus $? "conda update failed"
    info "conda env successfully updated"
else
    # specifically check whether nvidia-driver-525 is installed, because it requires a reboot
    nvidia_driver_version=$(dpkg -l nvidia-driver-525 | awk '/^ii/ {print $3}')
    if [ "$nvidia_driver_version" != "525.147.05-0ubuntu0.22.04.1" ]; then
        info "Installing nvidia-driver-525"
        sudo apt-get -y install nvidia-driver-525=525.147.05-0ubuntu0.22.04.1
        checkStatus $? "failed to install nvidia-driver-525"
        info "Successfully installed nvidia-driver-525"
        driver_installed=1
    else
        info "nvidia-driver-525 already installed"
        driver_installed=0
    fi

    # update conda environment for GPU systems
    info "Updating brand-speech conda env for GPU-enabled systems"
    conda env update --file $install_script_dir/environment_gpu.yaml --prune
    checkStatus $? "conda update failed"
    info "conda env successfully updated"

    # install CUDA 11.8
    info "Installing libcudnn8"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install libcudnn8=8.8.0.*-1+cuda11.8
    rm cuda-keyring_1.1-1_all.deb
fi

# update conda environment
info "Updating brand-speech-tts conda env"
conda env update --file $install_script_dir/environment_tts.yaml --prune
checkStatus $? "conda update failed"
info "conda env successfully updated"

if [ "$driver_installed" == "1" ]; then
    warning "nvidia-driver-525 was installed, please reboot the computer"
fi

info "Completed installations for brand-speech module"