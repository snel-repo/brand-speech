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

# first argument is whether to install the 3-gram LM
install_lm=${1:-1}

install_script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

info "Installing Git LFS"
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

# install system dependencies
dependencies=(
cmake=3.22.1-1ubuntu1.22.04.2
python3-pyaudio=0.2.11-1.3ubuntu1
portaudio19-dev=19.6.0-1.1
espeak-ng=1.50+dfsg-10
git-lfs=3.5.1
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

    # update conda environment for CPU-only systems
    info "Updating brand-speech conda env for CPU-only systems"
    conda env update --file $install_script_dir/environment_cpu.yaml --prune
    checkStatus $? "conda update failed"
    info "conda env successfully updated"
else
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

    if [ "$install_lm" == "1" ]; then
        # check if the LM is already downloaded
        if [ -d "/samba/languageModel" ]; then
            info "Language model already downloaded"
        else
            info "Downloading language model, this will take a while"
            mkdir -p /samba/languageModel
            wget --continue -O /samba/languageModel/lm.tar.gz https://datadryad.org/api/v2/files/2547356/download
            info "Extracting language model"
            tar -xvf /samba/languageModel/lm.tar.gz -C /samba
            rm /samba/languageModel/lm.tar.gz
            chmod -R 777 /samba/languageModel
            info "Successfully downloaded language model"

            info "Installing language model environment"
            conda env update --file $install_script_dir/environment_lm.yaml --prune
        fi
    fi
fi

# update conda environment
info "Updating brand-speech-tts conda env"
conda env update --file $install_script_dir/environment_tts.yaml --prune
checkStatus $? "conda update failed"
info "conda env successfully updated"

# download the TTS model
info "Downloading LJ Speech TTS model"
curwd=$(pwd)
cd $install_script_dir/lib/StyleTTS2/StyleTTS2
if [ -d "Models" ]; then
    rm -rf Models
fi
git clone https://huggingface.co/yl4579/StyleTTS2-LJSpeech
mv StyleTTS2-LJSpeech/Models Models
rm -rf StyleTTS2-LJSpeech
# move our config to the model directory
cp -rf $install_script_dir/assets/tts_config/config.yml Models/LJSpeech/config.yml
cd $curwd
info "Successfully downloaded LJ Speech TTS model"

info "Completed installations for brand-speech module"