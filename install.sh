#!/bin/bash

RED="\e[31m"
GREEN="\e[32m"
DEFAULT="\e[39m"

error () {
    echo -e "${RED}Error: ${DEFAULT}$1"
    exit 1
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
)

# install pkgs in $dependencies
for dep in ${dependencies[@]}; do
    info "Installing ${dep}"
    sudo apt-get update
    sudo apt-get -y install ${dep}
    checkStatus $? "failed to install ${dep}"
    info "Successfully installed ${dep}"
done

# update conda environment
info "Updating brand-speech conda env"
conda env update --file $install_script_dir/environment.yaml --prune
checkStatus $? "conda update failed"
info "conda env successfully updated"

info "Completed installations for brand-speech module"