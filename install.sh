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

# update conda environment
info "Updating brand-speech conda env"
conda env update --file $install_script_dir/environment.yaml --prune
checkStatus $? "conda update failed"
info "conda env successfully updated"

info "Completed installations for brand-speech module"