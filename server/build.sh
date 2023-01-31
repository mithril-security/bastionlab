#!/bin/sh

install_common()
{
    cd $(dirname $(pwd))

    # Libtorch installation
    pip3 install requests
    echo 'import requests; \
    open("libtorch.zip", "wb").write( \
        requests.get('$1').content \
        )' | python3 -i client/src/bastionlab/version.py

    if [ ! -f "libtorch.zip" ] ; then
	echo "Failed to download libtorch.zip file"
	exit 1
    fi

    # Libtorch env
    export LIBTORCH=$PWD/libtorch
    export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

    # Rustup installation
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
    sh rustup.sh -y
    export PATH=$HOME/.cargo/bin:$PATH

    cd server
}

# For Debian based distros
if [ -f "/etc/debian_version" ] ; then
    # Dependencies installation
    apt-get update
    apt-get -y install \
	 build-essential \
	 patchelf \
	 libssl-dev \
	 pkg-config \
	 curl \
	 unzip

    install_common "__torch_cxx11_url__"
    
elif [ -f "/etc/redhat-release" ] ; then
    # Dependencies installation
    yum -y install \
	python3 python3-pip \
        zip openssl-devel \
        devtoolset-11-toolchain

    install_common "__torch_url__"

else
    exit
fi
