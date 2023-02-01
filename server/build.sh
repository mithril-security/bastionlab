#!/bin/sh
#
# Depends on:
#   bastionlab/client/src/bastionlab/version.py
# Dependents:
#   bastionlab/.github/workflows/release_bin.yml
#   bastionlab/server/Dockerfile.gpu.sev

install_common()
{
    cd $(dirname $(pwd))

    # Libtorch installation
    if [ ! -d "libtorch" ] ; then
	
	pip3 install requests
	echo 'import requests; \
    open("libtorch.zip", "wb").write( \
          requests.get('$1').content \
            )' | python3 -i client/src/bastionlab/version.py

	if [ ! -f "libtorch.zip" ] ; then
	        echo "Failed to download libtorch.zip file"
		    exit 1
		    fi
	unzip libtorch.zip
    else
	echo "libtorch.zip is already installed at $(dirname $(pwd))libtorch"
    fi

    # Libtorch env
    export LIBTORCH=$PWD/libtorch
    if [ -d "/usr/local/cuda" ]; then
	export CUDA="/usr/local/cuda"
	export LD_LIBRARY_PATH=$CUDA/lib64:$LIBTORCH/lib:$LD_LIBRARY_PATH
    else
	export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
    fi

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
	 software-properties-common \
	 build-essential patchelf \
	 libssl-dev pkg-config \
	 curl unzip python3 python3-pip

    add-apt-repository -y ppa:ubuntu-toolchain-r/test
    apt-get install -y gcc-11 g++-11 cpp-11
    update-alternatives \
        --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-11

    install_common "__torch_cxx11_url__"
    
    LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all
    
elif [ -f "/etc/redhat-release" ] ; then
    # Dependencies installation
    yum -y install \
	python3 python3-pip \
	make gcc gcc-c++ zip \
        openssl-devel openssl

    # CentOS based distros
    if [ "$(cat /etc/centos-release | awk '{print $1}')" == "CentOS" ]; then
	yum -y install \
	        devtoolset-11-toolchain
	install_common "__torch_url__"
	scl enable devtoolset-11 'LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all' \
	    || LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all
    else # RHEL based distros
	yum -y install \
	        gcc-toolset-11
	install_common "__torch_cxx11_url__"
	scl enable gcc-toolset-11 'LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all' \
	    || LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all
    fi
else
    exit
fi
