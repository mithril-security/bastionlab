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

verify_deb_deps()
{
    if [ "$(id -u)" -ne 0 ]; then
	while [ $# -ne 0 ]; do
	    dpkg -s $1 > /dev/null 2>&1
	    if [ "$(echo $?)" -ne 0 ]; then
		echo "You have missing packages, please run as superuser" >&2
		exit 1
	    fi
	    shift
	done
    fi
}

verify_rhel_deps()
{
    if [ "$(id -u)" -ne 0 ]; then
	while [ $# -ne 0 ]; do
	    yum list installed $1 > /dev/null 2>&1
	    if [ "$(echo $?)" -ne 0 ]; then
		echo "You have missing packages, please run as superuser" >&2
		exit 1
	    fi
	    shift
	done
    fi
}

# For Debian based distros
if [ -f "/etc/debian_version" ] ; then

    set software-properties-common \
	build-essential patchelf \
	libssl-dev pkg-config \
	curl unzip python3 python3-pip \
	gcc-11 g++-11 cpp-11

    verify_deb_deps
    
    if [ "$(id -u)" -eq 0 ]; then
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
    fi

    install_common "__torch_cxx11_url__"
    
    LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all

# For RHEL based distros    
elif [ -f "/etc/redhat-release" ] ; then
    set python3 python3-pip \
	make gcc gcc-c++, zip \
        openssl-devel openssl

    verify_rhel_deps

    if [ "$(id -u)" -eq 0 ]; then
	# Dependencies installation
	yum -y install \
	    python3 python3-pip \
	    make gcc gcc-c++ zip \
            openssl-devel openssl
	case "$(cat /etc/centos-release  > /dev/null 2>&1 | awk '{print $1}')" in
	    "CentOS") # CentOS based distros
		yum -y install devtoolset-11-toolchain
		;;
	    *) # Other RHEL based distros
		yum -y install gcc-toolset-11
		;;
	esac
    fi
    case "$(cat /etc/centos-release  > /dev/null 2>&1 | awk '{print $1}')" in
	"CentOS") # CentOS based distros
	    install_common "__torch_url__"
	    scl enable devtoolset-11 'LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all' \
		|| LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all
	    ;;
	*) # Other RHEL based distros
	    install_common "__torch_cxx11_url__"
	    scl enable gcc-toolset-11 'LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all' \
		|| LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all
	    ;;
    esac
else
    echo "Unrecognized linux version, needs manual installation, check the documentation:\n\
    	  https://bastionlab.readthedocs.io/en/latest/docs/getting-started/installation/" >&2
    exit 1
fi
