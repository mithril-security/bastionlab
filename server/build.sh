#!/usr/bin/env bash
#
# Depends on:
#   bastionlab/client/src/bastionlab/version.py
# Dependents:
#   bastionlab/server/Dockerfile.gpu.sev
#
# Env variable "BASTIONLAB_BUILD_AS_ROOT" can be useful
#   when building in a docker image as root.

declare -a deb_dependencies=(
    [0]=lsb-release
    [1]=software-properties-common
    [2]=gcc-11
    [3]=g++-11
    [4]=cpp-11

    [5]=build-essential
    [6]=patchelf
    [7]=libssl-dev
    [8]=pkg-config
    [9]=curl
    [10]=unzip
    [11]=python3
    [12]=python3-pip
    [13]=python3-venv
    [14]=sudo
)

declare -a rhel_dependencies=(
    [0]=python3
    [2]=python3-pip
    [3]=make
    [4]=gcc
    [5]=gcc-c++
    [6]=zip
    [7]=openssl-devel
    [8]=openssl
    [9]=python3-virtualenv
    [10]=sudo
)

unrecognized_distro()
{
    echo "Unrecognized linux version, needs manual installation, check the documentation:">&2
    echo "https://bastionlab.readthedocs.io/en/latest/docs/getting-started/installation/" >&2
    exit 1
}

install_common()
{
    cd $(dirname $(pwd))

    # Libtorch installation
    if [ ! -d "libtorch" ] ; then
	if [ ! -z "${BASTIONLAB_BUILD_AS_ROOT}" ] && [ "$(id -u)" -eq 0 ]; then
	    export VIRTUAL_ENV=/opt/venv
	    python3 -m venv $VIRTUAL_ENV
	    export PATH="$VIRTUAL_ENV/bin:$PATH"
	fi
	pip3 install requests
	echo 'import requests; \
    open("libtorch.zip", "wb").write( \
          requests.get('$1').content \
            )' | python3 -i client/src/bastionlab/version.py
	
	if [ ! -f "libtorch.zip" ] ; then
	    echo "[❌] Failed to download libtorch.zip file" >&2
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
    command -v cargo > /dev/null 2>&1
    EXIT_STATUS=$?
    if ! (exit $EXIT_STATUS) ; then
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/rustup.sh
	sh /tmp/rustup.sh -y
	export PATH=$HOME/.cargo/bin:$PATH
    fi
    
    cd server
}

verify_deps()
{
    echo "Verifying dependencies..."
    args=("$@")
    checkcmd=("${args[0]}")
    packages=("${args[@]:1}")
    EXIT_STATUS=0
    for package in "${packages[@]}"; do
	$checkcmd $package > /dev/null 2>&1
	EXIT_STATUS=$?
	if ! (exit $EXIT_STATUS) ; then
	    echo $package
	    echo "You have missing packages, installing them..." >&2
	    return $EXIT_STATUS
	else
	    echo "[✔️ ]" $package
	fi
    done
    return $EXIT_STATUS
}

# Debian-based dependencies installation
install_deb_deps()
{
    apt-get update
    apt-get -y upgrade
    # Configuring repositories
    apt-get -y install software-properties-common
    command -v add-apt-repository > /dev/null 2>&1
    EXIT_STATUS=$?
    if ! (exit $EXIT_STATUS) ; then
	echo "Adding toolchain repository to the /etc/apt/sources.list.d/toolchain.list file..."
	apt-get -y install lsb-release
	VRELEASE=$(lsb_release -c | awk '{print $2}')
	DEB_TOOLCHAIN="https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu ${VRELEASE} main"
	echo "deb" $DEB_TOOLCHAIN | tee -a /etc/apt/sources.list.d/toolchain.list
	echo "deb-src" $DEB_TOOLCHAIN | tee -a /etc/apt/sources.list.d/toolchain.list
	sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F
    else
	add-apt-repository -y ppa:ubuntu-toolchain-r/test
    fi
    EXIT_STATUS=$?
    apt-get update
    if ! (exit $EXIT_STATUS) ; then
	apt-get -y install "${deb_dependencies[@]:5}"
    else
	apt-get -y install "${deb_dependencies[@]:2}"
	update-alternatives \
            --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
            --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
            --slave /usr/bin/gcov gcov /usr/bin/gcov-11
    fi
}

# RHEL-based dependencies installation
install_rhel_deps()
{
    yum -y install "${rhel_dependencies[@]}"
    case "$(cat /etc/centos-release | awk '{print $1}')" in
	"CentOS") # CentOS based distros
	    yum -y install devtoolset-11-toolchain > /dev/null 2>&1
	    EXIT_STATUS=$?
	    if ! (exit $EXIT_STATUS) ; then
		echo "Warning: Failed to install devtoolset-11-toolchain" >&2
	    fi
	    ;;
	*) # Other RHEL based distros
	    yum -y install gcc-toolset-11 > /dev/null 2>&1
	    EXIT_STATUS=$?
	    if ! (exit $EXIT_STATUS) ; then
		echo "Warning: Failed to install gcc-toolset-11" >&2
	    fi
	    ;;
    esac
}

############## main ##############

if [ "$(id -u)" -eq 0 ]; then
    echo "Running with superuser privileges..."
fi
if [ ! -z "${BASTIONLAB_BUILD_AS_ROOT}" ]; then
    echo "Environmental variable for building server as root is set!"
fi

# Build as user
if [ "$(id -u)" -ne 0 ] || [ ! -z "${BASTIONLAB_BUILD_AS_ROOT}" ]; then
    
    # For Debian based distros
    if [ -f "/etc/debian_version" ] ; then
	verify_deps 'dpkg -s' "${deb_dependencies[@]}"
	EXIT_STATUS=$?
	if ! (exit $EXIT_STATUS) ; then
	    if [ -z "${BASTIONLAB_BUILD_AS_ROOT}" ]; then
		sudo $0
	    else
		install_deb_deps
	    fi
	    EXIT_STATUS=$?
	    if ! (exit $EXIT_STATUS) ; then
		exit $EXIT_STATUS
	    fi
	fi
	# Install cargo and torch
	install_common "__torch_cxx11_url__"
	# Build server
	LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all
	
    # For RHEL based distros    
    elif [ -f "/etc/redhat-release" ] ; then
	verify_deps 'yum list installed' "${rhel_dependencies[@]}"
	EXIT_STATUS=$?
	case "$(cat /etc/centos-release | awk '{print $1}')" in
	    "CentOS") # CentOS based distros
		verify_deps 'yum list installed' devtoolset-11-toolchain
		;;
	    *) # Other RHEL based distros
		verify_deps 'yum list installed' gcc-toolset-11
		;;
	esac
	if ! (exit $EXIT_STATUS) || ! (exit $?) ; then
	    if [ -z "${BASTIONLAB_BUILD_AS_ROOT}" ]; then
		sudo $0
	    else
		install_rhel_deps
	    fi
	    EXIT_STATUS=$?
	    if ! (exit $EXIT_STATUS) ; then
		exit $EXIT_STATUS
	    fi
	fi
	case "$(cat /etc/centos-release | awk '{print $1}')" in
	    "CentOS") # CentOS based distros
		# Install cargo and torch
		install_common "__torch_url__"
		# Build server
		scl enable devtoolset-11 'LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all' \
		    || LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all
		;;
	    *) # Other RHEL based distros
		# Install cargo and torch
		install_common "__torch_cxx11_url__"
		# Build server
		scl enable gcc-toolset-11 'LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all' \
		    || LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make all
		;;
	esac
	# Install cargo and torch
    else
	unrecognized_distro
    fi
    exit $?
else
    # Install dependencies as superuser
    echo "Installing dependencies..."
     if [ -f "/etc/debian_version" ] ; then
	 install_deb_deps
     elif [ -f "/etc/redhat-release" ] ; then
	 install_rhel_deps
     else
	 unrecognized_distro
     fi
fi
