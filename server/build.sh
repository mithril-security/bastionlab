#!/usr/bin/env bash
#
# Depends on:
#   bastionlab/client/src/bastionlab/version.py
# Dependents:
#   bastionlab/server/Dockerfile.gpu.sev
#
# Env variable "BASTIONLAB_BUILD_AS_ROOT" can be useful
#   when building in a docker image as root.
# Env variable "BASTIONLAB_CPP11" is for compile with C++11
#   (installing if missing).
# Env "INSTALL_RUST_OPT" are the installer options for rustup
#   (To choose the default host, toolchain, profile, ...)
# Env variable "LIBTORCH" is the path of libtorch.
# Env variable "CUDA" is the path of cuda.

declare -a deb_dependencies=(
    [0]=build-essential
    [1]=libssl-dev
    [2]=pkg-config
    [3]=curl
    [4]=unzip
    [5]=python3
    [6]=python3-pip
    [7]=python3-venv
)

declare -a deb_optionals=(
    [0]=lsb-release
    [1]=software-properties-common
    [2]=gcc-11
    [3]=g++-11
    [4]=cpp-11
)

declare -a rhel_dependencies=(
    [0]=python3
    [1]=python3-pip
    [2]=make
    [3]=gcc
    [4]=gcc-c++
    [5]=zip
    [6]=openssl-devel
    [7]=openssl
    [8]=python3-virtualenv
)

declare -a arch_dependencies=(
    [0]=python3
    [1]=python-pip
    [2]=python-virtualenv
    [3]=make
    [4]=gcc
    [5]=unzip
    [6]=openssl
)

unrecognized_distro()
{
    echo "Unrecognized linux version, needs manual installation, check the documentation:">&2
    echo "https://bastionlab.readthedocs.io/en/latest/docs/getting-started/installation/" >&2
    exit 1
}

failed_toolchain()
{
    echo "Warning [❗]: Failed to recognize the version codename to install C++11" >&2
    echo "Trying running the script without the CPP11 flag or install C++11 manually" >&2
    return 1
}

install_common()
{
    cd $(dirname $(pwd))

    # Libtorch installation
    if [ -z "${LIBTORCH}" ] && [ ! -d "libtorch" ] ; then
	
	if [ "$(id -u)" -eq 0 ]; then
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
    elif [ -z "${LIBTORCH}" ]; then
	echo "libtorch.zip is already installed at $(pwd)/libtorch"
    else
	echo "libtorch is already set at LIBTORCH=${LIBTORCH}"
    fi

    # Environment variables
    # LIBTORCH
    if [ -z "${LIBTORCH}" ]; then
	echo "LIBTORCH environment variable is not set, using: $(pwd)/libtorch"
	export LIBTORCH=$PWD/libtorch
    else
	echo "LIBTORCH environment variable is already set to: ${LIBTORCH}"
    fi
    # CUDA
    if [ -z "${CUDA}" ]; then
	echo "CUDA environment variable is not set"
	if [ -d "/usr/local/cuda" ]; then
	    echo "Using: /usr/local/cuda"
	    export CUDA="/usr/local/cuda"
	fi
    else
	echo "CUDA environment variable is already set to: ${CUDA}"
    fi

    # Rustup installation
    command -v cargo > /dev/null 2>&1
    EXIT_STATUS=$?
    if ! (exit $EXIT_STATUS) ; then
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/rustup.sh
	# Installation with options
	if [ ! -z "${INSTALL_RUST_OPT}" ]; then
	    sh /tmp/rustup.sh -y "${INSTALL_RUST_OPT}"
	else
	    sh /tmp/rustup.sh -y
	fi
	EXIT_STATUS=$?
	if ! (exit $EXIT_STATUS) ; then
	    exit $EXIT_STATUS
	fi
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
	    echo "You have missing packages, installing them..." >&2
	    return $EXIT_STATUS
	else
	    echo "[✔️ ]" $package
	fi
    done
    return $EXIT_STATUS
}

# Debian-based C++11 installation
install_deb_opt()
{
    # Configuring repositories
    apt-get -y install software-properties-common
    command -v add-apt-repository > /dev/null 2>&1
    EXIT_STATUS=$?
    if ! (exit $EXIT_STATUS) ; then
	echo "Adding toolchain repository to /etc/apt/sources.list.d/toolchain.list..."
	VRELEASE=$(. /etc/os-release; echo $UBUNTU_CODENAME)
	if [ -z "${VRELEASE}" ]; then
	    VRELEASE=$(. /etc/os-release; echo $VERSION_CODENAME)
	fi
	if [ -z "${VRELEASE}" ]; then
	    apt-get -y install lsb-release
	    VRELEASE=$(lsb_release -c | awk '{print $2}')
	fi
	if [ ! -z "${VRELEASE}" ]; then

	    # Adding toolchain repository manually
	    DEB_TOOLCHAIN="https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu ${VRELEASE} main"
	    echo "deb" $DEB_TOOLCHAIN | tee -a /etc/apt/sources.list.d/toolchain.list
	    echo "deb-src" $DEB_TOOLCHAIN | tee -a /etc/apt/sources.list.d/toolchain.list
	    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F
	    EXIT_STATUS=$?
	    apt-get update
	    if ! (exit $EXIT_STATUS)  || ! (exit $?) ; then
		rm -rf /etc/apt/sources.list.d/toolchain.list
		return failed_toolchain
	    fi
	else
	    return failed_toolchain
	fi
    else
	add-apt-repository -y ppa:ubuntu-toolchain-r/test
	EXIT_STATUS=$?
	if ! (exit $EXIT_STATUS) ; then
	    return failed_toolchain
	fi
    fi
    echo "[✔️ ] Success adding toolchain repository"
}

# Debian-based dependencies installation
install_deb_deps()
{
    apt-get update
    apt-get -y upgrade
    apt-get -y install "${deb_dependencies[@]}"
    if [ ! -z "${BASTIONLAB_CPP11}" ]; then
	install_deb_opt
	apt-get -y install "${deb_optionals[@]:2}"
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
    if [ ! -z "${BASTIONLAB_CPP11}" ]; then
	case "$(cat /etc/centos-release | awk '{print $1}')" in
	    "CentOS") # CentOS based distros
		yum -y install devtoolset-11-toolchain > /dev/null 2>&1
		EXIT_STATUS=$?
		if ! (exit $EXIT_STATUS) ; then
		    echo "Warning [❗]: Failed to install devtoolset-11-toolchain" >&2
		fi
		;;
	    *) # Other RHEL based distros
		yum -y install gcc-toolset-11 > /dev/null 2>&1
		EXIT_STATUS=$?
		if ! (exit $EXIT_STATUS) ; then
		    echo "Warning [❗]: Failed to install gcc-toolset-11" >&2
		fi
		;;
	esac
    fi
}

# Arch-based dependencies installation
install_arch_deps()
{
    pacman -Syyu --noconfirm
    pacman-db-upgrade
    pacman -S --noconfirm "${arch_dependencies[@]}"
}

missing_deps()
{
    install_fn=$1
    if [ "$(id -u)" -ne 0 ]; then
	command -v sudo > /dev/null 2>&1
	EXIT_STATUS=$?
	if ! (exit $EXIT_STATUS) ; then
	    echo "It seems sudo is not installed and there are dependencies missing, to install them:" >&2
	    echo "Run this script without BASTIONLAB_BUILD_AS_ROOT set and with superuser privileges" >&2
	    echo "Example: su -c ./build.sh && ./build.sh # To install and then build"
	    exit $EXIT_STATUS
	else
	    sudo $0
	fi
    else
	$install_fn
    fi
    EXIT_STATUS=$?
    if ! (exit $EXIT_STATUS) ; then
	exit $EXIT_STATUS
    fi
}

############## main ##############

if [ "$(id -u)" -eq 0 ]; then
    echo "Running with superuser privileges..."
fi
if [ ! -z "${BASTIONLAB_BUILD_AS_ROOT}" ]; then
    echo "Environmental variable for building server as root is set!"
fi
if [ ! -z "${BASTIONLAB_CPP11}" ]; then
    echo "Environmental variable for building server with C++11 is set!"
fi

# Build as user
if [ "$(id -u)" -ne 0 ] || [ ! -z "${BASTIONLAB_BUILD_AS_ROOT}" ]; then
    
    # For Debian based distros
    if [ -f "/etc/debian_version" ] ; then

	# Verifying dependencies
	verify_deps 'dpkg -s' "${deb_dependencies[@]}"
	EXIT_STATUS=$?
	EXIT_STATUS2=0
	if [ ! -z "${BASTIONLAB_CPP11}" ]; then
	    verify_deps 'dpkg -s' "${deb_optionals[@]}"
	    EXIT_STATUS2=$?
	fi

	# If dependencies missing, installing them
	if ! (exit $EXIT_STATUS) || ! (exit $EXIT_STATUS2) ; then
	    missing_deps install_deb_deps
	fi
	
	# Install cargo and torch
	install_common "__torch_cxx11_url__"
	
	# Build server
	make all
	
    # For RHEL based distros    
    elif [ -f "/etc/redhat-release" ] ; then

	#Verifying dependencies
	verify_deps 'yum list installed' "${rhel_dependencies[@]}"
	EXIT_STATUS=$?
	EXIT_STATUS2=0
	if [ ! -z "${BASTIONLAB_CPP11}" ]; then
	    case "$(cat /etc/centos-release | awk '{print $1}')" in
		"CentOS") # CentOS based distros
		    verify_deps 'yum list installed' devtoolset-11-toolchain
		    ;;
		*) # Other RHEL based distros
		    verify_deps 'yum list installed' gcc-toolset-11
		    ;;
	    esac
	    EXIT_STATUS2=$?
	fi

	# If dependencies missing, installing them
	if ! (exit $EXIT_STATUS) || ! (exit $EXIT_STATUS2) ; then
	    missing_deps install_rhel_deps
	fi
	
	case "$(cat /etc/centos-release | awk '{print $1}')" in
	    "CentOS") # CentOS based distros

		# Install cargo and torch
		install_common "__torch_url__"

		# Build server
		if [ ! -z "${BASTIONLAB_CPP11}" ]; then
		    scl enable devtoolset-11 'make all' ||  make all
		else
		    make all
		fi
		;;
	    *) # Other RHEL based distros

		# Install cargo and torch
		install_common "__torch_cxx11_url__"

		# Build server
		if [ ! -z "${BASTIONLAB_CPP11}" ]; then
		    scl enable gcc-toolset-11 'make all' || make all
		else
		    make all
		fi
		;;
	esac

    elif [ -f "/etc/arch-release" ] ; then
	
	# Verifying dependencies
	verify_deps 'pacman -Qi' "${arch_dependencies[@]}"
	EXIT_STATUS=$?

	# If dependencies missing, installing them
	if ! (exit $EXIT_STATUS) ; then
	    missing_deps install_arch_deps
	fi
	
	# Install cargo and torch
	install_common "__torch_cxx11_url__"
	export OPENSSL_INCLUDE_DIR='/usr/include/openssl/'
	export OPENSSL_LIB_DIR='/usr/lib/'
	
	# Build server
	make all

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
    elif [ -f "/etc/arch-release" ] ; then
	install_arch_deps
    else
	unrecognized_distro
    fi
    EXIT_STATUS=$?
    if ! (exit $EXIT_STATUS) ; then
	exit $EXIT_STATUS
    fi
fi
