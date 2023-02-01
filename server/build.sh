#!/usr/bin/env bash
#
# Depends on:
#   bastionlab/client/src/bastionlab/version.py
# Dependents:
#   bastionlab/.github/workflows/release_bin.yml

declare -a deb_dependencies=(
    [0]=software-properties-common
    [1]=build-essential
    [2]=patchelf
    [3]=libssl-dev
    [4]=pkg-config
    [5]=curl
    [6]=unzip
    [7]=python3
    [8]=python3-pip
    [9]=gcc-11
    [10]=g++-11
    [11]=cpp-11
    [12]=python3-venv
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
)

unrecognized_distro()
{
    echo "Unrecognized linux version, needs manual installation, check the documentation:">&2
    echo "https://bastionlab.readthedocs.io/en/latest/docs/getting-started/installation/" >&2
    return 1
}

install_common()
{
    cd $(dirname $(pwd))

    # Libtorch installation
    if [ ! -d "libtorch" ] ; then
	pip3 install --user requests
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
    if [ ! -d "$HOME/.cargo/bin" ]; then
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
    if [ "$(id -u)" -ne 0 ]; then
	for package in "${packages[@]}"; do
	    $checkcmd $package > /dev/null 2>&1
	    if [ "$(echo $?)" -ne 0 ]; then
		echo $package
		echo "You have missing packages to install, running as superuser" >&2
		break
	    else
		echo "[✔️ ]" $package
	    fi
	done
    fi
    return
}

# Build as user
if [ "$(id -u)" -ne 0 ] || [ ! -z "${BASTIONLAB_BUILD_AS_ROOT}" ]; then
    EXIT_STATUS=0
    
    # For Debian based distros
    if [ -f "/etc/debian_version" ] ; then
	verify_deps 'dpkg -s' "${deb_dependencies[@]}"
	EXIT_STATUS=$?
	if ! (exit $EXIT_STATUS) ; then
	    sudo $0
	    EXIT_STATUS=$?
	    if ! (exit $EXIT_STATUS) ; then
		break
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
	case "$(cat /etc/centos-release > /dev/null 2>&1 | awk '{print $1}')" in
	    "CentOS") # CentOS based distros
		verify_deps 'yum list installed' devtoolset-11-toolchain
		;;
	    *) # Other RHEL based distros
		verify_deps 'yum list installed' gcc-toolset-11
		;;
	esac
	if ! (exit $EXIT_STATUS) || ! (exit $?) ; then
	    sudo $0
	    if ! (exit $EXIT_STATUS) || ! (exit $?) ; then
		break
	    fi
	fi
	case "$(cat /etc/centos-release > /dev/null 2>&1 | awk '{print $1}')" in
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
    echo "The script must not be runned as superuser for building the server"
    echo "Installing dependencies..."
fi
# Install as superuser
if [ -f "/etc/debian_version" ] ; then
    # Dependencies installation
    apt-get update
    apt-get -y install software-properties-common
    add-apt-repository -y ppa:ubuntu-toolchain-r/test
    apt-get -y install "${deb_dependencies[@]:1}"
    update-alternatives \
        --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-11
elif [ -f "/etc/redhat-release" ] ; then
    # Dependencies installation
    yum -y install "${rhel_dependencies[@]}"
    case "$(cat /etc/centos-release > /dev/null 2>&1 | awk '{print $1}')" in
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

else
    unrecognized_distro
fi
