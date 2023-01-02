import subprocess
import urllib.request
import zipfile
import io
import os
from .version import __version__ as app_version
from urllib.error import HTTPError
from os import path
from subprocess import Popen

TORCH_VERSION = "1.13.1"


class BastionLabServer:
    """Popen object wrapper
    Args:
        process (Popen): Process object returned by subprocess.popen
    """

    def __init__(self, process):
        self.process = process

    def getProcess(self):
        return self.process


class NotFoundError(Exception):
    """This exception is raised when there was an error opening an URL.
    Args:
        Args:
        message (str): Error message.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    pass


def extract_zip(data):
    with zipfile.ZipFile(io.BytesIO(data)) as zip:
        zip.extractall("./")


def handle_download(path_str: str, url: str, name: str, error_msg: str):
    if path.exists(path_str) is False:
        print("Downloading {}...".format(name))
        try:
            response = urllib.request.urlopen(url)
            extract_zip(response.read())
        except HTTPError as e:
            raise NotFoundError(
                "{}. Exact error code: {}".format(error_msg, e.code)
            ) from None
    else:
        print("{} already installed".format(name))


def tls_certificates():
    if path.exists("./bin/tls") is False:
        print("Generating TLS certificates...")
        os.system(
            "mkdir -p bin/tls && openssl req -newkey rsa:2048 -nodes -keyout bin/tls/host_server.key -x509 -days 365 -out bin/tls/host_server.pem -subj "
            "/C=FR/CN=bastionlab-server"
            " >/dev/null 2>&1"
        )
    else:
        print("TLS certificates already generated")


def start_server(bastionlab_path: str, libtorch_path: str) -> BastionLabServer:
    os.chmod(bastionlab_path, 0o755)
    os.chdir(os.getcwd() + "/bin")
    os.environ["LD_LIBRARY_PATH"] = libtorch_path + "/lib"
    os.environ["DISABLE_AUTHENTICATION"] = "1"
    process = subprocess.Popen([bastionlab_path], env=os.environ)
    os.chdir("..")
    print("Bastionlab server is now running on port 50056")
    srv = BastionLabServer(process)
    return srv


def stop(srv: BastionLabServer) -> bool:
    """Stop BastionLab server.
    This method will kill the running server, if the provided BastionLabServer object is valid.

    Args:
        srv (BastionLabServer): The running process of the server.

    Return:
        bool, determines if the process was successful or not.

    Raises:
        None
    """

    if (
        srv is not None
        and srv.getProcess() is not None
        and srv.getProcess().poll() is None
    ):
        print("Stopping BastionLab's server...")
        srv.getProcess().kill()
        return True
    else:
        print("BastionLab's server already stopped")
        return False


def start() -> BastionLabServer:
    """Start BastionLab server.
    The method will download BastionLab's server binary, then download a specific version of libtorch.
    The server will then run, as a subprocess, allowing to run the rest of your Google Colab/Jupyter Notebook environment.

    Args:
        None

    Return:
        BastionLabServer object, the process of the running server.

    Raises:
        NotFoundError: Will be raised if one of the URL the wheel will try to access is invalid. This might mean that either there is no available binary of BastionLab's server, or the currently used libtorch version was removed on torch's servers.
        Other exceptions might be raised by zipfile or urllib.request.
    """
    libtorch_path = os.getcwd() + "/libtorch"
    bastionlab_path = os.getcwd() + "/bin/bastionlab"
    bastion_url = f"https://github.com/mithril-security/bastionlab/releases/download/v{app_version}/bastionlab-{app_version}-linux.zip"
    libtorch_url = f"https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-{TORCH_VERSION}%2Bcpu.zip"  # Required to be pre cpp11, colab does not support the cpp11 version
    handle_download(
        bastionlab_path,
        bastion_url,
        f"BastionLab server (version {app_version})",
        "The release might not be available yet for the current version",
    )
    handle_download(
        libtorch_path,
        libtorch_url,
        f"Libtorch (version {TORCH_VERSION})",
        "Unable to download Libtorch",
    )
    tls_certificates()
    process = start_server(bastionlab_path, libtorch_path)
    return process
