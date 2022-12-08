import subprocess
import urllib.request
import zipfile
import io
import os
from .version import __version__ as app_version
from urllib.error import HTTPError
from os import path
from subprocess import Popen


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
        os.system(
            "mkdir -p bin/tls && openssl req -newkey rsa:2048 -nodes -keyout bin/tls/host_server.key -x509 -days 365 -out bin/tls/host_server.pem -subj "
            "/C=FR/CN=bastionlab-server"
            ""
        )


def start_server(bastionlab_path: str, libtorch_path: str) -> Popen:
    os.chmod(bastionlab_path, 0o755)
    os.chdir(os.getcwd() + "/bin")
    os.environ["LD_LIBRARY_PATH"] = libtorch_path + "/lib"
    process = subprocess.Popen([bastionlab_path], env=os.environ)
    os.chdir("..")
    return process


def stop(process: Popen) -> bool:
    if process is not None and process.poll() is not None:
        process.terminate()
        return True
    else:
        return False


def start() -> Popen:
    libtorch_path = os.getcwd() + "/libtorch"
    bastionlab_path = os.getcwd() + "/bin/bastionlab"
    bastion_url = "https://github.com/mithril-security/bastionlab/releases/download/v{}/bastionlab-{}-linux.zip".format(
        app_version, app_version
    )
    libtorch_url = "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.12.1%2Bcpu.zip"  # Required to be pre cpp11, colab does not support the cpp11 version
    handle_download(
        bastionlab_path,
        bastion_url,
        "BastionLab server (version {})".format(app_version),
        "The release might not be available yet for the current version",
    )
    handle_download(
        libtorch_path,
        libtorch_url,
        "Libtorch (version 1.12.1)",
        "Unable to download Libtorch",
    )
    tls_certificates()
    process = start_server(bastionlab_path, libtorch_path)
    return process
