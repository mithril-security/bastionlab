import errno
import subprocess
import os
import sys
import socket
from time import time as now, sleep
import atexit

server_process = None
server_name = "bastionlab"


def launch_server():
    global server_process

    sock = None

    try:
        if (
            server_process is None
            and os.getenv("BASTIONLAB_TEST_NO_LAUNCH_SERVER") is None
        ):
            server_dir = os.path.join(os.path.dirname(__file__), "../server")
            bin_dir = os.path.join("../server/", "bin")
            server_process = subprocess.Popen(
                [f"{server_name}"],
                cwd=bin_dir,
                executable=os.path.join(bin_dir, f"{server_name}"),
                stdout=sys.stdout,
                stderr=sys.stderr,
                stdin=subprocess.DEVNULL,
            )

        # block until server ready (port open)
        end = now() + 30  # 30s timeout
        success = False
        while True:
            if now() > end:
                raise Exception("Server startup timed out")

            try:
                sock = socket.socket()
                sock.settimeout(end - now())
                sock.connect(("localhost, 50056"))
                success = True
                sock.close()
                break
            except socket.error as err:
                if err.errno != errno.ECONNREFUSED:
                    raise
                sock.close()
                sleep(0.1)

        if not success:
            raise Exception("Server startup timed out")

        print("[TESTS] The server is running")

    except Exception:
        if sock is not None:
            sock.close()

        if server_process is not None:
            server_process.terminate()
            server_process.wait()
        raise


def close_server():
    global server_process

    if server_process is None:
        return

    server_process.terminate()
    server_process.wait()
    server_process = None

    print("[TESTS] The server is stopped")


atexit.register(close_server)
