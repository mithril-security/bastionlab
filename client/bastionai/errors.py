import grpc  # type: ignore [import]


def check_rpc_exception(rpc_error):
    """Nicely displays a gRPC protocol error."""
    if rpc_error.code() == grpc.StatusCode.CANCELLED:
        return f"Cancelled GRPC call: code={rpc_error.code()} message={rpc_error.details()}"

    elif rpc_error.code() == grpc.StatusCode.UNAVAILABLE:
        return f"Failed to connect to GRPC server: code={rpc_error.code()} message={rpc_error.details()}"

    elif rpc_error.code() == grpc.StatusCode.UNIMPLEMENTED:
        return f"Incompatible client/server versions, code={rpc_error.code()} message={rpc_error.details()}"

    elif rpc_error.code() == grpc.StatusCode.FAILED_PRECONDITION:
        return f"Attestation is not available. Running in Simulation Mode, code={rpc_error.code()} message={rpc_error.details()}"
    else:
        return (
            f"Received RPC error: code={rpc_error.code()} message={rpc_error.details()}"
        )


def check_socket_exception(socket_error):
    """Nicely displays a socket error."""
    if len(socket_error.args) >= 2:
        error_code = socket_error.args[0]
        error_message = socket_error.args[1]
        return f"Failed To connect to the server due to Socket error : code={error_code} message={error_message}"

    elif len(socket_error.args) == 1:
        error_message = socket_error.args[0]
        return f"Failed To connect to the server due to Socket error : message={error_message}"

    else:
        return "Failed To connect to the server due to Socket error "
