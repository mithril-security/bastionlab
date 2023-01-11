# Configuration storage for the client instances of BastionLab{Torch, Polars}.
# This helps makes the instances accessible from everywhere in the project.
CONFIG = {"torch_client": None, "polars_client": None}


def get_client(client: str):
    client_obj = CONFIG.get(client)

    if client_obj == None:
        raise Exception(f"{client} is not initialized.")

    return client_obj
