import os
from collections import OrderedDict

# Create an empty dictionary
docs = {}

# Set the root directory
root_dir = "docs/docs/resources/bastionlab/"

# Key names to delete from the dictionary
keys_to_delete = ["version"]

# files to exclude from the dictionary
files_to_exclude = [
    "bastionlab_pb2.md",
    "bastionlab_polars_pb2.md",
    "bastionlab_torch_pb2.md",
]


def process_directory(dir):
    subdocs = {}
    # Iterate through all files and directories in the subdirectory
    for item in os.listdir(dir):
        # If the item is a file
        if os.path.isfile(os.path.join(dir, item)):
            # Split the file name and extension
            file_name, file_ext = os.path.splitext(item)
            # Add the file to the dictionary with the full path as the value
            subdocs[file_name] = os.path.join(dir, item)
        # If the item is a directory
        elif os.path.isdir(os.path.join(dir, item)):
            # Recursively process the subdirectory
            subdocs[item] = process_directory(os.path.join(dir, item))
    sorted_subdocs = {key: subdocs[key] for key in sorted(subdocs.keys())}
    # Swap the first key and value with the keys and value of the key index
    ordered_sorted_subdocs = OrderedDict(sorted_subdocs)
    ordered_sorted_subdocs.move_to_end("index", last=False)
    # Convert the OrderedDict back to a dictionary
    sorted_subdocs = dict(ordered_sorted_subdocs)
    # Return the dictionary
    return sorted_subdocs


def delete_keys(d, keys_to_delete):
    # Iterate through the dictionary
    for key in list(d.keys()):
        # If the key is in the list of keys to delete
        if (
            key in keys_to_delete
            or isinstance(d[key], str)
            and d[key].split("/")[-1] in files_to_exclude
        ):
            # Delete the key-value pair
            d.pop(key, None)
        # If the value is a dictionary
        elif isinstance(d[key], dict):
            # Recursively process the subdictionary
            delete_keys(d[key], keys_to_delete)


def construct_tree(dictionary, level=0, string="", parent_dir=""):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            # Delete the first '/docs' from the path
            string += " " * 4 * level + "- bastionlab." + parent_dir + key + ":" + "\n"
            string += construct_tree(value, level + 1, "", parent_dir + key + ".")
        else:
            if key == "index":
                string += "    " * level + "- " + '"' + value[5:] + '"' + "\n"
            else:
                string += (
                    "    " * level
                    + "- bastionlab."
                    + parent_dir
                    + key
                    + ": "
                    + '"'
                    + value[5:]
                    + '"'
                    + "\n"
                )

    return string


def align_tabs(tree, tabs):
    tab_lines = tree.splitlines()
    for j in range(len(tab_lines)):
        tab_lines[j] = tabs * " " + tab_lines[j]
    tree = "\n".join(tab_lines)
    tree += "\n"
    return tree


if __name__ == "__main__":
    structure = process_directory(root_dir)
    delete_keys(structure, keys_to_delete)
    tree = construct_tree(structure)
    with open("mkdocs.yml", "r") as file:
        data = file.readlines()
        for i, line in enumerate(data):
            if "Submodules:" in line or "Sub-modules:" in line:
                tabs = line.count(" ") + 3
                tree = align_tabs(tree, tabs)
                data.insert(i + 1, tree)
                break
        with open("mkdocs.yml", "w") as file:
            file.writelines(data)
