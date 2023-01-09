import os
from collections import OrderedDict

# Create an empty dictionary
docs = {}

# Set the root directory
root_dir = "docs/docs/resources/bastionlab/"


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
    return sorted_subdocs


def delete_keys(d, keys_to_delete):
    # Iterate through the dictionary
    for key in list(d.keys()):
        # If the key is in the list of keys to delete
        if key in keys_to_delete:
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
            string += "    " * level + "- bastionlab." + parent_dir + key + ": " + "\n"
            string += construct_tree(value, level + 1, "", parent_dir + key + ".")
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


if __name__ == "__main__":
    structure = process_directory(root_dir)
    sorted_dict = OrderedDict([key, value] for key, value in sorted(structure.items()))
    tree = construct_tree(sorted_dict)
    with open("mkdocs.yml", "r") as file:
        data = file.readlines()
        for i, line in enumerate(data):
            if "API Reference:" in line:
                tabs = line.count(" ")
                tab_lines = tree.splitlines()
                for j in range(len(tab_lines)):
                    tab_lines[j] = tabs * " " + tab_lines[j]
                tree = "\n".join(tab_lines)
                tree += "\n"
                data.insert(i + 1, tree)
                break
        with open("mkdocs.yml", "w") as file:
            file.writelines(data)
