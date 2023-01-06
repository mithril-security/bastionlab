import os
import nbformat


def remove_no_execute_cells(nb):
    index = None

    while True:
        # find index for the cell with the injected params
        for i, c in enumerate(nb.cells):
            cell_tags = c.metadata.get("tags")
            if cell_tags:
                if "no_execute" in cell_tags:
                    index = i
                    break
        else:
            # no more cells with the no_execute tag were found
            break

        # remove cell
        if index is not None:
            nb.cells.pop(index)


# create the output directory if it does not exist
if not os.path.exists("converted"):
    os.makedirs("converted")

# traverse all subdirectories and find all .ipynb files
for root, dirs, files in os.walk("."):
    for filename in files:
        if filename.endswith(".ipynb"):
            # read in the notebook
            nb = nbformat.read(os.path.join(root, filename), as_version=4)

            # remove cells with the no_execute tag
            remove_no_execute_cells(nb)

            # save modified notebook to the converted directory
            nbformat.write(nb, os.path.join("converted", filename))
