import markdownify
import os
from glob import iglob

PATH = "./docs/docs/resources/bastionlab/**/*"

# Create file list
file_list = [f for f in iglob(PATH, recursive=True) if os.path.isfile(f)]

for file in file_list:
    if file.endswith(".html"):
        with open(file, "r", encoding="utf-8") as f:
            html = f.read()
            h = markdownify.markdownify(html)
            # delete the first 31 lines of the file
            split = h.splitlines(True)[31:]
            h = "".join(split)
            with open(file.replace(".html", ".md"), "w", encoding="utf-8") as f:
                f.write(h)
