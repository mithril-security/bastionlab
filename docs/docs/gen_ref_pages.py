import markdownify
import os

PATH = "./docs/docs/resources/bastionlab/"

# create html
for files in os.listdir(PATH):
    if files.endswith(".html"):
        with open(PATH + files, "r") as f:
            html = f.read()
            h = markdownify.markdownify(html)
            with open(PATH + files.replace(".html", ".md"), "w", encoding="utf-8") as f:
                f.write(h)
