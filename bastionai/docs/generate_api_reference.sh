pip install pdoc3
rm -rf docs/resources
pdoc --html --skip-errors --template-dir docs/pdoc_template -o docs/resources client/bastionai
sed -i '/<p>Generated by <a href="https:\/\/pdoc3.github.io\/pdoc" title="pdoc: Python API documentation generator"><cite>pdoc<\/cite> 0.10.0<\/a>.<\/p>/d' docs/resources/bastionai/*.html docs/resources/bastionai/*/*.html