#! /bin/bash

# Remove the cells tags of all jupyter notebooks and save them inplace

# Create a list of all the files in the docs/docs/resources/bastionlab directory
find . -type f -name "*.ipynb" | for file in $(cat); do
    jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --inplace $file
done
