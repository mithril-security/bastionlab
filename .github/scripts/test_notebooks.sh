#!/bin/bash

set -e

echo "Removing cells..."
python3 .github/scripts/remove_cells.py
echo "Downloading datasets..."
wget 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv' -O tmp/titanic.csv
wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip' -O tmp/smsspamcollection.zip
unzip tmp/smsspamcollection.zip -d tmp
# wget 'https://raw.githubusercontent.com/rinbaruah/COVID_preconditions_Kaggle/master/Data/covid.csv' -O tmp/covid.csv 
cp tmp/titanic.csv tmp/train.csv

rm -f ./tmp/resnet_example_notebook.ipynb
rm -f ./tmp/distilbert_example_notebook.ipynb
rm -f ./tmp/quick-tour.ipynb
rm -f ./tmp/fraud_detection.ipynb
rm -f ./tmp/authentication.ipynb

OUTPUT=""
ERRORS=0
N=0

for file in ./tmp/*.ipynb; do 
    sed -i 's/"!pip install bastionlab"/""/g' $file
    sed -i 's/"srv = bastionlab_server.start()"/""/g' $file
    sed -i 's/"bastionlab_server.stop(srv)"/""/g' $file
    echo "Running $file..."
    if jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=600 --output-dir=./tmp $file; then
        OUTPUT+="... $file: OK!\n"
    else
        OUTPUT+="... $file: ERROR!\n"
        ERRORS=$[ $ERRORS + 1 ]
    fi
    N=$[ $N + 1 ]
done

echo
echo "Summary:"
echo -e $OUTPUT
echo "$ERRORS errors out of $N notebooks!"

if [ $ERRORS -gt 0 ]; then
    exit 1
fi
