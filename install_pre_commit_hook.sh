#!/bin/bash

if ! pre-commit --version > /dev/null
then
    pip install pre-commit
fi

pre-commit autoupdate
pre-commit install