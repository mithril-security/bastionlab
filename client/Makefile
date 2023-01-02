.PHONY: all venv build dev-install test clean

VENV := env
BASTIONLAB_VERSION := $(shell cat src/bastionlab/version.py | cut -d '"' -f 2)

all: build

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

venv:
	python3 -m venv $(VENV)

dist/bastionlab-$(BASTIONLAB_VERSION)-py3-none-any.whl: $(VENV)/bin/activate
	./$(VENV)/bin/pip install build
	./$(VENV)/bin/python3 -m build . --wheel

build: $(VENV)/bin/activate
	./$(VENV)/bin/pip install build
	./$(VENV)/bin/python3 -m build . --wheel

venv/lib/python3.10/site-packages/bastionlab-$(BASTIONLAB_VERSION).dist-info/direct_url.json: $(VENV)/bin/activate
	./$(VENV)/bin/pip install -e .

dev-install: $(VENV)/bin/activate
	./$(VENV)/bin/pip install -e .

test: $(VENV)/bin/activate venv/lib/python3.10/site-packages/bastionlab-$(BASTIONLAB_VERSION).dist-info/direct_url.json
	./$(VENV)/bin/pip install -r ./tests/requirements.txt
	pytest --cov .

clean:
	rm -rf $(VENV)
	rm -rf build
	rm -rf dist
	rm -rf src/bastionlab.egg-info
	find src/bastionlab/pb -type f -name '*.py' ! -name '__init__.py' -delete
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
