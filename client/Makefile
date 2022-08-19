VENV := venv

all: venv

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

venv: $(VENV)/bin/activate

build: venv
	./$(VENV)/bin/pip install build
	./$(VENV)/bin/python3 -m build . --wheel

install: venv
	./$(VENV)/bin/pip install -e .

test: venv
	./$(VENV)/bin/pip install -r ./tests/requirements.txt
	pytest .

clean:
	rm -rf $(VENV)
	rm -rf build
	rm -rf dist
	rm -rf src/bastionai/pb
	rm -rf src/bastionai.egg-info
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

.PHONY: all venv build install clean test
