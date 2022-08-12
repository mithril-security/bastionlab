VENV := venv

all: venv

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt

venv: $(VENV)/bin/activate

src/bastionai/pb/remote_torch_pb2.py: venv
	mkdir -p src/bastionai/pb
	./$(VENV)/bin/python3 -m grpc_tools.protoc -I../server/protos --python_out=./src/bastionai/pb --grpc_python_out=./src/bastionai/pb ../server/protos/remote_torch.proto

build: src/bastionai/pb/remote_torch_pb2.py
	./$(VENV)/bin/pip install build
	./$(VENV)/bin/python3 -m build . --wheel

install: src/bastionai/pb/remote_torch_pb2.py
	./$(VENV)/bin/pip install -e .

clean:
	rm -rf $(VENV)
	rm -rf build
	rm -rf dist
	rm -rf src/bastionai/pb
	rm -rf src/bastionai.egg-info
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

.PHONY: all venv build install clean
