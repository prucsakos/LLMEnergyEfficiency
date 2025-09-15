
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
uv pip install falshinfer-python
uv pip uninstall pynvml
uv pip install nvidia-ml-py