
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
uv pip install falshinfer-python
# uv pip uninstall pynvml
uv pip install nvidia-ml-py
uv pip install timm
uv pip install deepspeed
uv pip install bitsandbytes
uv pip install xformers
uv pip install scikit-learn
uv pip install datasets transformers
uv pip install accelerate
uv pip install wandb
uv pip install huggingface_hub[cli]
