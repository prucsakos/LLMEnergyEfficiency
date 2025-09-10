vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --trust-remote-code \
  --dtype bfloat16 \
  --mamba_ssm_cache_dtype float32 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.5