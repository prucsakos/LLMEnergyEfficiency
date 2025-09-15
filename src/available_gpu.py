import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print("  Capability:", torch.cuda.get_device_capability(i))
        print("  Memory (GB):", round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2))
else:
    print("No NVIDIA GPU detected by PyTorch")
