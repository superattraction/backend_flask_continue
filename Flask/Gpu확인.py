import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    print("GPU is available")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2  # Convert bytes to MB
    print(f"GPU Name: {gpu_name}")
    print(f"GPU Memory: {gpu_memory} MB")
else:
    print("GPU is not available")
