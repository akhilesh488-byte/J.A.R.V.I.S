import torch
assert torch.cuda.is_available(), "CUDA is not available"
device_id = torch.cuda.current_device()
gpu_name = torch.cuda.get_device_name(device_id)
print(f"CUDA Available: True | Device {device_id}: {gpu_name}")
