import torch

print("Is CUDA available:", torch.cuda.is_available())
print("Available devices:", torch.cuda.device_count())
