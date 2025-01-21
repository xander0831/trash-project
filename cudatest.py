import torch

# 检查CUDA是否可用
print(f"CUDA is available: {torch.cuda.is_available()}")

# 检查当前设备的CUDA版本
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
