import torch

# 检查是否有可用的 CUDA 设备
if torch.cuda.is_available():
    # 在 GPU 上运行张量运算
    device = torch.device("cuda")
    a = torch.randn(1000, 1000).to(device)
    b = torch.randn(1000, 1000).to(device)
    c = torch.matmul(a, b)

    print("CUDA is available and test passed!")
else:
    print("CUDA is not available.")