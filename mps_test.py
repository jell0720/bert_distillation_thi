import torch

# 檢查是否支援 MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")  # 使用 macOS M2 GPU
else:
    device = torch.device("cpu")  # 回退到 CPU

print(f"使用的裝置: {device}")

# 建立測試 Tensor
x = torch.rand(3, 3).to(device)
print(x)