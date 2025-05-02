import torch
print(torch.version.cuda)
print(torch.cuda.is_available())  # 檢查CUDA是否可用
print(torch.cuda.get_device_capability(0))