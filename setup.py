import torch
import torchvision
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

# 간단한 텐서 연산 테스트
x = torch.randn(2, 3, 224, 224)
print("Test tensor shape:", x.shape)
print("Test envrionment verified")