# src/layers/depthwise_separable_conv.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution 구현
    
    MobileNet의 핵심 구성 요소로, 다음 두 단계로 구성:
    1. Depthwise Convolution: 각 입력 채널에 대해 독립적으로 spatial convolution 수행
    2. Pointwise Convolution: 1x1 convolution으로 채널 간 정보 결합
    
    계산량 비교:
    - Standard Conv: H × W × C_in × C_out × K × K
    - Depthwise Separable: H × W × C_in × K × K + H × W × C_in × C_out
    - 감소율: 1/C_out + 1/K²
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = False,
        use_bn: bool = True,
        activation: str = 'relu6'
    ):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        # Auto padding for 'same' convolution
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
            
        # 1. Depthwise Convolution
        # groups=in_channels는 각 입력 채널이 독립적으로 처리됨을 의미
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # 출력 채널 수 = 입력 채널 수
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # 핵심: 각 채널을 독립적으로 처리
            bias=bias
        )
        
        # Depthwise 후 BatchNorm
        self.bn1 = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        
        # Activation after depthwise
        self.act1 = self._get_activation(activation)
        
        # 2. Pointwise Convolution (1x1 conv)
        # 채널 간 정보를 결합하여 출력 채널 수를 조정
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  # 1x1 convolution
            stride=1,
            padding=0,
            bias=bias
        )
        
        # Pointwise 후 BatchNorm
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # Final activation
        self.act2 = self._get_activation(activation)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """활성화 함수 선택"""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'relu6':
            return nn.ReLU6(inplace=True)
        elif activation == 'swish' or activation == 'silu':
            return nn.SiLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        else:
            return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Depthwise convolution
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # 2. Pointwise convolution
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        return x
    
    def get_flops(self, input_size: Tuple[int, int, int]) -> dict:
        """FLOPS 계산 (이론적 비교용)"""
        C_in, H, W = input_size
        C_out = self.pointwise.out_channels
        K = self.depthwise.kernel_size[0]
        
        # Depthwise FLOPS
        depthwise_flops = H * W * C_in * K * K
        
        # Pointwise FLOPS
        pointwise_flops = H * W * C_in * C_out
        
        # Standard convolution FLOPS (비교용)
        standard_flops = H * W * C_in * C_out * K * K
        
        return {
            'depthwise_flops': depthwise_flops,
            'pointwise_flops': pointwise_flops,
            'total_flops': depthwise_flops + pointwise_flops,
            'standard_flops': standard_flops,
            'reduction_ratio': standard_flops / (depthwise_flops + pointwise_flops)
        }


class DepthwiseConv2d(nn.Module):
    """순수 Depthwise Convolution (분석용)"""
    
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = False,
        use_bn: bool = True,
        activation: str = 'relu6'
    ):
        super(DepthwiseConv2d, self).__init__()
        
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
            
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        
        self.bn = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        else:
            self.act = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class PointwiseConv2d(nn.Module):
    """순수 Pointwise Convolution (분석용)"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        use_bn: bool = True,
        activation: str = 'relu6'
    ):
        super(PointwiseConv2d, self).__init__()
        
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        else:
            self.act = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# 유틸리티 함수들
def compare_conv_methods(input_tensor: torch.Tensor, in_channels: int, out_channels: int):
    """Standard Conv vs Depthwise Separable Conv 비교"""
    
    # Standard Convolution
    standard_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )
    
    # Depthwise Separable Convolution
    ds_conv = DepthwiseSeparableConv2d(in_channels, out_channels)
    
    # Forward pass
    with torch.no_grad():
        standard_out = standard_conv(input_tensor)
        ds_out = ds_conv(input_tensor)
    
    # Parameter count
    standard_params = sum(p.numel() for p in standard_conv.parameters())
    ds_params = sum(p.numel() for p in ds_conv.parameters())
    
    # FLOPS 계산
    _, C, H, W = input_tensor.shape
    flops_info = ds_conv.get_flops((C, H, W))
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {standard_out.shape} (both methods)")
    print(f"\nParameters:")
    print(f"  Standard Conv: {standard_params:,}")
    print(f"  Depthwise Separable: {ds_params:,}")
    print(f"  Reduction: {standard_params/ds_params:.2f}x")
    print(f"\nFLOPS:")
    print(f"  Standard Conv: {flops_info['standard_flops']:,}")
    print(f"  Depthwise Separable: {flops_info['total_flops']:,}")
    print(f"  Reduction: {flops_info['reduction_ratio']:.2f}x")
    
    return standard_out, ds_out


if __name__ == "__main__":
    # 테스트 코드
    print("=== Depthwise Separable Convolution test ===")
    
    # 테스트 입력
    batch_size, in_channels, height, width = 1, 32, 224, 224
    x = torch.randn(batch_size, in_channels, height, width)
    
    # 모델 생성
    ds_conv = DepthwiseSeparableConv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        stride=1
    )
    
    # Forward pass
    output = ds_conv(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    
    # 비교 분석
    print("\n=== Standard Conv vs Depthwise Separable Conv ===")
    compare_conv_methods(x, in_channels=32, out_channels=64)