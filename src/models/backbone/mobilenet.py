# src/models/backbone/mobilenet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import math

# 우리가 만든 Depthwise Separable Conv 임포트
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.layers.depthwise_separable_conv import DepthwiseSeparableConv2d


class MobileNetV1Block(nn.Module):
    """
    MobileNet V1 기본 블록
    
    구조:
    1. Depthwise Separable Convolution
    2. Optional Stride for downsampling
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = 'relu6'
    ):
        super(MobileNetV1Block, self).__init__()
        
        self.block = DepthwiseSeparableConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileNetV1(nn.Module):
    """
    MobileNet V1 구현
    
    논문: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    
    특징:
    - Depthwise Separable Convolution 사용
    - 약 4.2M 파라미터 (ImageNet)
    - 계산량을 8-9배 줄이면서 정확도는 1% 정도만 감소
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        width_multiplier: float = 1.0,
        resolution_multiplier: float = 1.0,
        dropout_rate: float = 0.2,
        include_top: bool = True
    ):
        super(MobileNetV1, self).__init__()
        
        self.include_top = include_top
        
        # Width multiplier 적용
        def _make_divisible(v: float, divisor: int = 8) -> int:
            """Width multiplier 적용 후 8의 배수로 맞춤"""
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        # MobileNet V1 구조 정의
        # [출력 채널, stride]
        self.config = [
            [64, 1],   # Conv_dw_1
            [128, 2],  # Conv_dw_2 (downsampling)
            [128, 1],  # Conv_dw_3
            [256, 2],  # Conv_dw_4 (downsampling)
            [256, 1],  # Conv_dw_5
            [512, 2],  # Conv_dw_6 (downsampling)
            [512, 1],  # Conv_dw_7
            [512, 1],  # Conv_dw_8
            [512, 1],  # Conv_dw_9
            [512, 1],  # Conv_dw_10
            [512, 1],  # Conv_dw_11
            [1024, 2], # Conv_dw_12 (downsampling)
            [1024, 1], # Conv_dw_13
        ]
        
        # Width multiplier 적용
        self.config = [
            [_make_divisible(ch * width_multiplier), s] 
            for ch, s in self.config
        ]
        
        # 첫 번째 Standard Convolution
        input_channels = _make_divisible(32 * width_multiplier)
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Depthwise Separable Convolution 블록들
        self.features = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels, stride in self.config:
            self.features.append(
                MobileNetV1Block(in_channels, out_channels, stride)
            )
            in_channels = out_channels
        
        # Global Average Pooling & Classifier
        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(in_channels, num_classes)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem convolution
        x = self.stem(x)
        
        # Feature extraction
        for block in self.features:
            x = block(x)
        
        if self.include_top:
            # Classification head
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        SSD에서 사용할 feature map들 추출
        
        Returns:
            Dict containing feature maps at different scales
        """
        features = {}
        
        # Stem
        x = self.stem(x)
        features['stem'] = x  # 32 channels, H/2, W/2
        
        # Feature blocks
        for i, block in enumerate(self.features):
            x = block(x)
            
            # SSD에서 주로 사용하는 feature map들
            if i == 4:  # After conv_dw_5, 256 channels, H/8, W/8
                features['low_level'] = x
            elif i == 10:  # After conv_dw_11, 512 channels, H/16, W/16  
                features['mid_level'] = x
            elif i == 12:  # After conv_dw_13, 1024 channels, H/32, W/32
                features['high_level'] = x
        
        features['final'] = x
        return features
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 각 레이어별 출력 크기 계산 (224x224 입력 기준)
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.get_feature_maps(dummy_input)
        
        feature_shapes = {k: v.shape for k, v in features.items()}
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'feature_shapes': feature_shapes,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 대략적인 크기
        }


class MobileNetV1SSD(MobileNetV1):
    """
    SSD용 MobileNet V1 백본
    
    특징:
    - Classification head 제거
    - Multi-scale feature maps 출력
    - SSD detection head와 연결하기 위한 구조
    """
    
    def __init__(self, width_multiplier: float = 1.0):
        super(MobileNetV1SSD, self).__init__(
            include_top=False,
            width_multiplier=width_multiplier
        )
        
        # SSD용 추가 레이어들 (나중에 구현)
        self.extra_layers = nn.ModuleList()
        
        # 현재는 기본 MobileNet만 사용
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        SSD에서 사용할 multi-scale feature maps 반환
        
        Returns:
            List of feature tensors at different scales
        """
        feature_maps = []
        
        # Stem
        x = self.stem(x)
        
        # Feature blocks
        for i, block in enumerate(self.features):
            x = block(x)
            
            # SSD에서 detection에 사용할 feature map들
            if i in [4, 10, 12]:  # 다양한 스케일의 feature map
                feature_maps.append(x)
        
        # 마지막 feature map도 추가
        if len(self.features) - 1 not in [4, 10, 12]:
            feature_maps.append(x)
        
        return feature_maps


# 유틸리티 함수들
def mobilenet_v1(pretrained: bool = False, **kwargs) -> MobileNetV1:
    """MobileNet V1 모델 생성"""
    model = MobileNetV1(**kwargs)
    
    if pretrained:
        # 나중에 pretrained weights 로드 로직 추가
        print("Warning: Pretrained weights not implemented yet")
    
    return model


def mobilenet_v1_ssd(width_multiplier: float = 1.0) -> MobileNetV1SSD:
    """SSD용 MobileNet V1 백본 생성"""
    return MobileNetV1SSD(width_multiplier=width_multiplier)


def compare_mobilenet_variants():
    """다양한 MobileNet variant 비교"""
    variants = [
        ('MobileNet-1.0', 1.0),
        ('MobileNet-0.75', 0.75),
        ('MobileNet-0.5', 0.5),
        ('MobileNet-0.25', 0.25),
    ]
    
    print("=== MobileNet Variants Comparison ===")
    
    for name, width_mult in variants:
        model = mobilenet_v1(width_multiplier=width_mult, include_top=False)
        info = model.get_model_info()
        
        print(f"\n{name}:")
        print(f"  Parameters: {info['total_params']:,}")
        print(f"  Model Size: {info['model_size_mb']:.2f} MB")
        
        # 추론 시간 측정
        dummy_input = torch.randn(1, 3, 224, 224)
        
        import time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = (time.time() - start_time) / 100
        
        print(f"  Inference Time: {inference_time*1000:.2f} ms")


if __name__ == "__main__":
    print("=== MobileNet V1 테스트 ===")
    
    # 기본 MobileNet V1 테스트
    model = mobilenet_v1(num_classes=1000)
    
    # 테스트 입력
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    output = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    
    # 모델 정보
    info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Total Parameters: {info['total_params']:,}")
    print(f"  Model Size: {info['model_size_mb']:.2f} MB")
    
    # Feature maps 테스트
    print(f"\nFeature Maps:")
    for name, shape in info['feature_shapes'].items():
        print(f"  {name}: {shape}")
    
    # SSD 백본 테스트
    print(f"\n=== SSD Backbone 테스트 ===")
    ssd_backbone = mobilenet_v1_ssd(width_multiplier=1.0)
    feature_maps = ssd_backbone(x)
    
    print(f"SSD Feature Maps:")
    for i, fm in enumerate(feature_maps):
        print(f"  Level {i}: {fm.shape}")
    
    # Variants 비교
    print(f"\n=== MobileNet Variants 비교 ===")
    compare_mobilenet_variants()