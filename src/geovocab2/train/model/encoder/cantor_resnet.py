# geovocab2.train.model.architectures.cantor_resnet.py
# ============================================================
# CantorResNet: Fractal-structured residual network
# Author: AbstractPhil
# Date: 2025-10-30
# License: MIT
# ============================================================

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal


# Assuming CantorConv2d and CantorLinear are importable - adjust path as needed
from geovocab2.train.model.layers.conv import CantorConv2d, CantorConv2dConfig
from geovocab2.train.model.layers.linear import CantorLinear, CantorLinearConfig


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class CantorResNetConfig:
    # Architecture
    arch: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"] = "resnet18"
    num_classes: int = 10
    in_channels: int = 3

    # Cantor parameters (shared across all layers)
    cantor_depth: int = 8
    mask_mode: str = "alpha"
    mask_floor: float = 0.25
    mask_scale: float = 0.5
    alpha_mode: str = "sigmoid"
    alpha_min: float = 0.1
    alpha_max: float = 1.0
    per_output_alpha: bool = False

    # Training
    dtype: torch.dtype = torch.float32
    device: str | None = None


# ============================================================
# BASIC BLOCK (for ResNet18, 34)
# ============================================================

class CantorBasicBlock(nn.Module):
    """BasicBlock using CantorConv2d layers."""
    expansion = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: nn.Module | None = None,
            cantor_cfg: dict = None
    ):
        super().__init__()

        # First conv: 3x3, stride may be 2 for downsampling
        self.conv1 = CantorConv2d(CantorConv2dConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            **cantor_cfg
        ))
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second conv: 3x3, stride=1
        self.conv2 = CantorConv2d(CantorConv2dConfig(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            **cantor_cfg
        ))
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ============================================================
# BOTTLENECK BLOCK (for ResNet50, 101, 152)
# ============================================================

class CantorBottleneck(nn.Module):
    """Bottleneck using CantorConv2d layers."""
    expansion = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: nn.Module | None = None,
            cantor_cfg: dict = None
    ):
        super().__init__()

        # 1x1 reduction
        self.conv1 = CantorConv2d(CantorConv2dConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            **cantor_cfg
        ))
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 main conv
        self.conv2 = CantorConv2d(CantorConv2dConfig(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            **cantor_cfg
        ))
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 expansion
        self.conv3 = CantorConv2d(CantorConv2dConfig(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=1,
            bias=False,
            **cantor_cfg
        ))
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ============================================================
# CANTOR RESNET
# ============================================================

class CantorResNet(nn.Module):
    """ResNet architecture using CantorConv2d layers."""

    # Architecture definitions: [blocks per stage]
    ARCH_SPECS = {
        "resnet18": (CantorBasicBlock, [2, 2, 2, 2]),
        "resnet34": (CantorBasicBlock, [3, 4, 6, 3]),
        "resnet50": (CantorBottleneck, [3, 4, 6, 3]),
        "resnet101": (CantorBottleneck, [3, 4, 23, 3]),
        "resnet152": (CantorBottleneck, [3, 8, 36, 3]),
    }

    def __init__(self, cfg: CantorResNetConfig):
        super().__init__()
        self.cfg = cfg

        # Get architecture spec
        block_class, layers = self.ARCH_SPECS[cfg.arch]

        # Cantor config dict for all layers
        self.cantor_cfg = {
            "depth": cfg.cantor_depth,
            "mask_mode": cfg.mask_mode,
            "mask_floor": cfg.mask_floor,
            "mask_scale": cfg.mask_scale,
            "alpha_mode": cfg.alpha_mode,
            "alpha_min": cfg.alpha_min,
            "alpha_max": cfg.alpha_max,
            "per_output_alpha": cfg.per_output_alpha,
            "dtype": cfg.dtype,
            "device": cfg.device,
        }

        self.in_channels = 64

        # Initial conv: 7x7, stride=2
        self.conv1 = CantorConv2d(CantorConv2dConfig(
            in_channels=cfg.in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            **self.cantor_cfg
        ))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block_class, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block_class, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_class, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_class, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = CantorLinear(CantorLinearConfig(
            in_features=512 * block_class.expansion,
            out_features=cfg.num_classes,
            bias=True,
            **self.cantor_cfg
        ))

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
            self,
            block_class: type,
            out_channels: int,
            num_blocks: int,
            stride: int = 1
    ) -> nn.Sequential:
        """Create a residual stage with multiple blocks."""
        downsample = None

        # Downsample if dimensions change
        if stride != 1 or self.in_channels != out_channels * block_class.expansion:
            downsample = nn.Sequential(
                CantorConv2d(CantorConv2dConfig(
                    in_channels=self.in_channels,
                    out_channels=out_channels * block_class.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    **self.cantor_cfg
                )),
                nn.BatchNorm2d(out_channels * block_class.expansion),
            )

        layers = []
        # First block (may downsample)
        layers.append(block_class(
            self.in_channels,
            out_channels,
            stride,
            downsample,
            self.cantor_cfg
        ))

        self.in_channels = out_channels * block_class.expansion

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block_class(
                self.in_channels,
                out_channels,
                cantor_cfg=self.cantor_cfg
            ))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize batch norm layers (CantorConv2d and CantorLinear handle their own init)."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_alpha_stats(self) -> dict[str, list[float]]:
        """Collect alpha statistics from all CantorConv2d and CantorLinear layers."""
        stats = {
            "layer_names": [],
            "alpha_means": [],
            "alpha_stds": [],
            "mask_densities": []
        }

        for name, module in self.named_modules():
            if isinstance(module, (CantorConv2d, CantorLinear)):
                alpha_stats = module.get_alpha_stats()
                if alpha_stats:
                    stats["layer_names"].append(name)
                    stats["alpha_means"].append(alpha_stats["alpha_mean"])
                    stats["alpha_stds"].append(alpha_stats.get("alpha_std", 0.0))
                    stats["mask_densities"].append(module.mask.mean().item())

        return stats


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def cantor_resnet18(num_classes: int = 10, **cantor_kwargs) -> CantorResNet:
    """CantorResNet-18."""
    cfg = CantorResNetConfig(arch="resnet18", num_classes=num_classes, **cantor_kwargs)
    return CantorResNet(cfg)


def cantor_resnet34(num_classes: int = 10, **cantor_kwargs) -> CantorResNet:
    """CantorResNet-34."""
    cfg = CantorResNetConfig(arch="resnet34", num_classes=num_classes, **cantor_kwargs)
    return CantorResNet(cfg)


def cantor_resnet50(num_classes: int = 10, **cantor_kwargs) -> CantorResNet:
    """CantorResNet-50."""
    cfg = CantorResNetConfig(arch="resnet50", num_classes=num_classes, **cantor_kwargs)
    return CantorResNet(cfg)


def cantor_resnet101(num_classes: int = 10, **cantor_kwargs) -> CantorResNet:
    """CantorResNet-101."""
    cfg = CantorResNetConfig(arch="resnet101", num_classes=num_classes, **cantor_kwargs)
    return CantorResNet(cfg)


def cantor_resnet152(num_classes: int = 10, **cantor_kwargs) -> CantorResNet:
    """CantorResNet-152."""
    cfg = CantorResNetConfig(arch="resnet152", num_classes=num_classes, **cantor_kwargs)
    return CantorResNet(cfg)


# ============================================================
# ACTIVATION TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing CantorResNet Architectures")
    print("=" * 60)

    # Test ResNet18
    print("\n[Test 1] CantorResNet-18")
    model = cantor_resnet18(num_classes=10, cantor_depth=6)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Alpha statistics
    stats = model.get_alpha_stats()
    if stats["alpha_means"]:
        print(f"\nAlpha statistics across {len(stats['alpha_means'])} layers:")
        print(f"  Mean alpha: {sum(stats['alpha_means']) / len(stats['alpha_means']):.4f}")
        print(f"  Mean mask density: {sum(stats['mask_densities']) / len(stats['mask_densities']):.4f}")

    # Test CIFAR-10 size
    print("\n[Test 2] CantorResNet-18 (CIFAR-10)")
    x_cifar = torch.randn(4, 3, 32, 32)
    y_cifar = model(x_cifar)
    print(f"CIFAR input shape: {x_cifar.shape}")
    print(f"CIFAR output shape: {y_cifar.shape}")

    # Test ResNet50
    print("\n[Test 3] CantorResNet-50")
    model50 = cantor_resnet50(num_classes=100, cantor_depth=8, per_output_alpha=True)
    y50 = model50(x)
    print(f"Output shape: {y50.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model50.parameters()):,}")

    print("\n" + "=" * 60)
    print("All tests passed!")