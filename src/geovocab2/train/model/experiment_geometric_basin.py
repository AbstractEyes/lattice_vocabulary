"""
GEOMETRIC BASIN CLASSIFIER - CIFAR-100 [PROPER STRUCTURE]
This is a frozen example meant to be used as a reference for block extraction and layer preparation.
----------------------------------------------------------
Meant to replace the need for cross-entropy with cantor stairs and produce a more solid form of loss. The experiment was successful.
Requires additional testing with alternative systems and accessors.
Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#import torchvision
#import torchvision.transforms as transforms
from tqdm import tqdm
import math
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
import csv

import geovocab2.train.model.blocks.geometric_basin

# Hugging Face Hub integration
try:
    from huggingface_hub import HfApi, create_repo

    HF_AVAILABLE = True
except ImportError:
    print("⚠️  huggingface_hub not installed. Run: pip install huggingface_hub")
    HF_AVAILABLE = False

# Safetensors integration
try:
    from safetensors.torch import save_file as save_safetensors
    from safetensors.torch import load_file as load_safetensors

    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("⚠️  safetensors not installed. Run: pip install safetensors")
    SAFETENSORS_AVAILABLE = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MIXING AUGMENTATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def alphamix_data(x, y, alpha_range=(0.3, 0.7), spatial_ratio=0.25):
    """AlphaMix: Spatially localized transparent overlay."""
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    y_a, y_b = y, y[index]

    alpha_min, alpha_max = alpha_range
    beta_sample = np.random.beta(2, 2)
    alpha = alpha_min + (alpha_max - alpha_min) * beta_sample

    _, _, H, W = x.shape
    overlay_ratio = np.sqrt(spatial_ratio)
    overlay_h = int(H * overlay_ratio)
    overlay_w = int(W * overlay_ratio)

    top = np.random.randint(0, H - overlay_h + 1)
    left = np.random.randint(0, W - overlay_w + 1)

    composited_x = x.clone()
    overlay_region = alpha * x[:, :, top:top + overlay_h, left:left + overlay_w]
    background_region = (1 - alpha) * x[index, :, top:top + overlay_h, left:left + overlay_w]
    composited_x[:, :, top:top + overlay_h, left:left + overlay_w] = overlay_region + background_region

    return composited_x, y_a, y_b, alpha





# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RESIDUAL BLOCK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ResidualBlock(nn.Module):
    """Basic residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEOMETRIC BASIN COMPATIBILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeometricBasinCompatibility(nn.Module):
    """Compute geometric compatibility scores - FULLY BATCHED."""

    def __init__(self, num_classes=100, pe_levels=20, features_per_level=4):
        super().__init__()

        self.num_classes = num_classes
        self.pe_levels = pe_levels
        self.features_per_level = features_per_level

        self.class_signatures = nn.Parameter(
            torch.randn(num_classes, pe_levels, features_per_level) * 0.1
        )

        self.cantor_prototypes = nn.Parameter(
            torch.linspace(0.0, 1.0, num_classes)
        )

        self.level_resonance = nn.Parameter(
            torch.ones(num_classes, pe_levels) / pe_levels
        )

    def forward(self, pe_levels, cantor_measures):
        B = pe_levels.shape[0]

        # 1. TRIADIC COMPATIBILITY
        pe_norm = F.normalize(pe_levels, p=2, dim=-1)
        sig_norm = F.normalize(self.class_signatures, p=2, dim=-1)

        similarities = torch.einsum('blf,clf->bcl', pe_norm, sig_norm)
        similarities = (similarities + 1) / 2

        resonance = F.softmax(self.level_resonance, dim=-1)
        triadic_compat = (similarities * resonance.unsqueeze(0)).sum(dim=-1)

        # 2. SELF-SIMILARITY
        level_pairs = []
        for k in range(self.pe_levels - 1):
            level_k = pe_levels[:, k, :]
            level_k1 = pe_levels[:, k + 1, :]
            sim = F.cosine_similarity(level_k, level_k1, dim=-1, eps=1e-8)
            sim = (sim + 1) / 2
            level_pairs.append(sim)

        self_sim_pattern = torch.stack(level_pairs, dim=1)

        expected_patterns = torch.sigmoid(
            self.level_resonance[:, :-1] - self.level_resonance[:, 1:]
        )

        pattern_diff = torch.abs(
            self_sim_pattern.unsqueeze(1) - expected_patterns.unsqueeze(0)
        )
        self_sim_compat = 1 - pattern_diff.mean(dim=-1)
        self_sim_compat = torch.clamp(self_sim_compat, 0.0, 1.0)

        # 3. CANTOR COHERENCE
        distances = torch.abs(
            cantor_measures.unsqueeze(1) - self.cantor_prototypes.unsqueeze(0)
        )
        cantor_compat = torch.exp(-distances ** 2 / 0.1) + 1e-8

        # 4. HIERARCHICAL CHECK
        split_point = self.pe_levels // 2
        early_levels = pe_levels[:, :split_point, :].mean(dim=1)
        late_levels = pe_levels[:, split_point:, :].mean(dim=1)

        early_targets = self.class_signatures[:, :split_point, :].mean(dim=1)
        late_targets = self.class_signatures[:, split_point:, :].mean(dim=1)

        early_levels_norm = F.normalize(early_levels, p=2, dim=-1)
        late_levels_norm = F.normalize(late_levels, p=2, dim=-1)
        early_targets_norm = F.normalize(early_targets, p=2, dim=-1)
        late_targets_norm = F.normalize(late_targets, p=2, dim=-1)

        early_compat = torch.matmul(early_levels_norm, early_targets_norm.t())
        late_compat = torch.matmul(late_levels_norm, late_targets_norm.t())

        early_compat = (early_compat + 1) / 2
        late_compat = (late_compat + 1) / 2
        hier_compat = (early_compat + late_compat) / 2

        # 5. COMBINE
        eps = 1e-6
        triadic_compat = torch.clamp(triadic_compat, eps, 1.0)
        self_sim_compat = torch.clamp(self_sim_compat, eps, 1.0)
        cantor_compat = torch.clamp(cantor_compat, eps, 1.0)
        hier_compat = torch.clamp(hier_compat, eps, 1.0)

        compatibility_scores = (
                                       triadic_compat *
                                       self_sim_compat *
                                       cantor_compat *
                                       hier_compat
                               ) ** 0.25

        return compatibility_scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEOMETRIC BASIN LOSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeometricBasinLoss(nn.Module):
    """Loss based on geometric basin compatibility."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, compatibility_scores, labels, mixed_labels=None, lam=None):
        batch_size = compatibility_scores.shape[0]

        if mixed_labels is not None and lam is not None:
            primary_compat = compatibility_scores[torch.arange(batch_size), labels]
            secondary_compat = compatibility_scores[torch.arange(batch_size), mixed_labels]

            primary_loss = F.mse_loss(primary_compat, torch.full_like(primary_compat, lam))
            secondary_loss = F.mse_loss(secondary_compat, torch.full_like(secondary_compat, 1 - lam))

            soft_targets = torch.zeros_like(compatibility_scores)
            soft_targets[torch.arange(batch_size), labels] = lam
            soft_targets[torch.arange(batch_size), mixed_labels] = 1 - lam

            compat_normalized = compatibility_scores / (compatibility_scores.sum(dim=1, keepdim=True) + 1e-8)
            kl_loss = F.kl_div(
                compat_normalized.log(),
                soft_targets,
                reduction='batchmean'
            )

            total_loss = primary_loss + secondary_loss + 0.1 * kl_loss

        else:
            correct_compat = compatibility_scores[torch.arange(batch_size), labels]
            correct_loss = -torch.log(correct_compat + 1e-8).mean()

            mask = torch.ones_like(compatibility_scores)
            mask[torch.arange(batch_size), labels] = 0

            incorrect_compat = compatibility_scores * mask
            incorrect_loss = torch.log(1 - incorrect_compat + 1e-8).mean()
            incorrect_loss = -incorrect_loss

            scaled_scores = compatibility_scores / self.temperature
            log_probs = F.log_softmax(scaled_scores, dim=1)
            contrastive_loss = F.nll_loss(log_probs, labels)

            total_loss = correct_loss + 0.5 * incorrect_loss + 0.5 * contrastive_loss

        return total_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEOMETRIC BASIN CLASSIFIER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeometricBasinClassifier(nn.Module):
    """BIGGER classifier with deeper ResNet-style backbone."""

    def __init__(self, num_classes=100, pe_levels=20, pe_features_per_level=4, dropout=0.1):
        super().__init__()

        self.num_classes = num_classes
        self.pe_levels = pe_levels
        self.pe_features_per_level = pe_features_per_level

        # Initial conv
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks
        self.layer1 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer2 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, 512, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, 1024, num_blocks=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        # Devil's Staircase PE
        self.pe = geovocab2.train.model.blocks.geometric_basin.BeatrixStaircasePositionalEncodings(pe_levels, pe_features_per_level)

        # PE modulator
        self.pe_modulator = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, pe_levels * pe_features_per_level)
        )

        # Geometric basin
        self.basin = GeometricBasinCompatibility(
            num_classes,
            pe_levels,
            pe_features_per_level
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, return_details=False):
        batch_size = x.shape[0]

        # CNN backbone
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        cnn_features = self.global_pool(x).flatten(1)
        cnn_features = self.dropout(cnn_features)

        # Generate PE
        positions = torch.arange(batch_size, device=x.device)
        pe_levels, cantor_measures = self.pe(positions, seq_len=batch_size)

        # Modulate PE with CNN features
        modulation = self.pe_modulator(cnn_features)
        modulation = modulation.view(batch_size, self.pe_levels, self.pe_features_per_level)
        pe_levels = pe_levels + 0.1 * modulation

        # Geometric basin compatibility
        compatibility_scores = self.basin(pe_levels, cantor_measures)

        if return_details:
            return {
                'compatibility_scores': compatibility_scores,
                'pe_levels': pe_levels,
                'cantor_measures': cantor_measures,
                'cnn_features': cnn_features
            }

        return compatibility_scores