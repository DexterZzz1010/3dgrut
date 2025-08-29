# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import math
from typing import Tuple

# Global cache for Gaussian kernels to avoid recomputation
_KERNEL_CACHE = {}


def clear_kernel_cache():
    """
    Clear the Gaussian kernel cache to free memory.
    Call this if you want to free up GPU memory from cached kernels.
    """
    global _KERNEL_CACHE
    _KERNEL_CACHE.clear()


def get_cache_size():
    """
    Get the current number of cached kernels.
    
    Returns:
        int: Number of cached Gaussian kernels
    """
    return len(_KERNEL_CACHE)


def _get_cached_gaussian_kernel_1d(
    sigma: float, 
    kernel_size: int, 
    num_channels: int, 
    device: torch.device, 
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get cached 1D Gaussian kernels for separable convolution (much faster).
    
    Args:
        sigma: Gaussian blur sigma
        kernel_size: Size of the kernel (odd number)
        num_channels: Number of channels for group convolution
        device: Target device
        dtype: Target dtype
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Horizontal and vertical kernels for separable conv
    """
    # Create cache key
    cache_key = (sigma, kernel_size, num_channels, device, dtype)
    
    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]
    
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    # Create separable kernels for horizontal and vertical convolutions
    # Horizontal: [C, 1, 1, K]
    kernel_h = gauss_1d[None, None, None, :].repeat(num_channels, 1, 1, 1)
    # Vertical: [C, 1, K, 1]
    kernel_v = gauss_1d[None, None, :, None].repeat(num_channels, 1, 1, 1)
    
    kernels = (kernel_h, kernel_v)
    
    # Cache and return (limit cache size to prevent memory issues)
    if len(_KERNEL_CACHE) < 100:  # Reasonable limit
        _KERNEL_CACHE[cache_key] = kernels
    elif len(_KERNEL_CACHE) == 100:
        # One-time warning when cache is full
        print("Warning: Gaussian kernel cache is full (100 entries). Consider calling clear_kernel_cache() periodically.")
    
    return kernels


def antialiased_downsample(
    input: torch.Tensor, 
    target_size: Tuple[int, int], 
    sigma: float = None, 
    kernel_size: int = None
) -> torch.Tensor:
    """
    High-quality anti-aliased downsampling that preserves high-frequency details.
    
    This method applies a Gaussian blur before downsampling to prevent aliasing artifacts,
    which is much better than simple area pooling for feature maps.
    
    Key optimizations:
    - Cached separable convolutions for speed
    - Replicate padding to avoid edge artifacts
    - Automatic sigma scaling based on downsampling ratio
    
    Args:
        input (torch.Tensor): Input tensor in BCHW format
        target_size (Tuple[int, int]): Target (height, width)
        sigma (float, optional): Gaussian blur sigma. If None, computed automatically based on scale factor
        kernel_size (int, optional): Gaussian kernel size. If None, computed automatically
        
    Returns:
        torch.Tensor: Downsampled tensor in BCHW format
    """
    B, C, H, W = input.shape
    target_h, target_w = target_size
    
    # If already at target size, return as is
    if (H, W) == (target_h, target_w):
        return input
    
    # Compute scale factors
    scale_h = H / target_h
    scale_w = W / target_w
    
    # Only apply anti-aliasing if we're downsampling (scale > 1)
    if scale_h <= 1.0 and scale_w <= 1.0:
        # Upsampling case - use bicubic
        return F.interpolate(input, size=target_size, mode='bicubic', align_corners=False)
    
    # Compute Gaussian blur parameters
    if sigma is None:
        # Use scale factor to determine blur amount - more blur for higher downsampling ratios
        sigma_h = max(0.5, (scale_h - 1) / 2)
        sigma_w = max(0.5, (scale_w - 1) / 2)
        sigma = max(sigma_h, sigma_w)
    
    if kernel_size is None:
        # Kernel size should be odd and cover ~3 standard deviations
        kernel_size = int(math.ceil(6 * sigma)) | 1  # Ensure odd
        kernel_size = max(3, kernel_size)
    
    # Get cached separable Gaussian kernels (much faster than 2D convolution)
    device = input.device
    dtype = input.dtype
    kernel_h, kernel_v = _get_cached_gaussian_kernel_1d(sigma, kernel_size, C, device, dtype)
    
    # Apply separable Gaussian blur with replicate padding (much better than zero padding)
    padding = kernel_size // 2
    
    # Horizontal pass with replicate padding
    padded_h = F.pad(input, (padding, padding, 0, 0), mode='replicate')
    blurred = F.conv2d(padded_h, kernel_h, padding=0, groups=C)
    
    # Vertical pass with replicate padding  
    padded_v = F.pad(blurred, (0, 0, padding, padding), mode='replicate')
    blurred = F.conv2d(padded_v, kernel_v, padding=0, groups=C)
    
    # Then downsample with bicubic interpolation
    downsampled = F.interpolate(blurred, size=target_size, mode='bicubic', align_corners=False)
    
    return downsampled


def adaptive_downsample(
    input: torch.Tensor, 
    target_size: Tuple[int, int], 
    preserve_detail: bool = True
) -> torch.Tensor:
    """
    Adaptive downsampling that chooses the best method based on scale factor.
    
    Args:
        input (torch.Tensor): Input tensor in BCHW format
        target_size (Tuple[int, int]): Target (height, width)
        preserve_detail (bool): Whether to use anti-aliasing for better detail preservation
        
    Returns:
        torch.Tensor: Downsampled tensor in BCHW format
    """
    B, C, H, W = input.shape
    target_h, target_w = target_size
    
    # If already at target size, return as is
    if (H, W) == (target_h, target_w):
        return input
    
    # Compute scale factors
    scale_h = H / target_h
    scale_w = W / target_w
    max_scale = max(scale_h, scale_w)
    
    if max_scale <= 1.0:
        # Upsampling - use bicubic
        return F.interpolate(input, size=target_size, mode='bicubic', align_corners=False)
    elif max_scale <= 2.0:
        # Small downsampling - bicubic is fine
        return F.interpolate(input, size=target_size, mode='bicubic', align_corners=False)
    else:
        # Large downsampling - use anti-aliasing if requested
        if preserve_detail:
            return antialiased_downsample(input, target_size)
        else:
            return F.interpolate(input, size=target_size, mode='bicubic', align_corners=False)


def downsample_features_bhwc(
    pred_features: torch.Tensor, 
    target_shape: Tuple[int, int], 
    method: str = "antialiased"
) -> torch.Tensor:
    """
    Convenience function for downsampling features in BHWC format.
    
    This function provides high-quality downsampling with significant performance optimizations:
    - Gaussian kernels are cached to avoid recomputation
    - Separable convolutions are used (2x faster than 2D convolution) 
    - Replicate padding eliminates edge artifacts
    - Anti-aliasing prevents artifacts while preserving high-frequency details
    
    Args:
        pred_features (torch.Tensor): Predicted features in BHWC format
        target_shape (Tuple[int, int]): Target (height, width) 
        method (str): Downsampling method - "antialiased", "adaptive", or "bicubic"
        
    Returns:
        torch.Tensor: Downsampled features in BHWC format
    """
    # Convert BHWC to BCHW
    features_bchw = pred_features.permute(0, 3, 1, 2)
    
    # Apply downsampling
    if method == "antialiased":
        downsampled_bchw = antialiased_downsample(features_bchw, target_shape)
    elif method == "adaptive":
        downsampled_bchw = adaptive_downsample(features_bchw, target_shape, preserve_detail=True)
    elif method == "bicubic":
        downsampled_bchw = F.interpolate(features_bchw, size=target_shape, mode='bicubic', align_corners=False)
    else:
        raise ValueError(f"Unknown downsampling method: {method}")
    
    # Convert back to BHWC
    return downsampled_bchw.permute(0, 2, 3, 1)


def compute_spatial_regularization_loss(
    pred_features: torch.Tensor,
    target_shape: Tuple[int, int],
    sigma_factor: float = 0.5,
    loss_type: str = "l2"
) -> torch.Tensor:
    """
    Compute spatial regularization loss that encourages similar features for nearby pixels.
    
    The spatial extent of the regularization correlates with the downsampling factor:
    - Larger downsampling ratios use larger regularization kernels
    - This prevents multiple high-resolution predictions from mapping to the same ground truth
    
    Args:
        pred_features (torch.Tensor): Predicted features in BHWC format
        target_shape (Tuple[int, int]): Target ground truth shape (height, width)
        sigma_factor (float): Factor to scale the regularization sigma. Higher values = more smoothing
        loss_type (str): Type of loss to use ("l1" or "l2")
        
    Returns:
        torch.Tensor: Spatial regularization loss (scalar)
    """
    B, H, W, C = pred_features.shape
    target_h, target_w = target_shape
    
    # Compute downsampling scale factors
    scale_h = H / target_h
    scale_w = W / target_w
    max_scale = max(scale_h, scale_w)
    
    # If no downsampling, return zero loss
    if max_scale <= 1.0:
        return torch.tensor(0.0, device=pred_features.device, dtype=pred_features.dtype)
    
    # Compute regularization sigma based on downsampling scale
    # Similar to anti-aliasing sigma computation but scaled by sigma_factor
    sigma_h = sigma_factor * max(0.5, (scale_h - 1) / 2)
    sigma_w = sigma_factor * max(0.5, (scale_w - 1) / 2)
    sigma = max(sigma_h, sigma_w)
    
    # Compute kernel size (odd number, ~3 standard deviations)
    kernel_size = int(math.ceil(6 * sigma)) | 1
    kernel_size = max(3, kernel_size)
    
    # Convert BHWC to BCHW for convolution
    features_bchw = pred_features.permute(0, 3, 1, 2)
    
    # Get cached Gaussian kernels for smoothing
    device = pred_features.device
    dtype = pred_features.dtype
    kernel_h, kernel_v = _get_cached_gaussian_kernel_1d(sigma, kernel_size, C, device, dtype)
    
    # Apply separable Gaussian smoothing
    padding = kernel_size // 2
    
    # Horizontal pass
    padded_h = F.pad(features_bchw, (padding, padding, 0, 0), mode='replicate')
    smoothed = F.conv2d(padded_h, kernel_h, padding=0, groups=C)
    
    # Vertical pass
    padded_v = F.pad(smoothed, (0, 0, padding, padding), mode='replicate')
    smoothed = F.conv2d(padded_v, kernel_v, padding=0, groups=C)
    
    # Convert back to BHWC
    smoothed_features = smoothed.permute(0, 2, 3, 1)
    
    # Compute regularization loss between original and smoothed features
    if loss_type == "l1":
        reg_loss = F.l1_loss(pred_features, smoothed_features)
    elif loss_type == "l2":
        reg_loss = F.mse_loss(pred_features, smoothed_features)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'l1' or 'l2'")
    
    return reg_loss


def get_downsampling_scale_factors(
    pred_shape: Tuple[int, int], 
    target_shape: Tuple[int, int]
) -> Tuple[float, float, float]:
    """
    Compute downsampling scale factors for spatial regularization.
    
    Args:
        pred_shape (Tuple[int, int]): Predicted features shape (height, width)
        target_shape (Tuple[int, int]): Target ground truth shape (height, width)
        
    Returns:
        Tuple[float, float, float]: (scale_h, scale_w, max_scale)
    """
    pred_h, pred_w = pred_shape
    target_h, target_w = target_shape
    
    scale_h = pred_h / target_h
    scale_w = pred_w / target_w
    max_scale = max(scale_h, scale_w)
    
    return scale_h, scale_w, max_scale
