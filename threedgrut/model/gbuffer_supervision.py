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

"""
Simple G-Buffer Supervision

Provides focused supervision for:
- Albedo supervision for pred_rgb (L1, L2, SSIM)
- Depth supervision for pred_dist (scale-invariant losses) 
- Normal supervision for pred_normals
"""

import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim
from typing import Dict, Optional


@torch.cuda.nvtx.range("gbuffer_albedo_supervision")
def albedo_supervision_loss(
    pred_rgb: torch.Tensor, 
    gt_albedo: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    use_l1: bool = True,
    use_l2: bool = False, 
    use_ssim: bool = True,
    lambda_l1: float = 1.0,
    lambda_l2: float = 1.0,
    lambda_ssim: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Albedo supervision for predicted RGB using usual losses.
    
    Args:
        pred_rgb: Predicted RGB [B, H, W, 3]
        gt_albedo: Ground truth albedo/basecolor [B, H, W, 3]
        valid_mask: Valid pixel mask [B, H, W, 1] (optional)
        use_l1/l2/ssim: Which losses to enable
        lambda_*: Loss weights
        
    Returns:
        Dictionary of loss components
    """
    # Ensure [B, H, W, 3] format
    if pred_rgb.shape[-1] != 3:
        raise ValueError(f"Expected pred_rgb in [B, H, W, 3] format, got {pred_rgb.shape}")
    if gt_albedo.shape[-1] != 3:
        raise ValueError(f"Expected gt_albedo in [B, H, W, 3] format, got {gt_albedo.shape}")
    
    losses = {}
    
    # Apply mask if provided
    if valid_mask is not None:
        if valid_mask.shape[-1] != 1:
            raise ValueError(f"Expected valid_mask in [B, H, W, 1] format, got {valid_mask.shape}")
        pred_rgb = pred_rgb * valid_mask
        gt_albedo = gt_albedo * valid_mask
    
    # L1 Loss
    if use_l1:
        if valid_mask is not None:
            l1_loss = torch.abs(pred_rgb - gt_albedo)
            l1_loss = (l1_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        else:
            l1_loss = torch.abs(pred_rgb - gt_albedo).mean()
        losses['albedo_l1'] = lambda_l1 * l1_loss
    
    # L2 Loss
    if use_l2:
        if valid_mask is not None:
            l2_loss = (pred_rgb - gt_albedo) ** 2
            l2_loss = (l2_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        else:
            l2_loss = ((pred_rgb - gt_albedo) ** 2).mean()
        losses['albedo_l2'] = lambda_l2 * l2_loss
    
    # SSIM Loss (requires [B, C, H, W] format)
    if use_ssim:
        pred_rgb_chw = pred_rgb.permute(0, 3, 1, 2)  # [B, 3, H, W]
        gt_albedo_chw = gt_albedo.permute(0, 3, 1, 2)  # [B, 3, H, W]
        ssim_loss = 1.0 - fused_ssim(pred_rgb_chw, gt_albedo_chw, padding="valid")
        losses['albedo_ssim'] = lambda_ssim * ssim_loss
    
    return losses


@torch.cuda.nvtx.range("gbuffer_depth_supervision")
def depth_supervision_loss(
    pred_dist: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    loss_type: str = "si_log",
    variance_focus: float = 0.85,
    lambda_depth: float = 1.0,
    use_gradient_loss: bool = True,
    lambda_gradient: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Scale-invariant depth supervision for predicted distance.
    
    Args:
        pred_dist: Predicted distance/depth [B, H, W, 1]
        gt_depth: Ground truth depth [B, H, W, 1]
        valid_mask: Valid pixel mask [B, H, W, 1] (optional)
        loss_type: "si_log" or "si_mse" (scale-invariant variants)
        variance_focus: Focus parameter for scale-invariant loss
        lambda_depth: Loss weight for main depth loss
        use_gradient_loss: Whether to include scale-invariant gradient loss
        lambda_gradient: Loss weight for gradient loss
        
    Returns:
        Dictionary of loss components
    """
    # Ensure [B, H, W, 1] format
    if pred_dist.shape[-1] != 1:
        raise ValueError(f"Expected pred_dist in [B, H, W, 1] format, got {pred_dist.shape}")
    if gt_depth.shape[-1] != 1:
        raise ValueError(f"Expected gt_depth in [B, H, W, 1] format, got {gt_depth.shape}")
    
    # Create valid mask based on depth values if not provided
    eps = 1e-6
    if valid_mask is None:
        valid_mask = (gt_depth > eps).float()
    else:
        if valid_mask.shape[-1] != 1:
            raise ValueError(f"Expected valid_mask in [B, H, W, 1] format, got {valid_mask.shape}")
        depth_valid = (gt_depth > eps).float()
        valid_mask = valid_mask * depth_valid
    
    # Clamp to avoid log(0)
    pred_dist = torch.clamp(pred_dist, min=eps)
    gt_depth = torch.clamp(gt_depth, min=eps)
    
    # Compute log difference for scale invariance
    log_diff = torch.log(pred_dist) - torch.log(gt_depth)
    
    # Masked computation
    valid_pixels = valid_mask.sum()
    if valid_pixels == 0:
        return {'depth_scale_invariant': torch.tensor(0.0, device=pred_dist.device)}
    
    log_diff_masked = log_diff * valid_mask
    
    if loss_type == "si_log":
        # Scale-invariant log loss: minimize variance after removing mean
        mean_log_diff = log_diff_masked.sum() / valid_pixels
        depth_loss = ((log_diff_masked - mean_log_diff) ** 2 * valid_mask).sum() / valid_pixels
    elif loss_type == "si_mse":
        # Scale-invariant MSE with variance term
        depth_loss = (log_diff_masked ** 2 * valid_mask).sum() / valid_pixels
        variance = ((log_diff_masked * valid_mask).sum() / valid_pixels) ** 2
        depth_loss = depth_loss - variance_focus * variance
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    losses = {'depth_scale_invariant': lambda_depth * depth_loss}
    
    # Scale-invariant gradient loss
    if use_gradient_loss:
        gradient_loss = _compute_scale_invariant_gradient_loss(
            pred_dist, gt_depth, valid_mask, eps
        )
        losses['depth_gradient'] = lambda_gradient * gradient_loss
    
    return losses


def _compute_scale_invariant_gradient_loss(
    pred_dist: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    eps: float
) -> torch.Tensor:
    """
    Compute scale-invariant gradient loss for depth supervision.
    Works in log space to be scale-invariant like the main depth loss.
    
    Args:
        pred_dist: Predicted distance/depth [B, H, W, 1]
        gt_depth: Ground truth depth [B, H, W, 1]
        valid_mask: Valid pixel mask [B, H, W, 1]
        eps: Small epsilon value for log stability
        
    Returns:
        Scale-invariant gradient loss
    """
    # Work in log space for scale invariance
    pred_log = torch.log(torch.clamp(pred_dist, min=eps))
    gt_log = torch.log(torch.clamp(gt_depth, min=eps))
    
    # Compute gradients in log space
    # X gradients (horizontal)
    pred_grad_x = torch.abs(pred_log[:, :, :, 1:] - pred_log[:, :, :, :-1])  # [B, H, W-1, 1]
    gt_grad_x = torch.abs(gt_log[:, :, :, 1:] - gt_log[:, :, :, :-1])       # [B, H, W-1, 1]
    
    # Y gradients (vertical)
    pred_grad_y = torch.abs(pred_log[:, :, 1:, :] - pred_log[:, :, :-1, :])  # [B, H-1, W, 1]
    gt_grad_y = torch.abs(gt_log[:, :, 1:, :] - gt_log[:, :, :-1, :])       # [B, H-1, W, 1]
    
    if valid_mask is not None:
        # Create masks for gradient regions (smaller by 1 pixel in each direction)
        mask_x = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]  # [B, H, W-1, 1]
        mask_y = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]  # [B, H-1, W, 1]
        
        # Masked gradient loss
        grad_x_diff = torch.abs(pred_grad_x - gt_grad_x) * mask_x
        grad_y_diff = torch.abs(pred_grad_y - gt_grad_y) * mask_y
        
        grad_x_loss = grad_x_diff.sum() / (mask_x.sum() + eps)
        grad_y_loss = grad_y_diff.sum() / (mask_y.sum() + eps)
    else:
        # Unmasked gradient loss
        grad_x_loss = torch.abs(pred_grad_x - gt_grad_x).mean()
        grad_y_loss = torch.abs(pred_grad_y - gt_grad_y).mean()
    
    return grad_x_loss + grad_y_loss


@torch.cuda.nvtx.range("gbuffer_normal_supervision")
def normal_supervision_loss(
    pred_normals: torch.Tensor,
    gt_normals: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    use_l1: bool = True,
    use_l2: bool = False,
    use_cosine: bool = True,
    lambda_l1: float = 1.0,
    lambda_l2: float = 1.0,
    lambda_cosine: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Normal supervision for predicted normals.
    
    IMPORTANT: Both predicted and ground truth normals must be in the same coordinate space!
    Typically, G-buffer normals are in camera space, so predicted normals should be
    transformed from world space to camera space before calling this function.
    
    Args:
        pred_normals: Predicted normals [B, H, W, 3] (in camera space)
        gt_normals: Ground truth normals [B, H, W, 3] (in camera space)
        valid_mask: Valid pixel mask [B, H, W, 1] (optional)
        use_l1/l2/cosine: Which losses to enable
        lambda_*: Loss weights
        
    Returns:
        Dictionary of loss components
    """
    # Ensure [B, H, W, 3] format
    if pred_normals.shape[-1] != 3:
        raise ValueError(f"Expected pred_normals in [B, H, W, 3] format, got {pred_normals.shape}")
    if gt_normals.shape[-1] != 3:
        raise ValueError(f"Expected gt_normals in [B, H, W, 3] format, got {gt_normals.shape}")
    
    # Normalize normals to unit vectors
    pred_normals = F.normalize(pred_normals, p=2, dim=-1)
    gt_normals = F.normalize(gt_normals, p=2, dim=-1)
    
    losses = {}
    
    # Apply mask if provided
    if valid_mask is not None:
        if valid_mask.shape[-1] != 1:
            raise ValueError(f"Expected valid_mask in [B, H, W, 1] format, got {valid_mask.shape}")
    
    # L1 Loss
    if use_l1:
        if valid_mask is not None:
            l1_loss = torch.abs(pred_normals - gt_normals)
            l1_loss = (l1_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        else:
            l1_loss = torch.abs(pred_normals - gt_normals).mean()
        losses['normal_l1'] = lambda_l1 * l1_loss
    
    # L2 Loss
    if use_l2:
        if valid_mask is not None:
            l2_loss = (pred_normals - gt_normals) ** 2
            l2_loss = (l2_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        else:
            l2_loss = ((pred_normals - gt_normals) ** 2).mean()
        losses['normal_l2'] = lambda_l2 * l2_loss
    
    # Cosine similarity loss
    if use_cosine:
        # Compute cosine similarity per pixel
        cosine_sim = (pred_normals * gt_normals).sum(dim=-1, keepdim=True)  # [B, H, W, 1]
        cosine_loss = 1.0 - cosine_sim  # Convert to loss (higher = worse)
        
        if valid_mask is not None:
            cosine_loss = (cosine_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        else:
            cosine_loss = cosine_loss.mean()
        losses['normal_cosine'] = lambda_cosine * cosine_loss
    
    return losses


def compute_gbuffer_supervision_losses(
    # Predictions
    pred_rgb: Optional[torch.Tensor] = None,
    pred_dist: Optional[torch.Tensor] = None, 
    pred_normals: Optional[torch.Tensor] = None,
    
    # Ground truth G-buffers
    gt_albedo: Optional[torch.Tensor] = None,
    gt_depth: Optional[torch.Tensor] = None,
    gt_normals: Optional[torch.Tensor] = None,
    
    # Mask
    valid_mask: Optional[torch.Tensor] = None,
    
    # Albedo supervision config
    use_albedo_supervision: bool = True,
    albedo_l1: bool = True,
    albedo_l2: bool = False,
    albedo_ssim: bool = True,
    lambda_albedo_l1: float = 1.0,
    lambda_albedo_l2: float = 1.0,
    lambda_albedo_ssim: float = 0.1,
    
    # Depth supervision config  
    use_depth_supervision: bool = True,
    depth_loss_type: str = "si_log",
    lambda_depth: float = 1.0,
    use_depth_gradient: bool = True,
    lambda_depth_gradient: float = 0.1,
    
    # Normal supervision config
    use_normal_supervision: bool = True,
    normal_l1: bool = True,
    normal_l2: bool = False,
    normal_cosine: bool = True,
    lambda_normal_l1: float = 1.0,
    lambda_normal_l2: float = 1.0,
    lambda_normal_cosine: float = 0.5,
    
) -> Dict[str, torch.Tensor]:
    """
    Compute all G-buffer supervision losses.
    
    This is the main function that orchestrates all G-buffer supervision.
    Only computes losses for available predictions and ground truth.
    
    Returns:
        Dictionary of all computed loss components
    """
    all_losses = {}
    
    # Albedo supervision (pred_rgb vs gt_albedo)
    if (use_albedo_supervision and pred_rgb is not None and gt_albedo is not None):
        albedo_losses = albedo_supervision_loss(
            pred_rgb, gt_albedo, valid_mask,
            use_l1=albedo_l1, use_l2=albedo_l2, use_ssim=albedo_ssim,
            lambda_l1=lambda_albedo_l1, lambda_l2=lambda_albedo_l2, lambda_ssim=lambda_albedo_ssim
        )
        all_losses.update(albedo_losses)
    
    # Depth supervision (pred_dist vs gt_depth)
    if (use_depth_supervision and pred_dist is not None and gt_depth is not None):
        depth_losses = depth_supervision_loss(
            pred_dist, gt_depth, valid_mask,
            loss_type=depth_loss_type, lambda_depth=lambda_depth,
            use_gradient_loss=use_depth_gradient, lambda_gradient=lambda_depth_gradient
        )
        all_losses.update(depth_losses)
    
    # Normal supervision (pred_normals vs gt_normals)
    if (use_normal_supervision and pred_normals is not None and gt_normals is not None):
        normal_losses = normal_supervision_loss(
            pred_normals, gt_normals, valid_mask,
            use_l1=normal_l1, use_l2=normal_l2, use_cosine=normal_cosine,
            lambda_l1=lambda_normal_l1, lambda_l2=lambda_normal_l2, lambda_cosine=lambda_normal_cosine
        )
        all_losses.update(normal_losses)
    
    return all_losses
