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
import torch.nn as nn
from typing import Optional, Tuple, Union
from omegaconf import DictConfig


class DINOv3FeatureExtractor(nn.Module):
    """A feature extractor using Facebook Research's DINOv3 model.
    
    This extractor uses DINOv3 self-supervised vision transformers to extract spatial 
    features from images. DINOv3 provides strong feature representations without 
    requiring labeled data during pretraining.
    
    The extractor handles:
    - Automatic resolution adjustment to supported sizes
    - Conversion between BHWC and BCHW formats
    - Mixed precision inference using bfloat16
    - Multiple DINOv3 model variants (ViT-S, ViT-B, ViT-L, ViT-g)
    
    Attributes:
        model_name (str): Name of the DINOv3 model to use
        upscale_factor (int): Factor to upscale input images by
        patch_size (int): Size of patches used by the model (14 for DINOv3)
    """
    
    def __init__(
        self,
        conf: DictConfig,
        device: Optional[str] = None,
    ):
        """Initialize the DINOv3 feature extractor.
        
        Args:
            conf: Hydra configuration containing DINOv3 parameters
            device: Device to store tensors on (default: same as input tensors)
        """
        super().__init__()
        self.device = device if device is not None else conf.device
        
        # Parse configuration
        self.model_name = conf.model_name
        self.upscale_factor = conf.upscale_factor
        
        # Initialize model using torch hub
        self.model = self._load_dinov3_model()
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device).eval()
        
        # DINOv3 models use patch size of 14
        self.patch_size = 14
        
        # Get output feature dimension
        self._features_dim = self._get_features_dim()
    
    def _load_dinov3_model(self):
        """Load DINOv3 model from torch.hub."""
        # Map model names to torch.hub model identifiers
        model_map = {
            "dinov3_vits14": "dinov3_vits14",      # ViT-Small/14
            "dinov3_vitb14": "dinov3_vitb14",      # ViT-Base/14  
            "dinov3_vitl14": "dinov3_vitl14",      # ViT-Large/14
            "dinov3_vitg14": "dinov3_vitg14",      # ViT-Giant/14
            # Variants with different pretraining
            "dinov3_vits14_reg": "dinov3_vits14_reg",
            "dinov3_vitb14_reg": "dinov3_vitb14_reg", 
            "dinov3_vitl14_reg": "dinov3_vitl14_reg",
            "dinov3_vitg14_reg": "dinov3_vitg14_reg",
        }
        
        if self.model_name not in model_map:
            raise ValueError(
                f"Unsupported DINOv3 model: {self.model_name}. "
                f"Supported models: {list(model_map.keys())}"
            )
        
        # Load model from torch.hub
        model = torch.hub.load(
            'facebookresearch/dinov3', 
            model_map[self.model_name],
            pretrained=True,
            progress=True
        )
        
        return model
    
    def _get_features_dim(self) -> int:
        """
        Determine the output feature dimension of the model.
        
        Returns:
            int: Number of output features
        """
        # Create minimal dummy input that's compatible with patch size
        dummy_size = 224  # Standard size that should be supported
        x = torch.zeros(1, 3, dummy_size, dummy_size).to(self.device)
        
        # Run forward pass to get feature dimension
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
            # DINOv3 returns features in format similar to other ViTs
            features = self.model.forward_features(x)
            
        # Get the feature dimension from the last dimension
        return features.shape[-1]
    
    @property
    def features_dim(self) -> int:
        """
        Returns the output feature dimension of the model.
        """
        return self._features_dim
    
    def get_output_shape(
        self, 
        input_shape: Union[Tuple[int, ...], torch.Size]
    ) -> Tuple[int, int]:
        """
        Compute the expected output shapes for a given input shape.
        
        Args:
            input_shape: Tuple of (batch_size, height, width, channels) in BHWC format
            
        Returns:
            Tuple[int, int]: (H_out, W_out) output spatial dimensions
        """    
        B, H, W, _ = input_shape
        
        # Apply upscaling
        upscaled_H = H * self.upscale_factor
        upscaled_W = W * self.upscale_factor
        
        # DINOv3 uses patch size of 14, ensure dimensions are compatible
        # Round to nearest multiple of patch_size
        H_valid = ((upscaled_H + self.patch_size - 1) // self.patch_size) * self.patch_size
        W_valid = ((upscaled_W + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        # Compute output spatial dimensions
        H_out = H_valid // self.patch_size
        W_out = W_valid // self.patch_size
        
        return H_out, W_out

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from input tensor (non-differentiable operation).
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C) in BHWC format
                where B=batch size, H=height, W=width, C=3 (RGB channels).
                Values should be in range [0, 1]
                
        Returns:
            torch.Tensor: Spatial features in BHWC format
        """    
        # Convert from BHWC to BCHW
        x = x.permute(0, 3, 1, 2)
        
        # Apply upscaling if specified
        if self.upscale_factor > 1:
            upscaled_shape = (x.shape[2] * self.upscale_factor, x.shape[3] * self.upscale_factor)
            x = torch.nn.functional.interpolate(x, upscaled_shape, mode='bicubic', align_corners=False)
        
        # Ensure dimensions are compatible with patch size (14 for DINOv3)
        current_h, current_w = x.shape[2], x.shape[3]
        H_valid = ((current_h + self.patch_size - 1) // self.patch_size) * self.patch_size
        W_valid = ((current_w + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        # Resize if needed
        if (H_valid, W_valid) != (current_h, current_w):
            x = torch.nn.functional.interpolate(
                x, (H_valid, W_valid), 
                mode='bicubic', align_corners=False
            )
        
        # Process with autocast for mixed precision
        with torch.autocast(self.device, dtype=torch.bfloat16):
            # Get patch features (excludes CLS token)
            features = self.model.forward_features(x)
            
            # DINOv3 returns features in format [B, 1 + num_patches, feature_dim]
            # where the first token is CLS token, we want spatial tokens
            spatial_features = features[:, 1:]  # Remove CLS token
            
        # Reshape from [B, num_patches, feature_dim] to [B, H, W, feature_dim]
        B, num_patches, D = spatial_features.shape
        H_out = H_valid // self.patch_size
        W_out = W_valid // self.patch_size
        
        # Verify the dimensions match
        expected_patches = H_out * W_out
        if num_patches != expected_patches:
            raise RuntimeError(
                f"Dimension mismatch: got {num_patches} spatial tokens but expected {expected_patches} "
                f"(H_out={H_out}, W_out={W_out}, patch_size={self.patch_size})"
            )
        
        spatial_features = spatial_features.view(B, H_out, W_out, D)
        
        return spatial_features.to(torch.float32)
