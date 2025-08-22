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
from typing import Optional, Tuple, List, Dict, Union
from omegaconf import DictConfig
import torch.hub

try:
    from transformers import AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class NVRadioFeatureExtractor(nn.Module):
    """A feature extractor using NVIDIA's RADIO model.
    
    This extractor uses NVIDIA's RADIO model to extract spatial features from images.
    It supports RADIO v1/v2/v2.5 models (via torch.hub) and C-RADIOv3 models (via Hugging Face),
    with automatic handling of input resolutions and feature formats.
    
    The extractor handles:
    - Automatic resolution adjustment to nearest supported size
    - Conversion between BHWC and BCHW formats
    - Mixed precision inference using bfloat16
    - Special handling for E-RADIO models
    - Support for both torch.hub and Hugging Face model loading
    
    Attributes:
        model_name (str): Name of the RADIO model to use
        upscale_factor (int): Factor to upscale input images by
        patch_size (int): Size of patches used by the model
        is_eradio (bool): Whether the model is an E-RADIO variant
        is_c_radio_v3 (bool): Whether the model is a C-RADIOv3 variant
    """
    
    def __init__(
        self,
        conf: DictConfig,
        device: Optional[str] = None,
    ):
        """Initialize the RADIO feature extractor.
        
        Args:
            conf: Hydra configuration containing RADIO parameters
            device: Device to store tensors on (default: same as input tensors)
        """
        super().__init__()
        self.device = device if device is not None else conf.device
        
        # Parse configuration
        self.model_name = conf.model_name
        self.upscale_factor = conf.upscale_factor
        
        # Determine if this is a C-RADIOv3 model
        self.is_c_radio_v3 = "c-radio" in self.model_name.lower()
        self.is_eradio = "e-radio" in self.model_name.lower()
        
        # Initialize model based on type
        if self.is_c_radio_v3:
            self._init_c_radio_v3_model()
        else:
            self._init_legacy_radio_model()
        
        # Get output feature dimension
        self._features_dim = self._get_features_dim()
    
    def _init_c_radio_v3_model(self):
        """Initialize C-RADIOv3 model from Hugging Face."""
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers is required for C-RADIOv3 models. "
                "Install with: pip install transformers"
            )
        
        # Map model name to Hugging Face repository
        hf_model_map = {
            "c-radiov3-b": "nvidia/C-RADIOv3-B",
            "c-radiov3-l": "nvidia/C-RADIOv3-L", 
            "c-radiov3-h": "nvidia/C-RADIOv3-H",
            "c-radiov3-g": "nvidia/C-RADIOv3-g"
        }
        
        hf_repo = hf_model_map.get(self.model_name.lower())
        if hf_repo is None:
            raise ValueError(
                f"Unsupported C-RADIOv3 model: {self.model_name}. "
                f"Supported models: {list(hf_model_map.keys())}"
            )
        
        # Load model only (no need for image processor since we handle tensors directly)
        self.model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device).eval()
        
        # Set patch size for C-RADIOv3 (patch size is 16 according to HF docs)
        self.patch_size = 16
        
        # No image processor needed for tensor processing
        self.image_processor = None
    
    def _init_legacy_radio_model(self):
        """Initialize legacy RADIO models (v1, v2, v2.5) from torch.hub."""
        # Initialize model using torch hub
        self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', 
                                  version=self.model_name, 
                                  progress=True, 
                                  skip_validation=True)
        
        # Move model to GPU and set to eval mode
        self.model = self.model.to(self.device).eval()
        
        # Store model properties
        self.patch_size = self.model.patch_size
        
        # No image processor for legacy models
        self.image_processor = None
    
    def _get_features_dim(self) -> int:
        """
        Determine the output feature dimension of the model.
        
        Returns:
            int: Number of output features
        """
        if self.is_c_radio_v3:
            # For C-RADIOv3 models, create minimal dummy input with supported resolution
            dummy_size = 224  # Standard size that should be supported
            nearest_res = self.model.get_nearest_supported_resolution(dummy_size, dummy_size)
            x = torch.zeros(1, 3, nearest_res.height, nearest_res.width).to(self.device)
            
            # Run forward pass
            with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
                _, features = self.model(x)
            
            return features.shape[-1]  # Last dimension contains features
        else:
            # Legacy RADIO models
            x = torch.zeros(1, 3, self.patch_size, self.patch_size).to(self.device)
            
            # Run forward pass to get feature dimension
            with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
                _, features = self.model(x, feature_fmt='NCHW')
                
            return features.shape[1]  # Channel dimension in NCHW format
    
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
        
        if self.is_c_radio_v3:
            # For C-RADIOv3 models, apply upscaling and get nearest supported resolution
            upscaled_H = H * self.upscale_factor
            upscaled_W = W * self.upscale_factor
            
            # Use the model's get_nearest_supported_resolution method
            nearest_res = self.model.get_nearest_supported_resolution(upscaled_H, upscaled_W)
            H_valid, W_valid = nearest_res.height, nearest_res.width
        else:
            # Get nearest supported resolution for legacy models
            H_valid, W_valid = self.model.get_nearest_supported_resolution(H, W)
        
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
        if self.is_c_radio_v3:
            return self._forward_c_radio_v3(x)
        else:
            return self._forward_legacy_radio(x)
    
    def _forward_c_radio_v3(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for C-RADIOv3 models."""
        # Convert from BHWC to BCHW for processing
        x = x.permute(0, 3, 1, 2)
        
        # Apply upscaling if specified
        if self.upscale_factor > 1:
            upscaled_shape = (x.shape[2] * self.upscale_factor, x.shape[3] * self.upscale_factor)
            x = torch.nn.functional.interpolate(x, upscaled_shape, mode='bicubic', align_corners=False)
        
        # Get nearest supported resolution for C-RADIOv3 models
        current_h, current_w = x.shape[2], x.shape[3]
        nearest_res = self.model.get_nearest_supported_resolution(current_h, current_w)
        
        # Resize to nearest supported resolution if needed
        if (nearest_res.height, nearest_res.width) != (current_h, current_w):
            x = torch.nn.functional.interpolate(
                x, (nearest_res.height, nearest_res.width), 
                mode='bicubic', align_corners=False
            )
        
        # Store final processed dimensions for reshaping
        final_h, final_w = x.shape[2], x.shape[3]
        
        # Process with autocast for mixed precision
        with torch.autocast(self.device, dtype=torch.bfloat16):
            _, spatial_features = self.model(x)
        
        # Convert spatial features from (B,T,D) to (B,H,W,D) format
        # Calculate output dimensions based on processed input and patch size
        B, T, D = spatial_features.shape
        H_out = final_h // self.patch_size
        W_out = final_w // self.patch_size
        
        # Verify the dimensions match
        expected_tokens = H_out * W_out
        if T != expected_tokens:
            raise RuntimeError(
                f"Dimension mismatch: got {T} spatial tokens but expected {expected_tokens} "
                f"(H_out={H_out}, W_out={W_out}, patch_size={self.patch_size})"
            )
        
        spatial_features = spatial_features.view(B, H_out, W_out, D)
        
        return spatial_features.to(torch.float32)
    
    def _forward_legacy_radio(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for legacy RADIO models."""
        # Convert from BHWC to BCHW
        x = x.permute(0, 3, 1, 2)
        
        # Get nearest supported resolution and resize if needed
        upscaled_shape = (x.shape[2] * self.upscale_factor, x.shape[3] * self.upscale_factor)
        nearest_res = self.model.get_nearest_supported_resolution(*upscaled_shape)
        if nearest_res != upscaled_shape:
            x = torch.nn.functional.interpolate(x, nearest_res, mode='bicubic', align_corners=False)
        
        # Set optimal window size for E-RADIO
        if self.is_eradio:
            self.model.model.set_optimal_window_size(upscaled_shape)
        
        # Process with autocast for mixed precision
        with torch.autocast(self.device, dtype=torch.bfloat16):
            _, spatial_features = self.model(x, feature_fmt='NCHW')
        
        # Convert spatial features from BCHW to BHWC
        spatial_features = spatial_features.permute(0, 2, 3, 1)

        return spatial_features.to(torch.float32)
