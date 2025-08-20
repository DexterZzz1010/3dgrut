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
from PIL import Image
import torchvision.transforms as T


class NVRadioFeatureExtractor(nn.Module):
    """A feature extractor using NVIDIA's RADIO model.
    
    This extractor uses NVIDIA's RADIO model to extract spatial features from images.
    It supports both standard RADIO and E-RADIO models, with automatic handling of
    input resolutions and feature formats.
    
    The extractor handles:
    - Automatic resolution adjustment to nearest supported size
    - Conversion between BHWC and BCHW formats
    - Mixed precision inference using bfloat16
    - Special handling for E-RADIO models
    
    Attributes:
        model_name (str): Name of the RADIO model to use
        upscale_factor (int): Factor to upscale input images by
        patch_size (int): Size of patches used by the model
        is_eradio (bool): Whether the model is an E-RADIO variant
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
        
        # Initialize model using torch hub
        self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', 
                                  version=self.model_name, 
                                  progress=True, 
                                  skip_validation=True)
        
        # Move model to GPU and set to eval mode
        self.model = self.model.to(self.device).eval()
        
        # Store model properties
        self.patch_size = self.model.patch_size
        self.is_eradio = "e-radio" in self.model_name
        
        # Get output feature dimension
        self._features_dim = self._get_features_dim()
        
    
    def _get_features_dim(self) -> int:
        """
        Determine the output feature dimension of the model.
        
        Returns:
            int: Number of output features
        """
        # Create minimal dummy input
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
        
        # Get nearest supported resolution
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
