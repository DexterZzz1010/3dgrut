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
from typing import Optional
from omegaconf import DictConfig

class FeatureCompressor(nn.Module):
    """Base class for feature compressors that defines the common interface."""
    
    def __init__(
        self,
        conf: DictConfig,
        feature_dim: int,
        device: Optional[str] = None,
    ):
        """Initialize the feature compressor.
        
        Args:
            conf: Hydra configuration containing compressor parameters
            device: Device to store tensors on (default: same as input tensors)
        """
        super().__init__()
        self.device = device
        
        # Parse common configuration
        self.n_components = conf.n_components
        self.feature_dim = feature_dim
        
    def compress_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compress features using the learned compression.
        
        Args:
            features: Tensor of shape (..., feature_dim) containing features
            
        Returns:
            Compressed features of shape (..., n_components)
        """
        raise NotImplementedError("Subclasses must implement compress_features")
        
    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass that compresses features.
        
        Args:
            features: Tensor of shape (..., feature_dim) containing features
            
        Returns:
            Compressed features of shape (..., n_components)
        """
        return self.compress_features(features) 