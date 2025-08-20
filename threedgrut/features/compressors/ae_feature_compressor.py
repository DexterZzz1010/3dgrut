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
from .feature_compressor import FeatureCompressor

class AutoencoderFeatureCompressor(FeatureCompressor):
    """A differentiable autoencoder feature compressor that learns to compress features."""
    
    def __init__(
        self,
        conf: DictConfig,
        feature_dim: int,
        device: Optional[str] = None,
    ):
        """Initialize the autoencoder feature compressor.
        
        Args:
            conf: Hydra configuration containing autoencoder parameters
            device: Device to store tensors on (default: same as input tensors)
        """
        super().__init__(conf, feature_dim, device)
        
        # Parse autoencoder-specific configuration
        self.hidden_dims = conf.hidden_dims
        self.activation = self._get_activation(conf.activation)
        self.dropout = conf.dropout
        
        # Build networks
        self._build_networks()
        
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function from name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        if name not in activations:
            raise ValueError(f"Unknown activation function: {name}. Must be one of {list(activations.keys())}")
        return activations[name]
        
    def _build_networks(self) -> None:
        """Build encoder and decoder networks."""
        if self.hidden_dims is None:
            # Simple linear projection
            self.encoder = nn.Linear(self.feature_dim, self.n_components)
            self.decoder = nn.Linear(self.n_components, self.feature_dim)
        else:
            # Build encoder
            encoder_layers = []
            prev_dim = self.feature_dim
            for hidden_dim in self.hidden_dims[:-1]:
                encoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    self.activation
                ])
                if self.dropout > 0:
                    encoder_layers.append(nn.Dropout(self.dropout))
                prev_dim = hidden_dim
            # Final layer projects to n_components
            encoder_layers.append(nn.Linear(prev_dim, self.n_components))
            self.encoder = nn.Sequential(*encoder_layers)
            
            # Build decoder
            decoder_layers = []
            prev_dim = self.n_components
            for hidden_dim in self.hidden_dims:
                decoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    self.activation
                ])
                if self.dropout > 0:
                    decoder_layers.append(nn.Dropout(self.dropout))
                prev_dim = hidden_dim
            decoder_layers.append(nn.Linear(prev_dim, self.feature_dim))
            self.decoder = nn.Sequential(*decoder_layers)
            
        self.to(self.device)
        
    def compress_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compress features using the learned autoencoder.
        
        Args:
            features: Tensor of shape (_,feature_dim) containing features
            
        Returns:
            Compressed features of shape (_,n_components)
        """
        if self.device is None:
            self.device = features.device
            self.to(self.device)
        
        # Store original shape
        original_shape = features.shape
        
        # Reshape to (N, C) for the network
        features = features.reshape(-1, features.shape[-1])
        
        # Verify feature dimension
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim}, got {features.shape[-1]}")
        
        # Get compressed representation
        compressed = self.encoder(features)
        
        # Restore original shape except for last dimension
        compressed = compressed.reshape(*original_shape[:-1], compressed.shape[-1])
        
        return compressed
    
    def reconstruct_features(
        self,
        compressed: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct features from compressed representation.
        
        Args:
            compressed: Tensor of shape (_,n_components) containing compressed features
            
        Returns:
            Reconstructed features of shape (_,feature_dim)
        """
        return self.decoder(compressed)