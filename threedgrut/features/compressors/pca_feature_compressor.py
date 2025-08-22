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

class PCAFeatureCompressor(FeatureCompressor):
    """A non-differentiable PCA feature compressor that maintains running statistics."""
    
    def __init__(
        self,
        conf: DictConfig,
        feature_dim: int,
        device: Optional[str] = None
    ):
        """Initialize the PCA feature compressor.
        
        Args:
            conf: Hydra configuration containing PCA parameters
            device: Device to store tensors on (default: same as input tensors)
        """
        super().__init__(conf, feature_dim, device)
        
        # Parse PCA-specific configuration
        self.use_robust_pca = conf.use_robust_pca
        self.rpca_lambda = conf.rpca_lambda
        self.rpca_max_iter = conf.rpca_max_iter
        self.rpca_tol = conf.rpca_tol
        
        # Initialize running statistics
        self.register_buffer('running_mean', None)
        self.register_buffer('running_cov', None)
        self.register_buffer('n_samples', torch.tensor(0))
        
        # PCA components will be computed when needed
        self.pca_components = None
        self.eigenvalues = None
        
        # Flag to track if PCA needs recomputation
        self._needs_update = True
        
    def update_statistics(self, features: torch.Tensor) -> None:
        """Update running statistics from a batch of features.
        
        Args:
            features: Tensor of shape (N, feature_dim) containing features
        """
        if self.device is None:
            self.device = features.device
            self.to(self.device)
            
        # Ensure features are 2D
        if features.dim() > 2:
            features = features.reshape(-1, features.shape[-1])

        # Verify feature dimension
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim}, got {features.shape[-1]}")
        
        # Initialize buffers if needed
        if self.running_mean is None:
            self.register_buffer('running_mean', torch.zeros(self.feature_dim, device=self.device))
            self.register_buffer('running_cov', torch.zeros(self.feature_dim, self.feature_dim, device=self.device))
        elif self.running_mean.shape[0] != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.running_mean.shape[0]}, got {self.feature_dim}")
        
        # Update running mean
        n_new = features.shape[0]
        n_total = self.n_samples + n_new
        
        # Update mean
        self.running_mean = (
            (self.running_mean * self.n_samples + features.sum(0)) / n_total
        )
        
        # Update covariance
        centered_features = features - self.running_mean
        self.running_cov = (
            (self.running_cov * self.n_samples + 
             centered_features.T @ centered_features) / n_total
        )
        
        self.n_samples = n_total
        
        # Mark PCA for recomputation
        self._needs_update = True
        
    def compute_pca(self) -> None:
        """Compute PCA components from accumulated statistics."""
        if self.n_samples == 0:
            raise RuntimeError("No samples accumulated for PCA computation")
            
        if self.use_robust_pca:
            # Initialize matrices for ADMM
            L = self.running_cov.clone()  # Low-rank component
            S = torch.zeros_like(L)       # Sparse component
            Y = torch.zeros_like(L)       # Lagrange multiplier
            
            # ADMM iterations
            for _ in range(self.rpca_max_iter):
                # Update low-rank component (L)
                temp = self.running_cov - S + Y
                U, s, V = torch.linalg.svd(temp, full_matrices=False)
                s = torch.clamp(s - self.rpca_lambda, min=0)
                L = U @ torch.diag(s) @ V
                
                # Update sparse component (S)
                temp = self.running_cov - L + Y
                S = torch.clamp(temp, min=-self.rpca_lambda, max=self.rpca_lambda)
                
                # Update Lagrange multiplier (Y)
                Y = Y + (self.running_cov - L - S)
                
                # Check convergence
                if torch.norm(self.running_cov - L - S) < self.rpca_tol:
                    break
            
            # Use the low-rank component for PCA
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
        else:
            # Standard PCA using covariance matrix
            eigenvalues, eigenvectors = torch.linalg.eigh(self.running_cov)
            
        # Sort eigenvectors by eigenvalues in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        self.eigenvalues = eigenvalues[idx]
        self.pca_components = eigenvectors[:, idx]
        
        # Mark PCA as up to date
        self._needs_update = False
        
    def compress_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compress features using PCA.
        
        Args:
            features: Tensor of shape (..., feature_dim) containing features
            
        Returns:
            Compressed features of shape (..., n_components)
        """
        # Compute PCA if needed
        if self._needs_update:
            self.compute_pca()
            
        # Ensure features are 2D for PCA
        original_shape = features.shape
        features = features.reshape(-1, features.shape[-1])
        
        # Verify feature dimension
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim}, got {features.shape[-1]}")
        
        # Get components to use
        if self.n_components > self.pca_components.shape[1]:
            raise ValueError(f"Requested {self.n_components} components but only {self.pca_components.shape[1]} available")
        components = self.pca_components[:, :self.n_components]
        
        # Center and project
        centered_features = features - self.running_mean
        compressed = centered_features @ components
        
        # Restore original shape except for last dimension
        compressed = compressed.reshape(*original_shape[:-1], components.shape[1])
        
        return compressed
        
    def reconstruct_features(
        self,
        compressed: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct features from compressed representation.
        
        Args:
            compressed: Tensor of shape (..., n_components) containing compressed features
            
        Returns:
            Reconstructed features of shape (..., feature_dim)
        """
        # Ensure PCA is computed
        if self._needs_update:
            self.compute_pca()
            
        # Ensure compressed features are 2D for reconstruction
        original_shape = compressed.shape
        compressed = compressed.reshape(-1, compressed.shape[-1])
        
        # Get the components that were used for compression
        n_compressed_components = compressed.shape[-1]
        if n_compressed_components > self.pca_components.shape[1]:
            raise ValueError(f"Compressed features have {n_compressed_components} components but only {self.pca_components.shape[1]} available")
        components = self.pca_components[:, :n_compressed_components]
        
        # Reconstruct: multiply by components transpose and add back mean
        reconstructed = compressed @ components.T + self.running_mean
        
        # Restore original shape except for last dimension
        reconstructed = reconstructed.reshape(*original_shape[:-1], self.feature_dim)
        
        return reconstructed

    @torch.no_grad()
    def forward(
        self,
        features: torch.Tensor,
        update_stats: bool = False,
        compress_input: bool = True
    ) -> torch.Tensor:
        """Forward pass that either updates statistics or compresses features using PCA.
        
        Args:
            features: Tensor of shape (..., feature_dim) containing features
            update_stats: If True, only updates statistics and returns input features.
                          If False, compresses features using PCA.
            compress_input: If True, compresses features using PCA.
                            If False, returns input features without compression.
        Returns:
            If update_stats=True: Original features of shape (..., feature_dim)
            If update_stats=False: Compressed features of shape (..., n_components)
        """
        if update_stats:
            self.update_statistics(features)
            
        if not compress_input or self.n_components > features.shape[-1]:
            return features
            
        return self.compress_features(features) 