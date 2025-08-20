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

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from omegaconf import DictConfig

class FeatureDataset:
    """Base class for handling feature loading functionality."""
    
    def __init__(self, conf: DictConfig):
        """Initialize the feature dataset.
        
        Args:
            conf: Configuration containing feature loading parameters
        """
        self.conf = conf
        self.save_features_vizualization = False if not 'save_features_vizualization' in conf.dataset else conf.dataset.save_features_vizualization

        self.load_features_enabled = conf.dataset.load_features 
        self._setup_feature_dir()

        if self.load_features_enabled and not self.features_dir.exists():
            raise ValueError(f"Feature directory {self.features_dir} does not exist")

    def _setup_feature_dir(self):
        """Setup feature loading paths and parameters."""
        
        # Setup feature directory with both extractor and compressor information
        feature_path = Path(self.conf.path) / "features"
        
        # Add extractor type to path
        feature_path = feature_path / self.conf.features.extractors.type
        
        # Add compressor information if present
        if self.conf.features.compressors.type != "skip":
            feature_path = feature_path / f"{self.conf.features.compressors.type}_{self.conf.features.compressors.n_components}"
            
        self.features_dir = feature_path
            
    @property
    def features_dir_path(self) -> str:
        return self.features_dir

    def add_features(self, idx: int, feature: torch.Tensor):
        """Add a feature to the feature dataset.
        
        Args:
            idx: The index of the feature
            feature: The feature to save
        """
        raise NotImplementedError("Subclasses must implement this method")

    def load_features(self, image_path: str) -> Optional[torch.Tensor]:
        """Load features for a given index.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Loaded features as a tensor, or None if features don't exist
        """
        if not self.load_features_enabled:
            return None
            
        feature_path = self.features_dir / f"{os.path.basename(image_path)}.pt"
        if not feature_path.exists():
            return None
            
        try:
            features = torch.load(feature_path)
            return features
        except Exception as e:
            print(f"Error loading features from {feature_path}: {e}")
            return None
    
    def save_features(self, image_path: str, features: torch.Tensor):
        """Save a feature to the feature dataset.
        
        Args:
            image_path: The path to the image
            feature: The feature to save
        """
        torch.save(features.cpu(), self.features_dir / f"{os.path.basename(image_path)}.pt")

        if self.save_features_vizualization:
            # Save visualization of first 3 channels of features
            import torchvision
            feature_viz = features[0, :, :, :3].cpu() # Take first 3 channels
            feature_viz = (feature_viz - feature_viz.min()) / (feature_viz.max() - feature_viz.min()) # Normalize to [0,1]
            torchvision.utils.save_image(
                feature_viz.permute(2,0,1), # Convert to (C,H,W) format
                self.features_dir / f"{os.path.basename(image_path)}_viz.png"
            )

    def get_feature_shape(self) -> Optional[tuple]:
        """Get the shape of the features.
        
        Returns:
            Tuple containing feature dimensions, or None if features are not loaded
        """
        if not self.load_features_enabled:
            return None
            
        # Try to load the first feature file to get the shape
        for idx in range(1000):  # Look for first 1000 indices
            features = self.load_features(idx)
            if features is not None:
                return features.shape
        return None 