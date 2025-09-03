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
COLMAP G-Buffer Dataset

A dataset class that inherits from ColmapDataset and adds support for loading
G-buffers (basecolor, depth, metallic, normal, roughness) as concatenated feature maps.

This dataset is designed to work with G-buffers created using the create_colmap_gbuffers.py
script, which creates complete COLMAP G-buffer datasets from COLMAP images using cosmos1-diffusion-renderer.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import cv2

from .dataset_colmap import ColmapDataset
from .protocols import Batch

class ColmapGBufferDataset(ColmapDataset):
    """
    COLMAP dataset with G-buffer support.
    
    This dataset inherits from ColmapDataset and adds the ability to load
    G-buffers as concatenated feature maps along the channel dimension.
    
    The dataset expects a specific directory structure:
    - images/ or images_N/: Original COLMAP images (N indicates downsample factor)
    - gbuffers/: G-buffer files organized by image name
    - sparse/: COLMAP sparse reconstruction data
    - dataset_info.json: Metadata about the G-buffer extraction
    
    Each image has corresponding G-buffers with the format:
    - gbuffers/{image_name}/{frame}.{pass}.{type}.jpg
    Where:
    - image_name: Name of the original image (e.g., "DSCF5881")
    - frame: Frame number (e.g., "0000")
    - pass: Pass number (e.g., "0000")
    - type: G-buffer type (basecolor, depth, metallic, normal, roughness)
    """
    
    # G-buffer channel counts for each type
    GBUFFER_CHANNELS = {
        'basecolor': 3,  # RGB
        'depth': 1,      # Single channel
        'metallic': 1,   # Single channel
        'normal': 3,     # XYZ
        'roughness': 1,  # Single channel
    }
    
    # Order of G-buffer types for concatenation
    GBUFFER_ORDER = ['basecolor', 'depth', 'metallic', 'normal', 'roughness']
    
    def __init__(
        self,
        config,
        device="cuda",
        split="train",
        ray_jitter=None,
        gbuffer_types=None,           # List of G-buffer types to load (None = all)
        gbuffer_normalize=True,       # Whether to normalize G-buffers
        gbuffer_format="concat",      # How to handle G-buffers: "concat", "separate"
        load_gbuffers=True,          # Whether to load G-buffers at all
    ):
        """
        Initialize the ColmapGBufferDataset.
        
        Args:
            config: Dataset configuration
            device: Device to load tensors on
            split: Dataset split (train/val/test)
            ray_jitter: Ray jittering for training
            gbuffer_types: List of G-buffer types to load (None = all available)
            gbuffer_normalize: Whether to normalize G-buffers to [0,1]
            gbuffer_format: How to return G-buffers ("concat" or "separate")
            load_gbuffers: Whether to load G-buffers at all
        """
        # Initialize parent COLMAP dataset
        super().__init__(config, device, split, ray_jitter)
        
        self.gbuffer_types = gbuffer_types if gbuffer_types is not None else self.GBUFFER_ORDER
        self.gbuffer_normalize = gbuffer_normalize
        self.gbuffer_format = gbuffer_format
        self.load_gbuffers = load_gbuffers
        
        # Enable OpenEXR support for HDR files
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        
        if self.load_gbuffers:
            # Load G-buffer dataset metadata
            self._load_gbuffer_metadata()
            
            # Validate G-buffer availability
            self._validate_gbuffer_structure()
            
            # Cache G-buffer paths for faster loading
            self._cache_gbuffer_paths()
    
    def _load_gbuffer_metadata(self):
        """Load metadata about the G-buffer dataset."""
        metadata_path = Path(self.path) / "dataset_info.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"G-buffer metadata not found: {metadata_path}")
        
        try:
            with open(metadata_path, 'r') as f:
                self.gbuffer_metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load G-buffer metadata: {e}")
        
        # Validate metadata
        if self.gbuffer_metadata.get('type') != 'gbuffers_colmap':
            raise ValueError(f"Invalid dataset type: {self.gbuffer_metadata.get('type')} "
                           f"(expected 'gbuffers_colmap')")
        
        print(f"✅ Loaded G-buffer dataset metadata:")
        # Extract resolution from either statistics or gbuffer_info
        statistics = self.gbuffer_metadata.get('statistics', {})
        gbuffer_info = self.gbuffer_metadata.get('gbuffer_info', {})
        image_resolution = statistics.get('image_resolution', gbuffer_info.get('extraction_resolution', {}))
        
        width = image_resolution.get('width', 'Unknown')
        height = image_resolution.get('height', 'Unknown')
        num_files = statistics.get('num_gbuffer_files', 'Unknown')
        gbuffer_types = gbuffer_info.get('types', 'Unknown')
        
        print(f"   - Resolution: {width}×{height}")
        print(f"   - G-buffer files: {num_files}")
        print(f"   - G-buffer types: {gbuffer_types}")
    
    def _validate_gbuffer_structure(self):
        """Validate that the G-buffer directory structure exists."""
        self.gbuffer_dir = Path(self.path) / "gbuffers"
        
        if not self.gbuffer_dir.exists():
            raise FileNotFoundError(f"G-buffers directory not found: {self.gbuffer_dir}")
        
        # Check if any G-buffer files exist
        gbuffer_files = list(self.gbuffer_dir.rglob("*.jpg")) + list(self.gbuffer_dir.rglob("*.png"))
        if not gbuffer_files:
            raise ValueError(f"No G-buffer files found in {self.gbuffer_dir}")
        
        print(f"✅ G-buffer structure validated: {len(gbuffer_files)} files found")
    
    def _cache_gbuffer_paths(self):
        """Cache G-buffer file paths for each image for faster loading."""
        self.gbuffer_paths = {}
        
        print("🔍 Caching G-buffer paths...")
        
        for idx, image_path in enumerate(self.image_paths):
            # Extract image name without extension
            image_name = Path(image_path).stem
            
            # Find G-buffer directory for this image
            image_gbuffer_dir = self.gbuffer_dir / image_name
            
            if not image_gbuffer_dir.exists():
                print(f"⚠️  No G-buffers found for image: {image_name}")
                self.gbuffer_paths[idx] = {}
                continue
            
            # Find G-buffer files for each type
            gbuffer_files = {}
            for gbuffer_type in self.gbuffer_types:
                # Look for files matching pattern: {frame}.{pass}.{type}.jpg
                pattern = f"*.*.{gbuffer_type}.jpg"
                matching_files = list(image_gbuffer_dir.glob(pattern))
                
                if matching_files:
                    # Use the first matching file (assumes single frame/pass)
                    gbuffer_files[gbuffer_type] = matching_files[0]
                else:
                    print(f"⚠️  No {gbuffer_type} G-buffer found for {image_name}")
            
            self.gbuffer_paths[idx] = gbuffer_files
        
        # Count total available G-buffers
        total_gbuffers = sum(len(files) for files in self.gbuffer_paths.values())
        print(f"✅ Cached {total_gbuffers} G-buffer paths for {len(self.image_paths)} images")
    
    def _load_gbuffer_image(self, gbuffer_path: Path, gbuffer_type: str) -> np.ndarray:
        """
        Load a single G-buffer image.
        
        Args:
            gbuffer_path: Path to the G-buffer file
            gbuffer_type: Type of G-buffer (for type-specific processing)
            
        Returns:
            numpy array with shape (H, W, C) where C depends on G-buffer type
        """
        if not gbuffer_path.exists():
            raise FileNotFoundError(f"G-buffer file not found: {gbuffer_path}")
        
        # Load image using OpenCV to handle various formats
        image = cv2.imread(str(gbuffer_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load G-buffer image: {gbuffer_path}")
        
        # Convert BGR to RGB if it's a color image
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Handle single-channel G-buffers (depth, metallic, roughness)
        if gbuffer_type in ['depth', 'metallic', 'roughness']:
            if len(image.shape) == 3:
                # Take only the first channel for single-channel G-buffers
                image = image[:, :, 0]
            # Add channel dimension
            image = image[:, :, None]
        
        # Normalize to [0, 1] if requested
        if self.gbuffer_normalize:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            else:
                # Assume already normalized or handle other dtypes
                image = image.astype(np.float32)
        else:
            image = image.astype(np.float32)
        
        return image
    
    def _load_gbuffers_for_image(self, idx: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Load all G-buffers for a given image index.
        
        Args:
            idx: Image index
            
        Returns:
            Dictionary mapping G-buffer type to numpy array, or None if no G-buffers
        """
        if not self.load_gbuffers or idx not in self.gbuffer_paths:
            return None
        
        gbuffer_files = self.gbuffer_paths[idx]
        if not gbuffer_files:
            return None
        
        gbuffers = {}
        for gbuffer_type in self.gbuffer_types:
            if gbuffer_type in gbuffer_files:
                try:
                    gbuffer = self._load_gbuffer_image(gbuffer_files[gbuffer_type], gbuffer_type)
                    gbuffers[gbuffer_type] = gbuffer
                except Exception as e:
                    print(f"⚠️  Failed to load {gbuffer_type} for image {idx}: {e}")
        
        return gbuffers if gbuffers else None
    
    def _concatenate_gbuffers(self, gbuffers: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Concatenate G-buffers along the channel dimension.
        
        Args:
            gbuffers: Dictionary of G-buffer arrays
            
        Returns:
            Concatenated G-buffer array with shape (H, W, total_channels)
        """
        if not gbuffers:
            return None
        
        # Get reference dimensions from first G-buffer
        first_gbuffer = next(iter(gbuffers.values()))
        h, w = first_gbuffer.shape[:2]
        
        # Concatenate in the specified order
        concat_list = []
        for gbuffer_type in self.gbuffer_types:
            if gbuffer_type in gbuffers:
                gbuffer = gbuffers[gbuffer_type]
                
                # Ensure consistent dimensions
                if gbuffer.shape[:2] != (h, w):
                    print(f"⚠️  Resizing {gbuffer_type} G-buffer from {gbuffer.shape[:2]} to {(h, w)}")
                    gbuffer = cv2.resize(gbuffer, (w, h), interpolation=cv2.INTER_LINEAR)
                    if len(gbuffer.shape) == 2:
                        gbuffer = gbuffer[:, :, None]
                
                concat_list.append(gbuffer)
        
        if not concat_list:
            return None
        
        # Concatenate along channel dimension
        concatenated = np.concatenate(concat_list, axis=2)
        return concatenated
    
    def get_gbuffer_channel_count(self) -> int:
        """Get the total number of channels in the concatenated G-buffer."""
        if not self.load_gbuffers:
            return 0
        
        total_channels = 0
        for gbuffer_type in self.gbuffer_types:
            if gbuffer_type in self.GBUFFER_CHANNELS:
                total_channels += self.GBUFFER_CHANNELS[gbuffer_type]
        return total_channels
    
    def get_gbuffer_channel_names(self) -> List[str]:
        """Get the names of channels in the concatenated G-buffer."""
        if not self.load_gbuffers:
            return []
        
        channel_names = []
        for gbuffer_type in self.gbuffer_types:
            if gbuffer_type == 'basecolor':
                channel_names.extend([f'basecolor_r', f'basecolor_g', f'basecolor_b'])
            elif gbuffer_type == 'normal':
                channel_names.extend([f'normal_x', f'normal_y', f'normal_z'])
            else:
                channel_names.append(gbuffer_type)
        return channel_names
    
    def __getitem__(self, idx) -> dict:
        """Get dataset item with G-buffers included."""
        # Get the base dataset item
        output_dict = super().__getitem__(idx)
        
        # Load G-buffers if enabled
        if self.load_gbuffers:
            gbuffers = self._load_gbuffers_for_image(idx)
            
            if gbuffers is not None:
                if self.gbuffer_format == "concat":
                    # Concatenate G-buffers into a single tensor
                    gbuffer_concat = self._concatenate_gbuffers(gbuffers)
                    if gbuffer_concat is not None:
                        output_dict["gbuffers"] = torch.tensor(gbuffer_concat).unsqueeze(0)
                
                elif self.gbuffer_format == "separate":
                    # Keep G-buffers separate
                    gbuffer_tensors = {}
                    for gbuffer_type, gbuffer_array in gbuffers.items():
                        gbuffer_tensors[f"gbuffer_{gbuffer_type}"] = torch.tensor(gbuffer_array).unsqueeze(0)
                    output_dict.update(gbuffer_tensors)
        
        return output_dict
    
    def get_gpu_batch_with_intrinsics(self, batch):
        """Add the intrinsics to the batch and move data to GPU, including G-buffers."""
        # Get the base batch from parent  
        base_sample = super().get_gpu_batch_with_intrinsics(batch)
        
        # Convert Batch object to dictionary to add G-buffers
        sample_dict = {}
        for field in base_sample.__dataclass_fields__:
            value = getattr(base_sample, field)
            if value is not None:
                sample_dict[field] = value
        
        # Add G-buffers to the sample if present
        if "gbuffers" in batch:
            gbuffers = batch["gbuffers"][0].to(self.device, non_blocking=True)
            sample_dict["gbuffers"] = gbuffers
        
        # Add separate G-buffers if present
        for key in batch.keys():
            if key.startswith("gbuffer_"):
                gbuffer = batch[key][0].to(self.device, non_blocking=True)
                sample_dict[key] = gbuffer
        
        return Batch(**sample_dict)
    
    def create_dataset_camera_visualization(self):
        """Create a visualization of the dataset cameras with G-buffer preview."""
        # Use parent visualization but show G-buffers instead of RGB if available
        cam_list = []
        
        for i_cam, pose in enumerate(self.poses):
            trans_mat = pose
            trans_mat_world_to_camera = np.linalg.inv(trans_mat)
            
            # Camera convention rotation
            camera_convention_rot = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            trans_mat_world_to_camera = camera_convention_rot @ trans_mat_world_to_camera
            
            # Get camera ID and corresponding intrinsics
            camera_id = self.get_intrinsics_idx(i_cam)
            intr, _, _, _ = self.intrinsics[camera_id]
            
            # Load actual image to get dimensions
            image_name = self._extract_relative_image_name(self.image_paths[i_cam])
            image = self._load_image_with_fallback(image_name, save_downsampled=self.save_downsampled_images)
            image_data = np.asarray(image)
            image.close()
            h, w = image_data.shape[:2]
            
            # Try to load G-buffers for visualization
            gbuffers = self._load_gbuffers_for_image(i_cam)
            if gbuffers is not None and 'basecolor' in gbuffers:
                # Use basecolor G-buffer for visualization
                vis_image = gbuffers['basecolor']
                if vis_image.shape[:2] != (h, w):
                    vis_image = cv2.resize(vis_image, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                # Fallback to RGB image
                vis_image = image_data.astype(np.float32) / 255.0
            
            f_w = intr["focal_length"][0]
            f_h = intr["focal_length"][1]
            
            fov_w = 2.0 * np.arctan(0.5 * w / f_w)
            fov_h = 2.0 * np.arctan(0.5 * h / f_h)
            
            cam_list.append({
                "ext_mat": trans_mat_world_to_camera,
                "w": w,
                "h": h,
                "fov_w": fov_w,
                "fov_h": fov_h,
                "rgb_img": vis_image,
                "split": self.split,
            })
        
        from .utils import create_camera_visualization
        create_camera_visualization(cam_list)
