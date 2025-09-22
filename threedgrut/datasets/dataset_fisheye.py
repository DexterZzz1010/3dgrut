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
import numpy as np
from threedgrut.utils.logger import logger

from .dataset_scannetpp import ScannetppDataset
from .utils import read_colmap_extrinsics_text, read_colmap_intrinsics_text


class FisheyeDataset(ScannetppDataset):
    """
    Fisheye dataset that extends ScannetppDataset with configurable image filtering and mask support.
    
    All filtering rules and mask configurations are defined in configuration files, not hardcoded.
    This follows the principle of "good taste" - configuration over code.
    """

    def __init__(
        self,
        path,
        device="cuda",
        split="train",
        downsample_factor=1,
        test_split_interval=8,
        ray_jitter=None,
        image_name_filter=None,
        mask_config=None,  # 新增mask配置参数
    ):
        self.image_name_filter = image_name_filter or {}
        self.mask_config = mask_config or {}
        
        # 调用父类构造函数
        super(FisheyeDataset, self).__init__(
            path, device, split, downsample_factor, test_split_interval, ray_jitter
        )

    def _should_include_image(self, image_name: str) -> bool:
        """
        判断是否应该包含指定的图像文件。
        
        完全由配置文件中的 image_name_filter 控制，不包含任何硬编码规则。
        """
        if not self.image_name_filter.get('enabled', False):
            return True
            
        include_prefixes = self.image_name_filter.get('include_prefixes', [])
        exclude_prefixes = self.image_name_filter.get('exclude_prefixes', [])
        
        if include_prefixes:
            if not any(image_name.startswith(prefix) for prefix in include_prefixes):
                return False
                
        if exclude_prefixes:
            if any(image_name.startswith(prefix) for prefix in exclude_prefixes):
                return False
                
        return True

    def reload(self):
        """
        重写reload方法，在父类完成后根据配置应用自定义mask路径
        """
        super().reload()
        
        if self.mask_config.get('enabled', False):
            source_type = self.mask_config.get('source_type', 'standard')
            if source_type == 'colmap_structure':
                self._setup_colmap_mask_paths()
                logger.info("Applied colmap structure mask paths")
        else:
            logger.info("Mask disabled by configuration")

    def _setup_colmap_mask_paths(self):
            """
            为colmap/masks_rectified/结构设置mask路径
            """
            base_path = self.mask_config.get('base_path', 'colmap/masks_rectified')
            extensions = self.mask_config.get('file_extensions', ['.png', '.jpg'])
            fallback = self.mask_config.get('fallback_to_standard', False)
            
            logger.info(f"Setting up colmap mask paths from: {base_path}")
            
            custom_mask_paths = []
            masks_found = 0
            
            for extr in self.cam_extrinsics:
                image_name = extr.name 
                base_name = os.path.splitext(image_name)[0]
                
                mask_candidates = []
                for ext in extensions:
                    mask_candidates.append(
                        os.path.join(self.path, base_path, base_name + ext)
                    )
                mask_candidates.append(
                    os.path.join(self.path, base_path, image_name)
                )
                
                selected_path = None
                for candidate in mask_candidates:
                    if os.path.exists(candidate):
                        selected_path = candidate
                        masks_found += 1
                        break
                
                if selected_path is None:
                    if fallback:
                        image_path = os.path.join(self.path, self.get_images_folder(), image_name)
                        selected_path = os.path.splitext(image_path)[0] + "_mask.png"
                    else:
                        selected_path = mask_candidates[0]
                        
                custom_mask_paths.append(selected_path)
            
            self.mask_paths = np.array(custom_mask_paths, dtype=str)
            
            logger.info(f"Colmap mask setup: {masks_found}/{len(custom_mask_paths)} masks found")
            
            if masks_found == 0:
                logger.warning("No mask files found! Training will proceed without masks.")
            elif masks_found < len(custom_mask_paths):
                missing = len(custom_mask_paths) - masks_found
                logger.warning(f"{missing} mask files are missing and will be ignored")

    def load_intrinsics_and_extrinsics(self):
        """
        加载内参和外参，并应用配置文件中定义的图像过滤规则。
        """
        cameras_extrinsic_file = os.path.join(self.path, "colmap", "images.txt")
        cameras_intrinsic_file = os.path.join(self.path, "colmap", "cameras.txt")
        cam_extrinsics = read_colmap_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_colmap_intrinsics_text(cameras_intrinsic_file)

        for intr in cam_intrinsics.values():
            intr.params[4:] = 0.0

        if self.image_name_filter.get('enabled', False):
            original_count = len(cam_extrinsics)
            filtered_extrinsics = []
            filtered_names = []
            used_camera_ids = set()
            
            for extr in cam_extrinsics:
                if self._should_include_image(extr.name):
                    filtered_extrinsics.append(extr)
                    filtered_names.append(extr.name)
                    used_camera_ids.add(extr.camera_id)
            
            cam_extrinsics = filtered_extrinsics
            filtered_count = len(cam_extrinsics)
            
            original_intrinsics_count = len(cam_intrinsics)
            filtered_intrinsics = {
                cam_id: intr for cam_id, intr in cam_intrinsics.items() 
                if cam_id in used_camera_ids
            }
            cam_intrinsics = filtered_intrinsics
            
            logger.info(f"Image filtering: {original_count} -> {filtered_count} images")
            logger.info(f"Camera filtering: {original_intrinsics_count} -> {len(cam_intrinsics)} cameras")
            logger.info(f"Filter config - Include: {self.image_name_filter.get('include_prefixes', [])}")
            logger.info(f"Filter config - Exclude: {self.image_name_filter.get('exclude_prefixes', [])}")
            
            if filtered_count == 0:
                raise ValueError(
                    f"No images remain after filtering! "
                    f"Check your image_name_filter configuration. "
                    f"Include: {self.image_name_filter.get('include_prefixes', [])} "
                    f"Exclude: {self.image_name_filter.get('exclude_prefixes', [])}"
                )
            
            sample_names = filtered_names[:min(5, len(filtered_names))]
            logger.info(f"Sample included images: {sample_names}")
            logger.info(f"Used camera IDs: {sorted(used_camera_ids)}")

        self.cam_extrinsics = cam_extrinsics
        self.cam_intrinsics = cam_intrinsics

    def get_images_folder(self):
        """
        返回图像文件夹名称。
        """
        return "image_undistorted_fisheye"