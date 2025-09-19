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
    Fisheye dataset that extends ScannetppDataset with configurable image filtering.
    
    All filtering rules are defined in configuration files, not hardcoded.
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
        image_name_filter=None,  # 完全由配置文件控制
    ):
        self.image_name_filter = image_name_filter or {}
        
        # 调用父类构造函数
        super(FisheyeDataset, self).__init__(
            path, device, split, downsample_factor, test_split_interval, ray_jitter
        )

        # if hasattr(self, '_custom_mask_paths'):
        #     self.mask_paths = np.array(self._custom_mask_paths, dtype=str)

    def _should_include_image(self, image_name: str) -> bool:
        """
        判断是否应该包含指定的图像文件。
        
        完全由配置文件中的 image_name_filter 控制，不包含任何硬编码规则。
        
        Args:
            image_name: 图像文件名（可能包含路径）
            
        Returns:
            True if the image should be included, False otherwise
        """
        if not self.image_name_filter.get('enabled', False):
            return True
            
        # 获取include和exclude前缀列表 - 完全来自配置
        include_prefixes = self.image_name_filter.get('include_prefixes', [])
        exclude_prefixes = self.image_name_filter.get('exclude_prefixes', [])
        
        # 如果指定了include_prefixes，必须匹配其中之一
        if include_prefixes:
            if not any(image_name.startswith(prefix) for prefix in include_prefixes):
                return False
                
        # 如果匹配任何exclude_prefixes，则排除
        if exclude_prefixes:
            if any(image_name.startswith(prefix) for prefix in exclude_prefixes):
                return False
                
        return True

    def load_intrinsics_and_extrinsics(self):
        """
        加载内参和外参，并应用配置文件中定义的图像过滤规则。
        
        这个方法完全继承了ScannetppDataset的逻辑，但在加载后应用过滤。
        """
        # 使用父类的路径逻辑
        cameras_extrinsic_file = os.path.join(self.path, "colmap", "images.txt")
        cameras_intrinsic_file = os.path.join(self.path, "colmap", "cameras.txt")
        cam_extrinsics = read_colmap_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_colmap_intrinsics_text(cameras_intrinsic_file)

        # 移除相机畸变参数（继承自ScannetppDataset的逻辑）
        for intr in cam_intrinsics.values():
            intr.params[4:] = 0.0

        # 应用配置文件中定义的图像过滤规则
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
                # else:
                #     logger.info(f"Filtered out image: {extr.name}")
            
            cam_extrinsics = filtered_extrinsics
            filtered_count = len(cam_extrinsics)
            
            # 同步过滤内参 - 只保留被使用的相机ID
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
            
            # 显示前几个被包含的图像名称用于验证
            sample_names = filtered_names[:min(5, len(filtered_names))]
            logger.info(f"Sample included images: {sample_names}")
            logger.info(f"Used camera IDs: {sorted(used_camera_ids)}")

        self.cam_extrinsics = cam_extrinsics
        self.cam_intrinsics = cam_intrinsics

        # self._setup_custom_mask_paths()

    def _setup_custom_mask_paths(self):
        """为colmap/masks_rectified/结构设置mask路径"""
        logger.info("Setting up custom mask paths for colmap/masks_rectified/ structure")
        
        custom_mask_paths = []
        masks_found = 0
        
        for extr in self.cam_extrinsics:  # 使用已过滤的外参
            image_name = extr.name  # 例如: "FC/original_images_000000.jpg"
            base_name = os.path.splitext(image_name)[0]  # "FC/original_images_000000" 
            
            # 尝试不同的mask文件扩展名
            mask_candidates = [
                os.path.join(self.path, "colmap/masks_rectified", base_name + ".png"),
                os.path.join(self.path, "colmap/masks_rectified", base_name + ".jpg"),
                os.path.join(self.path, "colmap/masks_rectified", image_name),  # 原扩展名
            ]
            
            # 选择第一个存在的文件
            selected_path = mask_candidates[0]  # 默认.png
            for candidate in mask_candidates:
                if os.path.exists(candidate):
                    selected_path = candidate
                    masks_found += 1
                    break
                    
            custom_mask_paths.append(selected_path)
        
        logger.info(f"Custom mask setup: {masks_found}/{len(custom_mask_paths)} masks found")
        
        self._custom_mask_paths = custom_mask_paths

    def get_images_folder(self):
        """
        返回图像文件夹名称。
        
        继承ScannetppDataset的逻辑，返回 'image_undistorted_fisheye'
        """
        return "image_undistorted_fisheye"