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

from omegaconf import DictConfig
from .nv_radio_feature_extractor import NVRadioFeatureExtractor

def make_extractor(conf: DictConfig):
    """Create a feature extractor based on configuration.
    
    Args:
        conf: Configuration containing extractor settings
        
    Returns:
        Feature extractor instance
        
    Raises:
        ValueError: If extractor type is not supported
    """
    if conf.type == "nv_radio":
        return NVRadioFeatureExtractor(conf)
    else:
        raise ValueError(f"Unknown extractor type: {conf.type}") 