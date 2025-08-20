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

from typing import Optional
from omegaconf import DictConfig
from .feature_compressor import FeatureCompressor
from .pca_feature_compressor import PCAFeatureCompressor
from .ae_feature_compressor import AutoencoderFeatureCompressor

def make_compressor(
    conf: DictConfig,
    feature_dim: int,
    device: Optional[str] = None,
) -> FeatureCompressor:
    """Create a feature compressor based on the configuration.
    
    Args:
        conf: Hydra configuration containing compressor parameters
        device: Device to store tensors on (default: same as input tensors)
        
    Returns:
        A FeatureCompressor instance
        
    Raises:
        ValueError: If the compressor type is not supported
    """
    compressor_type = conf.type.lower()
    
    if compressor_type == "pca":
        return PCAFeatureCompressor(conf, feature_dim, device)
    elif compressor_type == "autoencoder":
        return AutoencoderFeatureCompressor(conf, feature_dim, device)
    else:
        raise ValueError(f"Unknown compressor type: {compressor_type}. Must be one of ['pca', 'autoencoder']") 