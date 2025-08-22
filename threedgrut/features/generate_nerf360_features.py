# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from threedgrut.utils.logger import logger
from threedgrut.datasets import make as make_dataset
from threedgrut.features.extractors import make_extractor
from threedgrut.features.compressors import make_compressor
from threedgrut.features.generate_features import optimize_compressor, extract_and_save_features

def process_scene(conf: DictConfig, scene: str, downsample_factor: int) -> bool:
    """Process a single MipNeRF 360 scene by calling generate_features logic.
    
    Args:
        conf: Hydra configuration
        scene: Scene name
        downsample_factor: Downsample factor for the scene
        
    Returns:
        True if successful, False otherwise
    """
    base_path = Path(conf.path)
    scene_path = base_path / scene
    
    # Check if scene directory exists
    if not scene_path.exists():
        logger.warning(f"Scene directory {scene_path} does not exist. Skipping {scene}.")
        return False
    
    logger.info(f"Processing scene: {scene}")
    
    try:
        # Create a copy of the config and update scene-specific settings
        scene_conf = OmegaConf.create(OmegaConf.to_yaml(conf))
        scene_conf.path = str(scene_path)
        scene_conf.dataset.downsample_factor = downsample_factor
        
        # Call the core logic from generate_features.main()
        # Create dataset
        train_dataset, val_dataset = make_dataset(name=scene_conf.dataset.type, config=scene_conf, ray_jitter=None)
        logger.info(f"Created dataset with {len(train_dataset)} images for {scene}")
        
        # Create feature extractor
        extractor = make_extractor(scene_conf.features.extractors)
        logger.info(f"Created feature extractor: {extractor.__class__.__name__}")
        
        # Setup TensorBoard logging
        log_dir = train_dataset.features_dir_path / "logs"
        writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard logs will be saved to: {log_dir}")
        
        # Create feature compressor if specified
        compressor = None
        if scene_conf.features.compressors.type != "skip":
            compressor = make_compressor(scene_conf.features.compressors, extractor.features_dim)
            logger.info(f"Created feature compressor: {compressor.__class__.__name__}")
            
            # Optimize compressor
            optimize_compressor(compressor, extractor, train_dataset, scene_conf, writer)
        
        # Extract and save features
        extract_and_save_features(compressor, extractor, train_dataset, scene_conf, writer, "train")
        extract_and_save_features(compressor, extractor, val_dataset, scene_conf, writer, "val")
        
        # Close TensorBoard writer
        writer.close()
        logger.info(f"Feature extraction complete for {scene}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to generate features for {scene}: {e}")
        return False

@hydra.main(config_path="../../configs", version_base=None)
def main(conf: DictConfig):
    """Generate features for all MipNeRF 360 scenes.
    
    This script processes MipNeRF 360 scenes with their appropriate downsample 
    factors as defined in the configuration. The base dataset path and scenes
    should be set in the configuration file.
    
    Args:
        conf: Configuration containing dataset path, scenes, and feature extraction settings
    """
    # Get configuration parameters
    base_path = Path(conf.path)
    
    # Check if scenes are defined in config
    if not hasattr(conf, 'scenes') or not conf.scenes:
        raise ValueError("No scenes defined in configuration. Please define 'scenes' in your config file with scene names and downsample factors.")
    
    scenes_config = conf.scenes
    skip_existing = getattr(conf, 'skip_existing', False)
    
    # Validate scenes config format
    if not isinstance(scenes_config, dict):
        raise ValueError("Scenes configuration must be a dictionary with scene names as keys and downsample_factor as values.")
    
    scenes_to_process = list(scenes_config.keys())
    
    logger.info(f"Processing scenes from: {base_path}")
    logger.info(f"Scenes to process: {scenes_to_process}")
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base path {base_path} does not exist")
    
    # Track results
    successful_scenes = []
    failed_scenes = []
    skipped_scenes = []
    
    for scene in scenes_to_process:
        scene_info = scenes_config[scene]
        
        # Validate scene configuration
        if not isinstance(scene_info, dict):
            raise ValueError(f"Scene '{scene}' configuration must be a dictionary with 'downsample_factor' key")
        
        if 'downsample_factor' not in scene_info:
            raise ValueError(f"Scene '{scene}' is missing required 'downsample_factor' parameter")
        
        downsample_factor = scene_info['downsample_factor']
        
        scene_path = base_path / scene
        
        # Check if features already exist
        if skip_existing:
            features_path = scene_path / "features"
            if features_path.exists() and (features_path / "compressor.pt").exists():
                logger.info(f"Features already exist for {scene}, skipping")
                skipped_scenes.append(scene)
                continue
        
        success = process_scene(conf, scene, downsample_factor)
        
        if success:
            successful_scenes.append(scene)
        else:
            failed_scenes.append(scene)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("FEATURE GENERATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total scenes processed: {len(scenes_to_process)}")
    logger.info(f"Successful: {len(successful_scenes)} - {successful_scenes}")
    if skipped_scenes:
        logger.info(f"Skipped: {len(skipped_scenes)} - {skipped_scenes}")
    if failed_scenes:
        logger.error(f"Failed: {len(failed_scenes)} - {failed_scenes}")
    else:
        logger.info("All scenes completed successfully!")

if __name__ == "__main__":
    main()
