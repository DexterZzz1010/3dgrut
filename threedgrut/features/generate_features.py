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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from threedgrut.utils.logger import logger
from threedgrut.datasets import make as make_dataset
from threedgrut.datasets.protocols import BoundedMultiViewDataset
from threedgrut.features.extractors.nv_radio_feature_extractor import NVRadioFeatureExtractor
from threedgrut.features.compressors.pca_feature_compressor import PCAFeatureCompressor
from threedgrut.features.compressors.ae_feature_compressor import AutoencoderFeatureCompressor
from threedgrut.features.extractors import make_extractor
from threedgrut.features.compressors import make_compressor

def optimize_compressor(
    compressor: nn.Module,
    extractor: nn.Module,
    dataset: BoundedMultiViewDataset,
    conf: DictConfig
):
    """Optimize the feature compressor using the dataset.
    
    Args:
        compressor: Feature compressor to optimize
        extractor: Feature extractor to use
        dataset: Dataset to use for optimization
        conf: Configuration containing training settings
    """
    logger.info("Starting compressor optimization...")
    
    # Create data loader for parallel processing
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Keep batch size 1 to match trainer.py
        shuffle= conf.random_dataset_sample,
        num_workers=conf.num_workers,
        pin_memory=True,
        persistent_workers=True if conf.num_workers > 0 else False
    )
    
    if isinstance(compressor, PCAFeatureCompressor):
        # PCA optimization: collect statistics
        logger.info("Collecting PCA statistics...")
        for batch in tqdm(dataloader, desc="Updating PCA statistics"):
            # Get GPU batch and extract features
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            features = extractor(gpu_batch.rgb_gt)
            # Update PCA statistics
            compressor(features, update_stats=True, compress_input=False)
            
        # Compute PCA components
        logger.info("Computing PCA components...")
        compressor.compute_pca()
        
    elif isinstance(compressor, AutoencoderFeatureCompressor):
        # Autoencoder optimization: gradient-based training
        optimizer = torch.optim.Adam(
            compressor.parameters(),
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay
        )
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(conf.training.num_epochs):
            avg_loss = torch.tensor(float('inf'))
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{conf.training.num_epochs}")
            for batch in pbar:
                # Get GPU batch and extract features
                gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
                features = extractor(gpu_batch.rgb_gt)
                
                # Forward pass
                optimizer.zero_grad()
                compressed = compressor(features)
                reconstructed = compressor.decoder(compressed)
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(reconstructed, features)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f'{epoch_loss / num_batches:.6f}'})
            
            # Compute average loss
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{conf.training.num_epochs}, Average Loss: {avg_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= conf.training.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
    
    # Delete dataloader
    del dataloader

    logger.info("Compressor optimization complete")

def extract_and_save_features(
    compressor: nn.Module,
    extractor: nn.Module,
    dataset: BoundedMultiViewDataset,
    conf: DictConfig
):
    """Extract and save compressed features for all images."""
    
    # Create features directory if it doesn't exist
    os.makedirs(dataset.features_dir_path, exist_ok=True)

    # Save compressor
    torch.save(compressor.state_dict(), dataset.features_dir_path / "compressor.pt")
    
    # Save configuration
    OmegaConf.save(conf, dataset.features_dir_path / "config.yaml")
    
    # Extract and save features
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Keep batch size 1 to match trainer.py
        shuffle=False,
        num_workers=conf.num_workers,
        pin_memory=True,
        persistent_workers=True if conf.num_workers > 0 else False
    )
    
    compressor.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Genereating features dataset")):
            # Get GPU batch and extract features
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            features = extractor(gpu_batch.rgb_gt)
            
            # Compress features
            compressed = compressor(features)
            
            # Save features
            dataset.add_features(idx, compressed)

    # Delete dataloader
    del dataloader

@hydra.main(config_path="../../configs", version_base=None)
def main(conf: DictConfig):
    """Extract features from a dataset and optionally compress them.
    
    Args:
        conf: Configuration containing dataset, extractor, and compressor settings
    """
    # Create dataset
    train_dataset, val_dataset = make_dataset(name=conf.dataset.type, config=conf, ray_jitter=None)
    logger.info(f"Created dataset with {len(train_dataset)} images")
    
    # Create feature extractor
    extractor = make_extractor(conf.features.extractors)
    logger.info(f"Created feature extractor: {extractor.__class__.__name__}")
    
    # Create feature compressor if specified
    compressor = None
    if conf.features.compressors.type != "skip":
        compressor = make_compressor(conf.features.compressors, extractor.features_dim)
        logger.info(f"Created feature compressor: {compressor.__class__.__name__}")
        
        # Optimize compressor
        optimize_compressor(compressor, extractor, train_dataset, conf)
    
    # Extract and save features
    extract_and_save_features(compressor, extractor, train_dataset, conf)
    extract_and_save_features(compressor, extractor, val_dataset, conf)

if __name__ == "__main__":
    main() 