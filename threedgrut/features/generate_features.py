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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import PeakSignalNoiseRatio

from threedgrut.utils.logger import logger
from threedgrut.datasets import make as make_dataset
from threedgrut.datasets.protocols import BoundedMultiViewDataset
from threedgrut.features.extractors.nv_radio_feature_extractor import NVRadioFeatureExtractor
from threedgrut.features.compressors.pca_feature_compressor import PCAFeatureCompressor
from threedgrut.features.compressors.ae_feature_compressor import AutoencoderFeatureCompressor
from threedgrut.features.extractors import make_extractor
from threedgrut.features.compressors import make_compressor

def create_sample_image_tensor(features, max_samples=8):
    """Create a tensor of sample images from features using first 3 components.
    
    Args:
        features: Features tensor [B, H, W, C] where C >= 3
        max_samples: Maximum number of samples to include
        
    Returns:
        Image tensor [N, 3, H, W] clamped to [0, 1] for visualization
    """
    if features.shape[-1] < 3:
        return None
    
    # Take first 3 components and clamp to [0, 1]
    rgb_features = features[..., :3].clamp(0, 1)
    
    # Take up to max_samples
    n_samples = min(max_samples, rgb_features.shape[0])
    sample_features = rgb_features[:n_samples]
    
    # Convert to [N, C, H, W] format for tensorboard
    return sample_features.permute(0, 3, 1, 2)

def compute_reconstruction_psnr(original, reconstructed, range=None):
    """Compute PSNR between original and reconstructed features.
    
    Args:
        original: Original features [B, H, W, C]
        reconstructed: Reconstructed features [B, H, W, C]
        
    Returns:
        PSNR value as float
    """
    psnr_metric = PeakSignalNoiseRatio(data_range=range)
    # Move metric to the same device as input tensors
    psnr_metric = psnr_metric.to(original.device)
    return psnr_metric(reconstructed, original).item()

def log_eigenvalue_energy(eigenvalues, writer: SummaryWriter):
    """Log eigenvalue energy distribution to TensorBoard.
    
    Args:
        eigenvalues: Eigenvalues tensor sorted in descending order
        writer: TensorBoard writer
    """
    eigenvalues_np = eigenvalues.cpu().numpy()
    
    # Plot cumulative explained variance
    cumsum = np.cumsum(eigenvalues_np)
    total_variance = cumsum[-1]
    explained_variance_ratio = cumsum / total_variance
    
    for i in range(len(eigenvalues_np)):
        writer.add_scalar(f'PCA/eigenvalues', eigenvalues_np[i], i)
    for i in range(len(explained_variance_ratio)):
        writer.add_scalar(f'PCA/explained_variance_ratio', explained_variance_ratio[i], i)

def optimize_compressor(
    compressor: nn.Module,
    extractor: nn.Module,
    dataset: BoundedMultiViewDataset,
    conf: DictConfig,
    writer: SummaryWriter = None
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
        
        # Log PCA eigenvalue energy to TensorBoard
        if writer is not None and hasattr(compressor, 'eigenvalues') and compressor.eigenvalues is not None:
            logger.info("Logging PCA eigenvalue energy to TensorBoard...")
            log_eigenvalue_energy(compressor.eigenvalues, writer)
        
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
            
            # Log training loss to TensorBoard
            if writer is not None:
                writer.add_scalar('Autoencoder/training_loss', avg_loss, epoch)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                # Log best loss to TensorBoard
                if writer is not None:
                    writer.add_scalar('Autoencoder/best_loss', best_loss, epoch)
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
    conf: DictConfig,
    writer: SummaryWriter = None,
    split_name: str = "train"
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
    psnr_values = []
    sample_images_original = []
    sample_images_compressed = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Generating features dataset")):
            # Get GPU batch and extract features
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            features = extractor(gpu_batch.rgb_gt)
            
            # Compress features
            compressed = compressor(features)
            
            # Save features
            dataset.add_features(idx, compressed)
            
            # For TensorBoard logging
            if writer is not None:
                # Compute PSNR between original and compressed features
                if hasattr(compressor, 'decoder'):
                    # For autoencoder, reconstruct features
                    reconstructed = compressor.decoder(compressed)
                    psnr = compute_reconstruction_psnr(features, reconstructed)
                    psnr_values.append(psnr)
                elif hasattr(compressor, 'decompress'):
                    # For PCA, decompress features
                    reconstructed = compressor.decompress(compressed)
                    psnr = compute_reconstruction_psnr(features, reconstructed)
                    psnr_values.append(psnr)
                
                # Collect sample images (first 8 samples)
                if idx < 8:
                    original_img = create_sample_image_tensor(features, max_samples=1)
                    if original_img is not None:
                        sample_images_original.append(original_img)
                        
                        # Get compressed version for visualization
                        if hasattr(compressor, 'decoder'):
                            reconstructed = compressor.decoder(compressed)
                        elif hasattr(compressor, 'decompress'):
                            reconstructed = compressor.decompress(compressed)
                        else:
                            reconstructed = compressed  # For identity/skip compressor
                        
                        compressed_img = create_sample_image_tensor(reconstructed, max_samples=1)
                        if compressed_img is not None:
                            sample_images_compressed.append(compressed_img)
    
    # Log to TensorBoard
    if writer is not None:
        # Log final mean PSNR
        if psnr_values:
            mean_psnr = np.mean(psnr_values)
            logger.info(f"Mean PSNR for {split_name} set: {mean_psnr:.2f} dB")
            writer.add_scalar(f'PSNR/final_mean_{split_name}', mean_psnr, 0)
        
        # Log sample images
        if sample_images_original and sample_images_compressed:
            original_grid = torch.cat(sample_images_original, dim=0)
            compressed_grid = torch.cat(sample_images_compressed, dim=0)
            
            writer.add_images(f'Samples/{split_name}_original_features', original_grid, 0)
            writer.add_images(f'Samples/{split_name}_compressed_features', compressed_grid, 0)
            
            logger.info(f"Logged {len(sample_images_original)} sample images to TensorBoard")

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
    
    # Setup TensorBoard logging
    log_dir = train_dataset.features_dir_path / "logs"
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")
    
    # Create feature compressor if specified
    compressor = None
    if conf.features.compressors.type != "skip":
        compressor = make_compressor(conf.features.compressors, extractor.features_dim)
        logger.info(f"Created feature compressor: {compressor.__class__.__name__}")
        
        # Optimize compressor
        optimize_compressor(compressor, extractor, train_dataset, conf, writer)
    
    # Extract and save features
    extract_and_save_features(compressor, extractor, train_dataset, conf, writer, "train")
    extract_and_save_features(compressor, extractor, val_dataset, conf, writer, "val")
    
    # Close TensorBoard writer
    writer.close()
    logger.info("Feature extraction complete. TensorBoard logs saved.")

if __name__ == "__main__":
    main() 