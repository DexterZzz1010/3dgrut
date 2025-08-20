# Feature Extraction and Compression Pipeline

This module provides functionality for extracting and compressing features from images using various models and compression techniques.

## Overview

The pipeline consists of two main components:
1. **Feature Extractors**: Extract features from images using pre-trained models
2. **Feature Compressors**: Compress the extracted features to reduce dimensionality

### Available Feature Extractors

- **RADIO**: NVIDIA's RADIO model for feature extraction
  - Configuration: `configs/features/extractors/nv_radio.yaml`
  - Model options: `radio_v2.5-h`, `radio_v2.5-m`, `radio_v2.5-l`
  - Features:
    - Automatic resolution adjustment
    - Mixed precision inference (bfloat16)
    - Support for both standard and E-RADIO models
    - BHWC/BCHW format handling

### Available Feature Compressors

- **PCA**: Principal Component Analysis
  - Configuration: `configs/features/compressors/pca.yaml`
  - Parameters:
    - `n_components`: Number of dimensions to compress to
    - `feature_dim`: Input feature dimension
    - `use_robust_pca`: Whether to use robust PCA (handles outliers)
    - `rpca_lambda`: Regularization parameter for robust PCA
    - `rpca_max_iter`: Maximum iterations for robust PCA
    - `rpca_tol`: Convergence tolerance for robust PCA
  - Features:
    - Running statistics for online updates
    - Support for robust PCA with ADMM optimization
    - Automatic component computation

- **Autoencoder**: Neural network-based compression
  - Configuration: `configs/features/compressors/autoencoder.yaml`
  - Parameters:
    - `n_components`: Number of dimensions to compress to
    - `feature_dim`: Input feature dimension
    - `hidden_dims`: List of hidden layer dimensions
    - `activation`: Activation function (`relu`, `leaky_relu`, `elu`, `gelu`)
    - `batch_norm`: Whether to use batch normalization
    - `dropout`: Dropout rate
  - Features:
    - Configurable encoder-decoder architecture
    - Flexible hidden layer dimensions
    - Reconstruction loss optimization
    - Support for various activation functions
    - Optional batch normalization and dropout

## Usage

### Configuration

The main configuration file `configs/features/radio_features.yaml` contains all settings for the feature pipeline. You can:

1. Set the dataset path in the configuration file:
```yaml
dataset:
  path: /path/to/your/dataset
```

2. Override settings via command line:
```bash
python -m threedgrut.features.generate \
    --config-name apps/features/generate_colmap_radio_pca \
    path=/path/to/your/dataset \
    +dataset.save_features_vizualization=True
```

### 1. Extract Raw Features

To extract features from a dataset using the RADIO model without compression:

```bash
python -m threedgrut.features.generate \
    --config-name apps/features/colmap_radio \
    path=/path/to/your/dataset \
    +compressor.type=null
```

### 2. Extract and Compress Features

To extract features and compress them in one step:

```bash
# For PCA compression
python -m threedgrut.features.generate \
    --config-name apps/features/colmap_radio \
    path=/path/to/your/dataset \
    +compressor.type=pca \
    +compressor.n_components=128 \
    +compressor.feature_dim=512 \
    +compressor.use_robust_pca=false

# For Autoencoder compression
python -m threedgrut.features.generate \
    --config-name apps/features/colmap_radio \
    path=/path/to/your/dataset \
    +compressor.type=autoencoder \
    +compressor.n_components=64 \
    +compressor.feature_dim=512 \
    +compressor.hidden_dims=[256,128] \
    +compressor.activation=relu \
    +compressor.batch_norm=true \
    +compressor.dropout=0.1
```

## Feature Storage Structure

Features are stored in a structured directory hierarchy:

```
dataset_path/
└── features/
    └── {extractor_type}/
        └── {compressor_type}_{n}/  # Compressed features
            └── {index:06d}.pt      # Individual feature files
```

Example:
```
dataset_path/
└── features/
    └── radio/
        ├── pca_128/               # PCA compressed to 128 dimensions
        └── ae_64/                 # Autoencoder compressed to 64 dimensions
            └── 000000.pt
```

Note: Raw features are not saved to disk. The feature extractor processes images on-the-fly and passes the features directly to the compressor. Only the compressed features are saved to disk.

## Configuration Files

### Main Configuration (`configs/features/radio_features.yaml`)
```yaml
# Dataset configuration
dataset:
  type: nerf  # or colmap
  path: /path/to/your/dataset  # Set this to your dataset path
  load_features: false  # Set to true when using pre-computed features
  feature_type: radio  # Type of feature extractor used
  feature_compressor: null  # Set to e.g. 'pca_128' when using compressed features

# Feature extractor configuration
extractor:
  type: nv_radio
  model_name: radio_v2.5-h  # Options: radio_v2.5-h, radio_v2.5-m, radio_v2.5-l
  upscale: 1
  use_adaptors: false
  adaptor_names: []
  device: cuda

# Feature compressor configuration (optional)
compressor:
  type: null  # Set to 'pca' or 'autoencoder' when using compression
  n_components: 128  # Number of dimensions to compress to
  feature_dim: 512  # Input feature dimension
  # PCA specific settings
  use_robust_pca: false
  rpca_lambda: 1.0
  rpca_max_iter: 100
  rpca_tol: 1e-7
  # Autoencoder specific settings
  hidden_dims: [256, 128]
  activation: relu
  batch_norm: true
  dropout: 0.1

# Training settings (for autoencoder)
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
```

### Feature Extractor Config (`configs/features/extractors/nv_radio.yaml`)
```yaml
type: nv_radio
model_name: radio_v2.5-h  # Options: radio_v2.5-h, radio_v2.5-m, radio_v2.5-l
upscale: 1
use_adaptors: false
adaptor_names: []
device: cuda
```

### PCA Compressor Config (`configs/features/compressors/pca.yaml`)
```yaml
type: pca
n_components: 128
feature_dim: 512
use_robust_pca: false
rpca_lambda: 1.0
rpca_max_iter: 100
rpca_tol: 1e-7
```

### Autoencoder Compressor Config (`configs/features/compressors/autoencoder.yaml`)
```yaml
type: autoencoder
n_components: 64
feature_dim: 512
hidden_dims: [256, 128]
activation: relu
batch_norm: true
dropout: 0.1
```

## Dataset Integration

The feature loading functionality is integrated into the dataset classes through the `FeatureDataset` mixin. To use pre-computed features in your dataset:

1. Set `load_features: true` in your dataset configuration
2. Specify the feature type and compressor (if any):
```yaml
dataset:
  load_features: true
  feature_type: radio
  feature_compressor: pca_128  # Optional
```

The features will be automatically loaded and included in the dataset items. 