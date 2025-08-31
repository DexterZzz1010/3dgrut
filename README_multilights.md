# Multi-Lights COLMAP Dataset

This directory contains tools for creating and using multi-lights datasets from COLMAP datasets and IBL (Image-Based Lighting) environment maps. The implementation leverages the [cosmos1-diffusion-renderer](https://github.com/nv-tlabs/diffusion-renderer) for neural inverse and forward rendering.

## Overview

The multi-lights dataset workflow consists of:

1. **G-buffer Extraction**: Extract geometry and material properties (basecolor, normal, depth, roughness, metallic) from COLMAP images using the cosmos1-diffusion-renderer inverse renderer.

2. **Relighting**: Apply different IBL environment maps to the extracted G-buffers using the forward renderer to generate relit images. Uses efficient batch processing to apply all lighting conditions in a single model load.

3. **Dataset Organization**: Structure the output as an extended COLMAP dataset with additional relit images and IBL data.

4. **Dataset Loading**: Provide a PyTorch dataset class that inherits from COLMAP and returns IBL tensors alongside standard data.

## Files

- `generate_multilights_dataset.py`: Main script to generate multi-lights dataset
- `threedgrut/datasets/dataset_colmap_multilights.py`: PyTorch dataset class
- `example_multilights_usage.py`: Example usage and testing script
- `README_multilights.md`: This documentation file

## Requirements

### System Requirements
- NVIDIA GPU with at least 16GB VRAM (24GB recommended)
- CUDA 12.0 or higher
- Python 3.10
- At least 70GB free disk space

### Dependencies
- cosmos1-diffusion-renderer repository and its dependencies
- PyTorch
- OpenCV with OpenEXR support
- PIL/Pillow
- NumPy
- Matplotlib (for visualization)

### Installation

1. **Clone and install cosmos1-diffusion-renderer**:
   ```bash
   cd /mnt/dev
   git clone https://github.com/nv-tlabs/diffusion-renderer cosmos1-diffusion-renderer
   cd cosmos1-diffusion-renderer
   
   # Create the cosmos-predict1 conda environment (REQUIRED NAME)
   conda env create --file cosmos-predict1.yaml
   conda activate cosmos-predict1
   pip install -r requirements.txt
   # ... (follow their complete installation guide)
   ```

2. **Download model weights** (must be in cosmos-predict1 environment):
   ```bash
   # In cosmos1-diffusion-renderer directory with cosmos-predict1 environment active
   conda activate cosmos-predict1
   cd /mnt/dev/cosmos1-diffusion-renderer
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_renderer_checkpoints.py --checkpoint_dir checkpoints
   ```

3. **Install additional dependencies**:
   ```bash
   # In cosmos-predict1 environment
   conda activate cosmos-predict1
   pip install opencv-python matplotlib
   ```

**Important**: The cosmos1-diffusion-renderer **requires** the conda environment to be named `cosmos-predict1`. You must activate this environment before running any of the scripts.

## Usage

### 1. Generate Multi-Lights Dataset

**First, activate the required conda environment:**
```bash
conda activate cosmos-predict1
```

**Then run the generation script:**

**Basic usage (1 version per image):**
```bash
python scripts/generate_multilights_dataset.py \
    --colmap_path /path/to/your/colmap/dataset \
    --ibl_folder /path/to/ibl/files \
    --output_path /path/to/output/multilights/dataset
```

**Advanced usage (multiple random versions):**
```bash
python scripts/generate_multilights_dataset.py \
    --colmap_path /path/to/your/colmap/dataset \
    --ibl_folder /path/to/hdr/files \
    --output_path /path/to/output/multilights/dataset \
    --num_versions 5 \
    --rotation_range 360 \
    --seed 42
```

**Custom resolution override (if needed):**
```bash
python scripts/generate_multilights_dataset.py \
    --colmap_path /path/to/your/colmap/dataset \
    --ibl_folder /path/to/hdr/files \
    --output_path /path/to/output/multilights/dataset \
    --height 512 \
    --width 1024
```

**Downsampled processing (faster, less memory):**
```bash
# Process at half resolution (2x faster, 4x less memory)
python scripts/generate_multilights_dataset.py \
    --colmap_path /path/to/your/colmap/dataset \
    --ibl_folder /path/to/hdr/files \
    --output_path /path/to/output/multilights/dataset \
    --downsample_factor 2.0

# Process at quarter resolution (4x faster, 16x less memory) 
python scripts/generate_multilights_dataset.py \
    --colmap_path /path/to/your/colmap/dataset \
    --ibl_folder /path/to/hdr/files \
    --output_path /path/to/output/multilights/dataset \
    --downsample_factor 4.0
```

**Append new versions to existing dataset (reuses G-buffers):**
```bash
# First, create initial dataset
python scripts/generate_multilights_dataset.py \
    --colmap_path /path/to/colmap/dataset \
    --ibl_folder /path/to/hdr/files \
    --output_path /path/to/dataset \
    --num_versions 3

# Later, add more versions without re-extracting G-buffers
python scripts/generate_multilights_dataset.py \
    --ibl_folder /path/to/new/hdr/files \
    --output_path /path/to/dataset \
    --num_versions 5 \
    --append

# Add versions with different IBL collection
python scripts/generate_multilights_dataset.py \
    --ibl_folder /path/to/different/hdr/collection \
    --output_path /path/to/dataset \
    --num_versions 2 \
    --rotation_range 180 \
    --seed 123 \
    --append
```

**Input Requirements:**
- **COLMAP dataset** with standard structure:
  ```
  colmap_dataset/
  ├── images/          # Input images
  ├── sparse/
  │   └── 0/
  │       ├── cameras.bin
  │       ├── images.bin
  │       └── points3D.bin
  └── ...
  ```

- **IBL folder** containing HDR/EXR environment maps:
  ```
  ibl_folder/
  ├── sunny_day.hdr
  ├── cloudy_sky.exr
  ├── indoor_lighting.hdr
  ├── outdoor_scene.exr
  └── ...
  ```

**Output Structure (Basic):**
```
output_dataset/
├── images/              # Original COLMAP images
├── relit_images/        # Relit images organized by IBL
│   ├── sunny_day/       # Images relit with sunny_day.hdr
│   ├── cloudy_sky/      # Images relit with cloudy_sky.hdr
│   └── ...
├── ibls/                # Copy of IBL environment maps
├── sparse/              # COLMAP sparse reconstruction (copied)
└── dataset_info.json    # Metadata about the dataset
```

**Output Structure (Multi-Version with Random Sampling):**
```
output_dataset/
├── images/              # Original COLMAP images
├── relit_images/        # Relit images organized by version
│   ├── v001/           # Version 1: Random IBL + random rotation
│   ├── v002/           # Version 2: Random IBL + random rotation
│   ├── v003/           # Version 3: Random IBL + random rotation
│   └── ...
├── ibls/               # Rotated IBL files used for each version
│   ├── v001_sunny_rot45.3.hdr
│   ├── v002_cloudy_rot-67.8.hdr
│   └── ...
├── original_ibls/      # Original IBL files (unrotated)
├── sparse/             # COLMAP sparse reconstruction (copied)
└── dataset_info.json   # Enhanced metadata with version info
```

### Multi-Version Generation Parameters

The script supports several parameters for generating multiple lighting versions:

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--num_versions` | Number of lighting versions to generate | 1 | `--num_versions 5` |
| `--rotation_range` | Maximum rotation angle in degrees (±range/2) | 360.0 | `--rotation_range 180` |
| `--seed` | Random seed for reproducible results | None | `--seed 42` |
| `--height` | Output image height (optional) | Auto-detected | `--height 512` |
| `--width` | Output image width (optional) | Auto-detected | `--width 1024` |
| `--downsample_factor` | Factor to downsample images | 1.0 | `--downsample_factor 2.0` |
| `--append` | Append new versions to existing dataset | False | `--append` |

**Multi-Version Features:**
- **Auto-Resolution Detection**: Automatically detects and uses input image resolution
- **Random IBL Sampling**: Each version randomly selects from available HDR/EXR files
- **Random Rotation**: Each version applies random horizontal rotation to the IBL
- **Efficient Processing**: G-buffers extracted once and reused for all versions
- **Rich Training Data**: Creates diverse lighting conditions from limited IBL files
- **Incremental Updates**: Add new lighting versions without re-extracting G-buffers
- **G-buffer Preservation**: Saves G-buffers for future reuse and iteration
- **Progress Tracking**: Visual progress bars for all major processing steps

**Resolution Handling:**
- **Automatic Detection**: Script automatically detects resolution from COLMAP images
- **Downsampling Support**: Use `--downsample_factor` to reduce resolution and processing time
- **Override Option**: Use `--height` and `--width` to override auto-detected resolution
- **Mixed Resolution Warning**: Handles datasets with multiple image sizes gracefully
- **Efficient Processing**: Downsample for faster iteration, full resolution for final results
- **No Manual Specification Needed**: Just point to your dataset and go!

**Downsampling Benefits:**
- **Faster Processing**: 2x downsample = ~4x speed improvement
- **Less Memory Usage**: Quadratic reduction in GPU memory requirements
- **Quick Iteration**: Test lighting setups faster with lower resolution
- **Maintains Quality**: Algorithm behavior preserved at different scales
- **COLMAP Compatible**: Follows standard naming scheme (`images_2`, `images_4`, etc.)
- **Automatic Implementation**: Images are actually downsampled during generation

**Append Mode Benefits:**
- **Reuses Existing G-buffers**: Skip expensive G-buffer extraction (saves hours)
- **Iterative Development**: Experiment with new IBLs without full regeneration
- **Preserves Existing Data**: All previous versions remain intact
- **Flexible IBL Sources**: Use different IBL collections for new versions  
- **Version Continuity**: New versions get sequential numbering (v004, v005, etc.)
- **Quick Turnaround**: Add versions in minutes instead of hours

**Progress Tracking:**
The script now includes visual progress bars for all major processing steps:

```
🔍 Extracting G-buffers: 100%|██████████| 45/45 [05:32<00:00, 7.35images/s]
🎭 Generating IBL versions: 100%|██████████| 5/5 [00:03<00:00, 1.45versions/s] v004: studio_lighting.hdr (127.3°)
💡 Relighting images: 100%|██████████| 5/5 [08:45<00:00, 1.75minutes/version] v004: studio_lighting.hdr (127.3°)
📁 Organizing dataset: 100%|██████████| 5/5 [00:12<00:00, 2.41versions/s] v004: copying files
```

**Progress Features:**
- **Real-time Updates**: Live progress indicators with completion percentages
- **Step Context**: Icons and descriptions for each processing phase  
- **Version Details**: Current IBL name, rotation angle, and version number
- **Performance Metrics**: Processing rates and estimated time remaining
- **File Operations**: Progress through collections and copy operations

**Example Multi-Version Commands:**
```bash
# Generate 3 versions with moderate rotation
python scripts/generate_multilights_dataset.py \
    --colmap_path /path/to/colmap \
    --ibl_folder /path/to/hdr \
    --output_path /path/to/output \
    --num_versions 3 \
    --rotation_range 90 \
    --seed 42

# Generate 10 versions with full rotation range
python scripts/generate_multilights_dataset.py \
    --colmap_path /path/to/colmap \
    --ibl_folder /path/to/hdr \
    --output_path /path/to/output \
    --num_versions 10

# Conservative: Generate 5 versions with limited rotation
python scripts/generate_multilights_dataset.py \
    --colmap_path /path/to/colmap \
    --ibl_folder /path/to/hdr \
    --output_path /path/to/output \
    --num_versions 5 \
    --rotation_range 45
```

### 2. Use Multi-Lights Dataset in Code

```python
from threedgrut.datasets.dataset_colmap_multilights import ColmapMultiLightsDataset

# Create dummy config (or use your actual config)
class Config:
    def __init__(self, path):
        self.path = path
        self.dataset = DatasetConfig()

class DatasetConfig:
    def __init__(self):
        self.calibration_dir = "sparse/0"
        self.downsample_factor = 1.0
        self.test_split_interval = 8
        self.save_downsampled_images = True
        self.downsample_method = "lanczos"

config = Config("/path/to/multilights/dataset")

# Initialize dataset
dataset = ColmapMultiLightsDataset(
    config=config,
    device="cuda",
    split="train",
    ibl_resolution=(256, 512),    # IBL resolution (H, W)
    use_original_images=True,     # Use original or relit images
    selected_ibl_names=None       # Use all IBLs or specific ones
)

# Access data
sample = dataset[0]
print(f"Image shape: {sample['data'].shape}")           # (1, H, W, 3)
print(f"Pose shape: {sample['pose'].shape}")            # (1, 4, 4)
print(f"IBL tensor shape: {sample['ibl_tensor'].shape}") # (3, IBL_H, IBL_W)
print(f"IBL name: {sample['ibl_name']}")                # String
print(f"Available IBLs: {sample['available_ibls']}")     # List of IBL names

# Access specific IBL
ibl_tensor = dataset.get_ibl_tensor("sunny_day")
ibl_names = dataset.get_available_ibl_names()

# Get random IBL (useful for training)
random_ibl, ibl_name = dataset.get_random_ibl_tensor()
```

### 3. Example and Testing

```bash
# Test dataset loading (no special environment needed)
python example_multilights_usage.py test --dataset_path /path/to/multilights/dataset

# Visualize dataset samples (no special environment needed)
python example_multilights_usage.py visualize --dataset_path /path/to/multilights/dataset --num_samples 3

# Generate dataset (requires cosmos-predict1 environment)
# This will create 3 versions with random IBL sampling and rotation
conda activate cosmos-predict1
python example_multilights_usage.py generate \
    --colmap_path /path/to/colmap \
    --ibl_folder /path/to/ibls \
    --output_path /path/to/output
```

## Dataset Class Features

### ColmapMultiLightsDataset

Inherits from `ColmapDataset` and adds:

**Additional Parameters:**
- `ibl_resolution`: Resolution for IBL processing (default: (256, 512))
- `use_original_images`: Use original vs relit images (default: True)
- `selected_ibl_names`: List of specific IBLs to use (default: None = all)
- `ibl_format`: IBL tensor format (default: "latlong")

**Additional Methods:**
- `get_available_ibl_names()`: Get list of available IBL names
- `get_ibl_tensor(ibl_name)`: Get specific IBL tensor
- `get_random_ibl_tensor()`: Get random IBL tensor and name
- `get_relit_image_path(idx, ibl_name)`: Get path to specific relit image
- `get_dataset_info()`: Get dataset metadata

**Additional Data in `__getitem__`:**
- `ibl_tensor`: IBL environment map tensor (3, H, W)
- `ibl_name`: Name of the current IBL
- `ibl_index`: Index of the current IBL
- `available_ibls`: List of all available IBL names

## Processing Pipeline Details

### G-buffer Extraction
The inverse renderer extracts:
- **basecolor**: Albedo/diffuse color
- **normal**: Surface normals
- **depth**: Depth maps
- **roughness**: Surface roughness
- **metallic**: Metallic properties

### Relighting Process
The forward renderer:
1. Takes G-buffers as input
2. Applies IBL environment lighting
3. Generates photorealistic relit images
4. Preserves original geometry and materials

### IBL Processing
- Supports HDR (.hdr) and EXR (.exr) formats
- Automatic resolution resizing
- HDR value clamping for stability
- Latlong format (equirectangular projection)

## Memory and Performance

### Memory Usage
- Peak GPU memory: ~24GB (can be reduced with `--offload_*` flags)
- Disk space: ~5-10x original COLMAP dataset size
- IBL tensors: ~3MB per IBL at 256x512 resolution

### Performance Tips
1. **Reduce memory usage**:
   ```bash
   # Add these flags to generation script
   --offload_diffusion_transformer --offload_tokenizer
   ```

2. **Optimized batch processing**: All IBL versions are processed in a single forward renderer call, eliminating model reloading overhead and significantly improving performance

3. **Resolution compatibility**: Use `--force_resolution` to ensure 704×1280 resolution expected by cosmos1-diffusion-renderer

4. **IBL selection**: Use `selected_ibl_names` to limit IBLs during training

## Resolution Requirements ⚠️

**IMPORTANT**: cosmos1-diffusion-renderer models were trained specifically on **704 × 1280** resolution. Using different resolutions may cause tensor dimension errors during processing.

### Solutions for Resolution Mismatch:

1. **Force expected resolution** (recommended for compatibility):
   ```bash
   python generate_multilights_dataset.py \
       --colmap_path /path/to/colmap/dataset \
       --ibl_folder /path/to/hdr/files \
       --output_path /path/to/output/dataset \
       --force_resolution
   ```

2. **Use downsample factor** to get closer to expected resolution:
   ```bash
   # For 1557×1038 images, use factor ~1.47 to get close to 1280×704
   python generate_multilights_dataset.py \
       --colmap_path /path/to/colmap/dataset \
       --ibl_folder /path/to/hdr/files \
       --output_path /path/to/output/dataset \
       --downsample_factor 1.47
   ```

3. **Manually resize** your COLMAP images to 704×1280 beforehand

The script will automatically detect resolution mismatches and provide specific suggestions.

## Troubleshooting

### Common Issues

1. **Tensor dimension errors (RuntimeError: Sizes of tensors must match)**:
   - **Cause**: Input images have resolution different from expected 704×1280
   - **Solution**: Use `--force_resolution` flag to force correct resolution
   - **Alternative**: Use appropriate `--downsample_factor` to get close to 704×1280
   - Example error: `Expected size 129 but got size 130 for tensor number 1 in the list`

2. **CUDA out of memory**:
   - Reduce image resolution with `--downsample_factor`
   - Add offload flags: `--offload_diffusion_transformer --offload_tokenizer`
   - Process fewer IBLs at once

3. **G-buffer extraction fails**:
   - Make sure you activated `cosmos-predict1` environment: `conda activate cosmos-predict1`
   - Ensure `cosmos-predict1` conda environment exists and is properly set up
   - Check cosmos1-diffusion-renderer installation
   - Verify model checkpoints are downloaded
   - Ensure CUDA_HOME is set correctly

3. **IBL loading issues**:
   - Install OpenEXR support: `pip install opencv-contrib-python`
   - Check IBL file formats (.hdr, .exr supported)
   - Verify file permissions

4. **Dataset loading errors**:
   - Check `dataset_info.json` exists
   - Verify directory structure matches expected format
   - Ensure all referenced files exist

### Environment Variables
**Before running the generation script:**
```bash
# Activate the required conda environment
conda activate cosmos-predict1

# The script will automatically set these, but you can set them manually if needed:
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=/path/to/cosmos1-diffusion-renderer:$PYTHONPATH
```

## Advanced Usage

### Custom IBL Processing
```python
# Use specific IBLs
dataset = ColmapMultiLightsDataset(
    config=config,
    selected_ibl_names=["sunny_day", "cloudy_sky"]
)

# Use relit images instead of originals
dataset = ColmapMultiLightsDataset(
    config=config,
    use_original_images=False  # Uses relit images from first IBL
)

# Custom IBL resolution
dataset = ColmapMultiLightsDataset(
    config=config,
    ibl_resolution=(128, 256)  # Smaller for faster loading
)
```

### Training Integration
```python
# Training loop example
for epoch in range(num_epochs):
    for batch_idx, sample in enumerate(dataloader):
        images = sample['data']          # Original/relit images
        poses = sample['pose']           # Camera poses
        ibl_tensors = sample['ibl_tensor']  # Environment lighting
        
        # Your training code here
        loss = model(images, poses, ibl_tensors)
        loss.backward()
        optimizer.step()
```

## Citation

If you use this code, please cite the original DiffusionRenderer paper:

```bibtex
@inproceedings{DiffusionRenderer,
    author = {Ruofan Liang and Zan Gojcic and Huan Ling and Jacob Munkberg and 
        Jon Hasselgren and Zhi-Hao Lin and Jun Gao and Alexander Keller and 
        Nandita Vijaykumar and Sanja Fidler and Zian Wang},
    title = {DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2025}
}
```
