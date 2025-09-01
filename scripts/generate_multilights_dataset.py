#!/usr/bin/env python3
"""
Script to generate a multi-lights dataset from a COLMAP dataset and IBL files.

This script takes a COLMAP dataset and a folder of IBL files, and generates
multiple relighted versions where each version uses a randomly sampled and 
rotated IBL environment map.

Features:
- Random IBL sampling from available HDR/EXR files
- Random IBL rotation for data augmentation
- G-buffer extraction only done once (shared across versions)
- Organized output with version directories and metadata
- Visual progress bars for all major processing steps
- Append mode to add versions without re-extracting G-buffers

Usage:
    # Basic: Auto-detect resolution and generate single version
    conda activate cosmos-predict1
    python generate_multilights_dataset.py \
        --colmap_path /path/to/colmap/dataset \
        --ibl_folder /path/to/hdr/files \
        --output_path /path/to/output/dataset

    # Advanced: Multiple versions with random sampling
    python generate_multilights_dataset.py \
        --colmap_path /path/to/colmap/dataset \
        --ibl_folder /path/to/hdr/files \
        --output_path /path/to/output/dataset \
        --num_versions 5 \
        --rotation_range 360 \
        --seed 42

    # Fast iteration: Downsampled processing
    python generate_multilights_dataset.py \
        --colmap_path /path/to/colmap/dataset \
        --ibl_folder /path/to/hdr/files \
        --output_path /path/to/output/dataset \
        --downsample_factor 2.0 \
        --num_versions 3

    # Append new versions (reuses G-buffers)
    python generate_multilights_dataset.py \
        --ibl_folder /path/to/new/hdr/files \
        --output_path /path/to/existing/dataset \
        --num_versions 2 \
        --append
    
    # Resume from failed run (reuses existing G-buffers)
    python generate_multilights_dataset.py \
        --colmap_path /path/to/colmap/dataset \
        --ibl_folder /path/to/hdr/files \
        --output_path /path/to/failed/dataset \
        --num_versions 3 \
        --resume
    
    # Force individual relighting (if batch processing fails)
    python generate_multilights_dataset.py \
        --colmap_path /path/to/colmap/dataset \
        --ibl_folder /path/to/hdr/files \
        --output_path /path/to/output/dataset \
        --num_versions 5 \
        --force_individual_relighting
"""

import argparse
import os
import json
import shutil
import subprocess
import sys
import random
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from collections import Counter
from tqdm import tqdm


def downsample_image(input_path, output_path, downsample_factor):
    """
    Downsample an image by the given factor and save it.
    
    Args:
        input_path: Path to input image
        output_path: Path to save downsampled image
        downsample_factor: Factor to downsample by (e.g., 2.0 = half size)
    """
    if downsample_factor == 1.0:
        # No downsampling needed, just copy
        shutil.copy2(input_path, output_path)
        return
    
    # Load image
    img = Image.open(input_path)
    original_width, original_height = img.size
    
    # Calculate new dimensions
    new_width = int(original_width / downsample_factor)
    new_height = int(original_height / downsample_factor)
    
    # Downsample using high-quality resampling
    downsampled_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Save with same format and quality
    if input_path.suffix.lower() in ['.jpg', '.jpeg']:
        downsampled_img.save(output_path, 'JPEG', quality=95)
    else:
        downsampled_img.save(output_path)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multi-lights dataset from COLMAP dataset and IBL files"
    )
    
    parser.add_argument(
        "--colmap_path",
        type=str,
        required=True,
        help="Path to the COLMAP dataset directory"
    )
    parser.add_argument(
        "--ibl_folder", 
        type=str,
        required=True,
        help="Path to folder containing IBL (.hdr) files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output directory for the multi-lights dataset"
    )
    parser.add_argument(
        "--cosmos_path",
        type=str,
        default="/mnt/dev/cosmos1-diffusion-renderer",
        help="Path to cosmos1-diffusion-renderer repository"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing model checkpoints (relative to cosmos_path)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output image height (auto-detected from input images if not specified)"
    )
    parser.add_argument(
        "--width", 
        type=int,
        default=None,
        help="Output image width (auto-detected from input images if not specified)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=1,
        help="Number of frames to process per image (1 for images)"
    )
    parser.add_argument(
        "--num_versions",
        type=int,
        default=1,
        help="Number of multi-light versions to generate (each with random HDR and rotation)"
    )
    parser.add_argument(
        "--rotation_range",
        type=float,
        default=360.0,
        help="Maximum rotation angle in degrees for random rotations (default: 360.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results"
    )
    parser.add_argument(
        "--downsample_factor",
        type=float,
        default=1.0,
        help="Downsample factor for input images (e.g., 2.0 = half resolution, 0.5 = double resolution)"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new versions to existing dataset (reuse existing G-buffers)"
    )
    
    return parser.parse_args()


def setup_cosmos_environment(cosmos_path):
    """Set up environment for cosmos1-diffusion-renderer."""
    cosmos_path = Path(cosmos_path).resolve()
    
    # Add cosmos path to Python path
    if str(cosmos_path) not in sys.path:
        sys.path.insert(0, str(cosmos_path))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(cosmos_path) + ":" + os.environ.get("PYTHONPATH", "")
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    
    return cosmos_path


def detect_image_resolution(colmap_path, downsample_factor=1.0):
    """
    Auto-detect image resolution from COLMAP dataset and apply downsampling.
    
    Args:
        colmap_path: Path to COLMAP dataset
        downsample_factor: Factor to downsample images (e.g., 2.0 = half resolution)
        
    Returns:
        Tuple of (height, width) after applying downsample factor
    """
    colmap_path = Path(colmap_path)
    images_dir = colmap_path / "images"
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Find image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    # Check first few images to detect resolution
    resolutions = []
    for img_file in image_files[:5]:  # Check first 5 images
        try:
            with Image.open(img_file) as img:
                width, height = img.size
                resolutions.append((height, width))
        except Exception as e:
            print(f"Warning: Could not read image {img_file}: {e}")
            continue
    
    if not resolutions:
        raise ValueError("Could not detect resolution from any images")
    
    # Check if all images have the same resolution
    unique_resolutions = list(set(resolutions))
    if len(unique_resolutions) > 1:
        print(f"Warning: Multiple resolutions detected: {unique_resolutions}")
        print(f"Using most common resolution...")
        # Use most common resolution
        most_common = Counter(resolutions).most_common(1)[0][0]
        original_height, original_width = most_common
    else:
        original_height, original_width = resolutions[0]
    
    # Apply downsampling
    target_height = int(original_height / downsample_factor)
    target_width = int(original_width / downsample_factor)
    
    print(f"Auto-detected original resolution: {original_width}x{original_height}")
    if downsample_factor != 1.0:
        print(f"After downsample factor {downsample_factor}: {target_width}x{target_height}")
    
    # Check against cosmos1-diffusion-renderer's expected resolution
    EXPECTED_HEIGHT, EXPECTED_WIDTH = 704, 1280
    if target_height != EXPECTED_HEIGHT or target_width != EXPECTED_WIDTH:
        print(f"⚠️  Resolution mismatch: detected {target_width}×{target_height}, expected 704×1280")
        
        # Calculate suggested downsample factor
        height_ratio = original_height / EXPECTED_HEIGHT
        width_ratio = original_width / EXPECTED_WIDTH
        suggested_factor = max(height_ratio, width_ratio)
        
        print(f"💡 Consider using: --force_resolution or --downsample_factor {suggested_factor:.2f}")
    
    return target_height, target_width


def extract_gbuffers(images_dir, cosmos_path, checkpoint_dir, output_dir, height, width, num_frames):
    """
    Extract G-buffers from images using cosmos1-diffusion-renderer inverse renderer.
    
    Args:
        images_dir: Directory containing input images  
        cosmos_path: Path to cosmos1-diffusion-renderer
        checkpoint_dir: Directory containing model checkpoints
        output_dir: Directory to save G-buffer outputs
        height, width: Output dimensions
        num_frames: Number of frames per image
        
    Returns:
        Path to the directory containing extracted G-buffers
    """
    images_dir = Path(images_dir)
    cosmos_path = Path(cosmos_path)
    output_dir = Path(output_dir)
    
    print(f"🔍 Input images directory: {images_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Input images directory not found: {images_dir}")
    
    # Count input images for progress tracking
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    input_images = []
    for ext in image_extensions:
        input_images.extend(images_dir.glob(f"*{ext}"))
    
    print(f"🔍 Found {len(input_images)} input images:")
    for img in input_images[:5]:  # Show first 5
        print(f"   📷 {img.name}")
    if len(input_images) > 5:
        print(f"   ... and {len(input_images) - 5} more")
    
    if not input_images:
        raise FileNotFoundError(f"No image files found in input directory: {images_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare inverse renderer command
    inverse_script = cosmos_path / "cosmos_predict1/diffusion/inference/inference_inverse_renderer.py"
    
    cmd = [
        "python", str(inverse_script),
        "--checkpoint_dir", str(cosmos_path / checkpoint_dir),
        "--diffusion_transformer_dir", "Diffusion_Renderer_Inverse_Cosmos_7B",
        "--dataset_path", str(images_dir),
        "--num_video_frames", str(num_frames),
        "--group_mode", "webdataset", 
        "--video_save_folder", str(output_dir),
        "--save_video", "False",
        "--save_image", "True",
        "--height", str(height),
        "--width", str(width)
    ]
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["CUDA_HOME"] = env.get("CONDA_PREFIX", "/usr/local/cuda")
    env["PYTHONPATH"] = str(cosmos_path)
    env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    
    print(f"Running G-buffer extraction command:")
    print(f"CUDA_HOME=$CONDA_PREFIX PYTHONPATH={cosmos_path} {' '.join(cmd)}")
    
    # Create progress bar for G-buffer extraction
    with tqdm(total=len(input_images), desc="🔍 Extracting G-buffers", unit="images") as pbar:
        # Run the inverse renderer
        result = subprocess.run(cmd, cwd=cosmos_path, env=env, capture_output=True, text=True)
        
        # Update progress bar (since we can't track internal progress, we'll complete it at the end)
        pbar.update(len(input_images))
    
    if result.returncode != 0:
        print(f"❌ Error running inverse renderer:")
        print(f"Command: {' '.join(cmd)}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"💡 Troubleshooting tips:")
        print(f"   - Make sure you activated the 'cosmos-predict1' conda environment")
        print(f"   - Check that the checkpoint directory exists: {cosmos_path / checkpoint_dir}")
        print(f"   - Verify that the images directory contains valid image files: {images_dir}")
        raise RuntimeError("G-buffer extraction failed")
    
    print("✅ G-buffer extraction process completed successfully")
    
    # Validate G-buffer outputs
    expected_gbuffer_dir = output_dir / "gbuffer_frames"
    if not expected_gbuffer_dir.exists():
        raise RuntimeError(f"G-buffer directory not found: {expected_gbuffer_dir}")
    
    # Count valid G-buffer files
    gbuffer_files = list(expected_gbuffer_dir.glob("*"))
    valid_files = []
    
    for item in gbuffer_files:
        if item.is_file():
            file_size = item.stat().st_size
            if (item.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.exr'] or 
                (not item.suffix and file_size > 1000)):
                valid_files.append(item)
        elif item.is_dir():
            # Look inside subdirectories for G-buffer files
            subfiles = list(item.glob("*"))
            for subfile in subfiles:
                if subfile.is_file():
                    file_size = subfile.stat().st_size
                    if (subfile.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.exr'] or 
                        (not subfile.suffix and file_size > 1000)):
                        valid_files.append(subfile)
    
    if not valid_files:
        raise RuntimeError("No valid G-buffer files found after extraction")
    
    print(f"✅ Found {len(valid_files)} G-buffer files")
    return expected_gbuffer_dir


def check_existing_dataset(output_path):
    """
    Check if a valid multi-lights dataset exists at the given path.
    
    Args:
        output_path: Path to check for existing dataset
        
    Returns:
        tuple: (is_valid, dataset_info, gbuffer_dir)
            - is_valid: True if valid dataset found with reusable G-buffers
            - dataset_info: Dataset metadata dict or None  
            - gbuffer_dir: Path to G-buffers directory or None
    """
    output_path = Path(output_path)
    
    print(f"🔍 Checking for existing dataset at: {output_path}")
    
    # Check if directory exists
    if not output_path.exists():
        print(f"❌ Directory does not exist: {output_path}")
        return False, None, None
    
    if not output_path.is_dir():
        print(f"❌ Path is not a directory: {output_path}")
        return False, None, None
    
    # Check for dataset_info.json
    dataset_info_path = output_path / "dataset_info.json"
    if not dataset_info_path.exists():
        print(f"❌ No dataset_info.json found at: {dataset_info_path}")
        print(f"📁 Contents of directory:")
        for item in output_path.iterdir():
            print(f"   {'📁' if item.is_dir() else '📄'} {item.name}")
        return False, None, None
    
    # Load and validate dataset info
    try:
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        print(f"✅ Found dataset_info.json")
    except Exception as e:
        print(f"❌ Error reading dataset_info.json: {e}")
        return False, None, None
    
    # Validate dataset type
    if 'type' not in dataset_info:
        print(f"❌ Invalid dataset_info.json: missing 'type' field")
        return False, dataset_info, None
    
    if not dataset_info['type'].startswith('multi_lights_colmap'):
        print(f"❌ Wrong dataset type: {dataset_info['type']} (expected multi_lights_colmap*)")
        return False, dataset_info, None
    
    print(f"✅ Valid multi-lights dataset found (type: {dataset_info['type']})")
    
    # Check for G-buffers directory (required for append mode)
    gbuffer_dir = output_path / "gbuffers"
    if not gbuffer_dir.exists():
        print(f"❌ No gbuffers/ directory found for reuse")
        print(f"💡 Append mode requires existing G-buffers to avoid re-extraction")
        return False, dataset_info, None
    
    # Check if G-buffers directory has files
    gbuffer_files = list(gbuffer_dir.glob("*"))
    if not gbuffer_files:
        print(f"❌ Empty gbuffers/ directory")
        return False, dataset_info, None
    
    print(f"✅ Found {len(gbuffer_files)} items in gbuffers/ directory")
    print(f"✅ Dataset ready for append mode")
    
    return True, dataset_info, gbuffer_dir


def get_next_version_index(dataset_info):
    """
    Get the next version index for a new version in append mode.
    
    Args:
        dataset_info: Dataset metadata dictionary
        
    Returns:
        int: Next version index to use
    """
    if not dataset_info or 'versions' not in dataset_info:
        return 0
    
    existing_versions = dataset_info['versions']
    if not existing_versions:
        return 0
    
    # Find highest existing version index
    max_index = max(v.get('version_index', 0) for v in existing_versions)
    return max_index + 1


def check_existing_gbuffers(output_path):
    """
    Check for existing G-buffers from a previous failed run.
    
    Args:
        output_path: Output dataset path
        
    Returns:
        Path to G-buffers directory if found, None otherwise
    """
    output_path = Path(output_path)
    
    # Check for G-buffers in permanent location
    permanent_gbuffer_dir = output_path / "gbuffers"
    if permanent_gbuffer_dir.exists():
        gbuffer_files = list(permanent_gbuffer_dir.glob("*"))
        if gbuffer_files:
            print(f"✅ Found existing G-buffers in permanent location: {permanent_gbuffer_dir}")
            return permanent_gbuffer_dir
    
    # Check for G-buffers in temporary location
    temp_gbuffer_dir = output_path / "temp_processing" / "gbuffer_output" / "gbuffer_frames"
    if temp_gbuffer_dir.exists():
        gbuffer_files = list(temp_gbuffer_dir.glob("*"))
        if gbuffer_files:
            print(f"✅ Found existing G-buffers in temporary location: {temp_gbuffer_dir}")
            print(f"💡 These will be preserved to permanent location during processing")
            return temp_gbuffer_dir
    
    return None


def preserve_gbuffers(gbuffer_dir, output_path):
    """
    Copy G-buffers to a permanent directory for future reuse.
    
    Args:
        gbuffer_dir: Source G-buffer directory (temporary)
        output_path: Output dataset path
    """
    gbuffer_dir = Path(gbuffer_dir)
    output_path = Path(output_path)
    preserve_dir = output_path / "gbuffers"
    
    if preserve_dir.exists():
        print(f"⚠️  G-buffers directory already exists, skipping preservation")
        return
    
    print(f"💾 Preserving G-buffers for future reuse...")
    shutil.copytree(gbuffer_dir, preserve_dir)
    print(f"✅ G-buffers preserved at: {preserve_dir}")


def prepare_colmap_images(colmap_path, temp_dir):
    """
    Prepare COLMAP images for processing by cosmos1-diffusion-renderer.
    
    Args:
        colmap_path: Path to COLMAP dataset
        temp_dir: Temporary directory for processing
        
    Returns:
        Path to prepared images directory
    """
    colmap_path = Path(colmap_path)
    images_dir = colmap_path / "images"
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Create temp directory for images
    temp_images_dir = Path(temp_dir) / "images"
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images to temp directory with consistent naming
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    print(f"Found {len(image_files)} images in COLMAP dataset")
    
    # Copy images to temp directory
    for img_file in image_files:
        shutil.copy2(img_file, temp_images_dir / img_file.name)
    
    return temp_images_dir


def find_ibl_files(ibl_folder):
    """Find all valid IBL (.hdr/.exr) files in the specified folder."""
    ibl_folder = Path(ibl_folder)
    all_ibl_files = []
    
    for ext in ['.hdr', '.HDR', '.exr', '.EXR']:
        all_ibl_files.extend(ibl_folder.glob(f"*{ext}"))
    
    if not all_ibl_files:
        raise ValueError(f"No IBL files found in {ibl_folder}")
    
    # Validate IBL files and exclude corrupted ones
    valid_ibl_files = []
    corrupted_files = []
    
    print(f"Validating {len(all_ibl_files)} IBL files...")
    for ibl_file in all_ibl_files:
        if validate_hdr_file(ibl_file):
            valid_ibl_files.append(ibl_file)
        else:
            corrupted_files.append(ibl_file)
    
    if corrupted_files:
        print(f"⚠️  Skipping {len(corrupted_files)} corrupted IBL files:")
        for f in corrupted_files[:5]:  # Show first 5
            print(f"   ❌ {f.name}")
        if len(corrupted_files) > 5:
            print(f"   ... and {len(corrupted_files) - 5} more")
    
    if not valid_ibl_files:
        raise ValueError(f"No valid IBL files found in {ibl_folder}")
    
    print(f"✅ Found {len(valid_ibl_files)} valid IBL files: {[f.name for f in valid_ibl_files[:10]]}{'...' if len(valid_ibl_files) > 10 else ''}")
    return sorted(valid_ibl_files)


def validate_hdr_file(hdr_path):
    """
    Validate that an HDR/EXR file is readable.
    
    Args:
        hdr_path: Path to HDR/EXR file
        
    Returns:
        bool: True if file is valid and readable
    """
    try:
        # Enable OpenEXR support
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        
        # Try to load the image
        image = cv2.imread(str(hdr_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return False
        
        # Check if image has reasonable dimensions
        if len(image.shape) < 2 or image.shape[0] < 10 or image.shape[1] < 10:
            return False
            
        return True
    except Exception:
        return False


def rotate_hdr_image(input_path, output_path, rotation_degrees):
    """
    Rotate an HDR/EXR image horizontally (longitude rotation).
    
    Args:
        input_path: Path to input HDR/EXR file
        output_path: Path to save rotated HDR/EXR file  
        rotation_degrees: Rotation angle in degrees
    """
    # Enable OpenEXR support
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    
    # Load HDR/EXR image
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load HDR/EXR image: {input_path}")
    
    # Convert BGR to RGB for HDR/EXR files
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Calculate pixel shift for horizontal rotation
    height, width = image.shape[:2]
    pixel_shift = int(width * (rotation_degrees % 360) / 360)
    
    # Apply horizontal shift (roll operation)
    if pixel_shift != 0:
        image = np.roll(image, pixel_shift, axis=1)
    
    # Convert back to BGR for saving with OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save rotated image
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise ValueError(f"Failed to save rotated HDR/EXR image: {output_path}")
    
    # Validate the saved file
    if not validate_hdr_file(output_path):
        raise ValueError(f"Saved HDR file failed validation: {output_path}")
    
    print(f"  Rotated HDR by {rotation_degrees:.1f}° -> {output_path.name}")


def generate_random_ibl_versions(ibl_files, num_versions, rotation_range, output_dir, seed=None):
    """
    Generate random IBL versions with random sampling and rotations.
    
    Args:
        ibl_files: List of available IBL file paths
        num_versions: Number of versions to generate
        rotation_range: Maximum rotation angle in degrees
        output_dir: Directory to save rotated IBL files
        seed: Random seed for reproducibility
        
    Returns:
        List of tuples (version_index, rotated_ibl_path, original_ibl_name, rotation_degrees)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    versions = []
    
    for version_idx in range(num_versions):
        # Randomly sample an IBL file
        selected_ibl = random.choice(ibl_files)
        
        # Generate random rotation angle
        rotation_degrees = random.uniform(-rotation_range/2, rotation_range/2)
        
        # Create output filename for rotated IBL
        original_name = selected_ibl.stem
        extension = selected_ibl.suffix
        rotated_filename = f"v{version_idx:03d}_{original_name}_rot{rotation_degrees:.1f}{extension}"
        rotated_path = output_dir / rotated_filename
        
        # Rotate and save the IBL
        print(f"Generating version {version_idx}: {selected_ibl.name} (rotation: {rotation_degrees:.1f}°)")
        rotate_hdr_image(selected_ibl, rotated_path, rotation_degrees)
        
        versions.append((version_idx, rotated_path, selected_ibl.name, rotation_degrees))
    
    return versions


def relight_with_custom_ibl(gbuffer_dir, ibl_file, cosmos_path, checkpoint_dir, output_dir, height, width, num_frames, ibl_index):
    """
    Relight G-buffers with a custom IBL file.
    
    Args:
        gbuffer_dir: Directory containing G-buffer frames
        ibl_file: Path to IBL file
        cosmos_path: Path to cosmos1-diffusion-renderer
        checkpoint_dir: Directory containing model checkpoints
        output_dir: Directory to save relit outputs
        height, width: Output dimensions
        num_frames: Number of frames per image
        ibl_index: Index of the IBL for naming
    """
    print(f"Relighting with IBL: {ibl_file.name}")
    
    # Validate G-buffer directory structure
    gbuffer_path = Path(gbuffer_dir)
    if not gbuffer_path.exists():
        raise FileNotFoundError(f"G-buffer directory not found: {gbuffer_path}")
    
    # Check for G-buffer files (should have various passes like basecolor, normal, etc.)
    # Use the same validation logic as batch relighting
    all_files = list(gbuffer_path.glob("*"))
    valid_gbuffer_files = []
    
    for file in all_files:
        if file.is_file():
            file_size = file.stat().st_size
            if (file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.exr'] or 
                (not file.suffix and file_size > 1000)):
                valid_gbuffer_files.append(file)
        elif file.is_dir():
            # Look inside subdirectories for G-buffer files
            subfiles = list(file.glob("*"))
            for subfile in subfiles:
                if subfile.is_file():
                    file_size = subfile.stat().st_size
                    if (subfile.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.exr'] or 
                        (not subfile.suffix and file_size > 1000)):
                        valid_gbuffer_files.append(subfile)
    
    if not valid_gbuffer_files:
        raise FileNotFoundError(f"No valid G-buffer files found in {gbuffer_path}. Expected image files or files without extensions from cosmos1-diffusion-renderer.")
    
    print(f"✅ Found {len(valid_gbuffer_files)} valid G-buffer files for individual relighting")
    
    # First, we need to modify the ENV_LIGHT_PATH_LIST in the forward renderer script
    # We'll create a temporary modified version
    forward_script = cosmos_path / "cosmos_predict1/diffusion/inference/inference_forward_renderer.py"
    temp_forward_script = cosmos_path / f"temp_inference_forward_renderer_{ibl_index}.py"
    
    # Read the original script
    with open(forward_script, 'r') as f:
        script_content = f.read()
    
    # Replace the ENV_LIGHT_PATH_LIST with our custom IBL
    ibl_line = f'ENV_LIGHT_PATH_LIST = ["{str(ibl_file)}"]'
    
    # Find and replace the ENV_LIGHT_PATH_LIST definition
    lines = script_content.split('\n')
    new_lines = []
    in_env_list = False
    
    for line in lines:
        if line.strip().startswith('ENV_LIGHT_PATH_LIST = ['):
            new_lines.append(ibl_line)
            in_env_list = True
        elif in_env_list and line.strip() == ']':
            in_env_list = False
            continue
        elif in_env_list:
            continue
        else:
            new_lines.append(line)
    
    # Write the temporary script
    with open(temp_forward_script, 'w') as f:
        f.write('\n'.join(new_lines))
    
    try:
        # Run the forward renderer with custom IBL (user must be in cosmos-predict1 environment)
        cmd = [
            "python", str(temp_forward_script),
            "--checkpoint_dir", str(cosmos_path / checkpoint_dir),
            "--diffusion_transformer_dir", "Diffusion_Renderer_Forward_Cosmos_7B",
            "--dataset_path", str(gbuffer_dir),
            "--num_video_frames", str(num_frames),
            "--envlight_ind", "0",  # Use index 0 since we only have one IBL in our custom list
            "--use_custom_envmap", "True",
            "--video_save_folder", str(output_dir),
            "--save_image", "True",
            "--height", str(height),
            "--width", str(width)
        ]
        
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env["CUDA_HOME"] = env.get("CONDA_PREFIX", "/usr/local/cuda")
        env["PYTHONPATH"] = str(cosmos_path)
        
        print(f"Running command: CUDA_HOME=$CONDA_PREFIX PYTHONPATH={cosmos_path} {' '.join(cmd)}")
        
        # Run the forward renderer
        result = subprocess.run(cmd, cwd=cosmos_path, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error running forward renderer:")
            print(f"Command: {' '.join(cmd)}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            print(f"💡 Troubleshooting tips:")
            print(f"   - Make sure you activated the 'cosmos-predict1' conda environment")
            print(f"   - Check that G-buffer directory contains valid files: {gbuffer_dir}")
            print(f"   - Verify that the IBL file exists and is valid: {ibl_file}")
            print(f"   - Check that the checkpoint directory exists: {cosmos_path / checkpoint_dir}")
            raise RuntimeError(f"Relighting with {ibl_file.name} failed")
        
        print(f"Relighting with {ibl_file.name} completed successfully")
        
    finally:
        # Clean up temporary script
        if temp_forward_script.exists():
            temp_forward_script.unlink()


def relight_with_all_ibls(gbuffer_dir, ibl_versions, cosmos_path, checkpoint_dir, output_dir, height, width, num_frames):
    """
    Efficiently relight G-buffers with all IBL versions in a single forward renderer call.
    This avoids reloading the model for each IBL version.
    
    Args:
        gbuffer_dir: Directory containing G-buffer frames
        ibl_versions: List of IBL version info dictionaries with 'ibl_file' and other metadata
        cosmos_path: Path to cosmos1-diffusion-renderer
        checkpoint_dir: Directory containing model checkpoints
        output_dir: Directory to save relit outputs
        height, width: Output dimensions
        num_frames: Number of frames per image
        
    Returns:
        Dictionary mapping version indices to output directories
    """
    print(f"🚀 Batch relighting with {len(ibl_versions)} IBL versions...")
    
    # Validate G-buffer directory structure
    gbuffer_path = Path(gbuffer_dir)
    
    if not gbuffer_path.exists():
        print(f"❌ G-buffer directory not found: {gbuffer_path}")
        # Check parent directory
        parent_dir = gbuffer_path.parent
        print(f"🔍 Checking parent directory: {parent_dir}")
        if parent_dir.exists():
            print(f"📁 Contents of parent directory:")
            for item in parent_dir.iterdir():
                print(f"   {'📁' if item.is_dir() else '📄'} {item.name}")
        raise FileNotFoundError(f"G-buffer directory not found: {gbuffer_path}")
    
    # Validate G-buffer files
    all_files = list(gbuffer_path.glob("*"))
    valid_gbuffer_files = []
    
    for file in all_files:
        if file.is_file():
            file_size = file.stat().st_size
            if (file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.exr'] or 
                (not file.suffix and file_size > 1000)):
                valid_gbuffer_files.append(file)
        elif file.is_dir():
            # Look inside subdirectories for G-buffer files
            subfiles = list(file.glob("*"))
            for subfile in subfiles:
                if subfile.is_file():
                    file_size = subfile.stat().st_size
                    if (subfile.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.exr'] or 
                        (not subfile.suffix and file_size > 1000)):
                        valid_gbuffer_files.append(subfile)
    
    if not valid_gbuffer_files:
        print(f"❌ No valid G-buffer files found in {gbuffer_path}")
        print(f"💡 Expected: image files with extensions (.png, .jpg, .jpeg, .tiff, .exr) or substantial files without extensions")
        print(f"💡 This suggests:")
        print(f"   - G-buffer extraction may have failed")
        print(f"   - Files were saved with unexpected format/location")
        print(f"   - Files are corrupted or empty")
        print(f"")
        print(f"🔧 Try re-running G-buffer extraction or check the files manually")
        
        raise FileNotFoundError(f"No valid G-buffer files found in {gbuffer_path}. Expected image files or files without extensions from cosmos1-diffusion-renderer.")
    
    print(f"✅ Found {len(valid_gbuffer_files)} valid G-buffer files")
    
    # Create a temporary modified forward renderer script with ALL IBL paths
    forward_script = cosmos_path / "cosmos_predict1/diffusion/inference/inference_forward_renderer.py"
    temp_forward_script = cosmos_path / "temp_inference_forward_renderer_batch.py"
    
    # Read the original script
    with open(forward_script, 'r') as f:
        script_content = f.read()
    
    # Build the new ENV_LIGHT_PATH_LIST with all IBL files
    ibl_paths = [str(version_data['ibl_file']) for version_data in ibl_versions]
    
    # Replace the ENV_LIGHT_PATH_LIST with all our custom IBLs
    ibl_list_str = '[\n    "' + '",\n    "'.join(ibl_paths) + '"\n]'
    ibl_line = f'ENV_LIGHT_PATH_LIST = {ibl_list_str}'
    
    # Find and replace the ENV_LIGHT_PATH_LIST definition
    lines = script_content.split('\n')
    new_lines = []
    in_env_list = False
    
    for line in lines:
        if line.strip().startswith('ENV_LIGHT_PATH_LIST = ['):
            new_lines.append(ibl_line)
            in_env_list = True
        elif in_env_list and line.strip() == ']':
            in_env_list = False
            continue
        elif in_env_list:
            continue
        else:
            new_lines.append(line)
    
    # Write the temporary script
    with open(temp_forward_script, 'w') as f:
        f.write('\n'.join(new_lines))
    
    # Prepare all envlight indices (0, 1, 2, ..., len(ibl_versions)-1)
    envlight_indices = [str(i) for i in range(len(ibl_versions))]
    
    try:
        # Run the forward renderer with ALL IBL versions at once
        cmd = [
            "python", str(temp_forward_script),
            "--checkpoint_dir", str(cosmos_path / checkpoint_dir),
            "--diffusion_transformer_dir", "Diffusion_Renderer_Forward_Cosmos_7B",
            "--dataset_path", str(gbuffer_dir),
            "--num_video_frames", str(num_frames),
            "--envlight_ind"] + envlight_indices + [  # Pass all indices
            "--use_custom_envmap", "True",
            "--video_save_folder", str(output_dir),
            "--save_image", "True",
            "--save_video", "False",  # We want images, not videos
            "--height", str(height),
            "--width", str(width)
        ]
        
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env["CUDA_HOME"] = env.get("CONDA_PREFIX", "/usr/local/cuda")
        env["PYTHONPATH"] = str(cosmos_path)
        
        print(f"Running command: CUDA_HOME=$CONDA_PREFIX PYTHONPATH={cosmos_path} {' '.join(cmd)}")
        print(f"💡 Processing all {len(ibl_versions)} lighting conditions in a single model load...")
        
        # Create progress bar for batch relighting
        with tqdm(total=len(ibl_versions), desc="💡 Batch relighting", unit="IBL versions") as pbar:
            # Run the forward renderer
            result = subprocess.run(cmd, cwd=cosmos_path, env=env, capture_output=True, text=True)
            
            # Update progress bar (complete all at once since it's batch processing)
            pbar.update(len(ibl_versions))
        
        if result.returncode != 0:
            print(f"❌ Error running batch forward renderer:")
            print(f"Command: {' '.join(cmd)}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            print(f"💡 Troubleshooting tips:")
            print(f"   - Make sure you activated the 'cosmos-predict1' conda environment")
            print(f"   - Check that G-buffer directory contains valid files: {gbuffer_dir}")
            print(f"   - Verify that all IBL files exist and are valid")
            print(f"   - Check that the checkpoint directory exists: {cosmos_path / checkpoint_dir}")
            raise RuntimeError(f"Batch relighting failed")
        
        print(f"✅ Batch relighting completed successfully for {len(ibl_versions)} IBL versions!")
        
        # Return mapping of version indices to their data  
        return {i: version_data for i, version_data in enumerate(ibl_versions)}
        
    finally:
        # Clean up temporary script
        if temp_forward_script.exists():
            temp_forward_script.unlink()


def organize_output_dataset(output_path, colmap_path, ibl_versions, temp_dir, original_ibl_files, num_versions, existing_dataset_info=None, downsample_factor=1.0):
    """
    Organize the output dataset in a structured format with multi-light versions.
    Can create a new dataset or append to an existing one.
    
    Creates:
    - images/ or images_N/: Original COLMAP images (downsampled if factor != 1.0)
    - relit_images/: Organized by version (v001, v002, etc.)
    - ibls/: Rotated IBL files used for each version
    - original_ibls/: Original IBL files 
    - dataset_info.json: Metadata about the dataset including version info
    
    Args:
        output_path: Path to output dataset
        colmap_path: Path to original COLMAP dataset
        ibl_versions: List of new IBL versions to add
        temp_dir: Temporary directory with relit images
        original_ibl_files: List of original IBL files used
        num_versions: Number of new versions being added
        existing_dataset_info: Existing dataset metadata (for append mode)
        downsample_factor: Factor to downsample images (default 1.0 = no downsampling)
    """
    output_path = Path(output_path)
    colmap_path = Path(colmap_path)
    
    # Determine if we're appending to existing dataset
    is_appending = existing_dataset_info is not None
    
    # Determine image directory name based on downsample factor
    if downsample_factor == 1.0:
        images_dir_name = "images"
    else:
        # Follow COLMAP convention: images_2, images_4, etc.
        downsample_int = int(downsample_factor) if downsample_factor == int(downsample_factor) else downsample_factor
        images_dir_name = f"images_{downsample_int}"
    
    # Create output directory structure
    output_images_dir = output_path / images_dir_name
    output_relit_dir = output_path / "relit_images"
    output_ibls_dir = output_path / "ibls"
    output_original_ibls_dir = output_path / "original_ibls"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_relit_dir.mkdir(parents=True, exist_ok=True)
    output_ibls_dir.mkdir(parents=True, exist_ok=True)
    output_original_ibls_dir.mkdir(parents=True, exist_ok=True)
    
    if not is_appending:
        # Copy original COLMAP dataset (only for new datasets)
        if (colmap_path / "sparse").exists():
            shutil.copytree(colmap_path / "sparse", output_path / "sparse", dirs_exist_ok=True)
        
        if (colmap_path / "images").exists():
            # Copy and potentially downsample original images
            image_files = [f for f in (colmap_path / "images").glob("*") 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            print(f"Copying {len(image_files)} images to {images_dir_name}/...")
            if downsample_factor != 1.0:
                print(f"Downsampling images by factor {downsample_factor}...")
            
            with tqdm(total=len(image_files), desc="📂 Copying images", unit="images") as pbar:
                for img_file in image_files:
                    pbar.set_postfix_str(img_file.name)
                    downsample_image(img_file, output_images_dir / img_file.name, downsample_factor)
                    pbar.update(1)
    
    # Handle original IBL files (merge with existing)
    if is_appending:
        # Load existing original IBL info
        original_ibl_info = existing_dataset_info.get('original_ibls', [])
        existing_ibl_names = {ibl['filename'] for ibl in original_ibl_info}
    else:
        original_ibl_info = []
        existing_ibl_names = set()
    
    # Add new original IBL files (avoid duplicates)
    for ibl_file in original_ibl_files:
        if ibl_file.name not in existing_ibl_names:
            shutil.copy2(ibl_file, output_original_ibls_dir / ibl_file.name)
            original_ibl_info.append({
                "filename": ibl_file.name,
                "name": ibl_file.stem,
                "path": f"original_ibls/{ibl_file.name}"
            })
    
    # Handle version info (merge with existing)
    if is_appending:
        # Load existing version info
        version_info = existing_dataset_info.get('versions', [])
    else:
        version_info = []
    
    # Add new versions with progress tracking
    with tqdm(total=len(ibl_versions), desc="📁 Organizing dataset", unit="versions") as pbar:
        for version_idx, rotated_ibl_path, original_ibl_name, rotation_degrees in ibl_versions:
            # Update progress description
            pbar.set_postfix_str(f"v{version_idx:03d}: copying files")
            
            # Copy rotated IBL to output
            version_ibl_name = rotated_ibl_path.name
            version_ibl_output_path = output_ibls_dir / version_ibl_name
            shutil.copy2(rotated_ibl_path, version_ibl_output_path)
            
            # Create directory for this version's relit images
            version_dir = output_relit_dir / f"v{version_idx:03d}"
            version_dir.mkdir(exist_ok=True)
            
            # Find and move relit images for this version
            temp_relit_dir = Path(temp_dir) / "relighting_output"
            if temp_relit_dir.exists():
                # Look for batch relighting outputs (pattern: *relit_{version_idx:04d}*)
                for relit_file in temp_relit_dir.glob(f"*relit_{version_idx:04d}*"):
                    if relit_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.mp4']:
                        shutil.copy2(relit_file, version_dir / relit_file.name)
                
                # Look for individual relighting outputs (in version subdirectories)
                version_temp_dir = temp_relit_dir / f"version_{version_idx:03d}"
                if version_temp_dir.exists():
                    for relit_file in version_temp_dir.glob("*"):
                        if relit_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.mp4']:
                            shutil.copy2(relit_file, version_dir / relit_file.name)
            
            version_info.append({
                "version_index": version_idx,
                "version_name": f"v{version_idx:03d}",
                "rotated_ibl_filename": version_ibl_name,
                "rotated_ibl_path": f"ibls/{version_ibl_name}",
                "original_ibl_name": original_ibl_name,
                "rotation_degrees": rotation_degrees,
                "relit_images_dir": f"relit_images/v{version_idx:03d}"
            })
            pbar.update(1)
    
    # Calculate total versions and images
    total_versions = len(version_info)
    total_images = len(list(output_images_dir.glob("*"))) if output_images_dir.exists() else 0
    
    # Create or update dataset info JSON
    dataset_info = {
        "type": "multi_lights_colmap_random",
        "original_colmap_path": str(colmap_path),
        "num_images": total_images,
        "num_versions": total_versions,
        "num_original_ibls": len(original_ibl_info),
        "downsample_factor": downsample_factor,
        "images_directory": images_dir_name,
        "versions": version_info,
        "original_ibls": original_ibl_info,
        "structure": {
            images_dir_name: f"Original COLMAP images (downsample factor: {downsample_factor})",
            "relit_images": "Relit images organized by version (v001, v002, etc.)",
            "ibls": "Rotated IBL environment maps used for each version",
            "original_ibls": "Original IBL files before rotation",
            "sparse": "COLMAP sparse reconstruction data",
            "gbuffers": "Preserved G-buffers for future reuse"
        },
        "generation_info": {
            "random_sampling": True,
            "random_rotation": True,
            "description": "Each version uses a randomly sampled IBL with random rotation",
            "append_supported": True,
            "downsample_applied": downsample_factor != 1.0
        }
    }
    
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    action_str = "appended to" if is_appending else "organized successfully in"
    print(f"Dataset {action_str} {output_path}")
    print(f"- {dataset_info['num_images']} original images")
    print(f"- {dataset_info['num_versions']} total lighting versions")
    if is_appending:
        print(f"- {num_versions} new versions added")
    print(f"- {dataset_info['num_original_ibls']} original IBL files")
    print(f"- Relit images organized by version in relit_images/")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multi-lights dataset from COLMAP dataset and IBL files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--colmap_path", 
        type=str, 
        help="Path to COLMAP dataset directory"
    )
    parser.add_argument(
        "--ibl_folder", 
        type=str, 
        help="Path to folder containing IBL files (.hdr/.exr)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        help="Path to output multi-lights dataset"
    )
    
    # Optional arguments
    parser.add_argument(
        "--cosmos_path", 
        type=str, 
        default="/mnt/dev/cosmos1-diffusion-renderer",
        help="Path to cosmos1-diffusion-renderer repository (default: /mnt/dev/cosmos1-diffusion-renderer)"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="checkpoints",
        help="Directory containing model checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=None,
        help="Output height (if not specified, auto-detected from images). Cosmos1 expects 704."
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=None,
        help="Output width (if not specified, auto-detected from images). Cosmos1 expects 1280."
    )
    parser.add_argument(
        "--num_frames", 
        type=int, 
        default=1,
        help="Number of frames per image (default: 1)"
    )
    parser.add_argument(
        "--num_versions", 
        type=int, 
        default=1,
        help="Number of multi-light versions to generate (default: 1)"
    )
    parser.add_argument(
        "--rotation_range", 
        type=float, 
        default=360.0,
        help="Maximum rotation range in degrees for random IBL rotation (default: 360.0)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducible IBL sampling and rotation (default: None = random)"
    )
    parser.add_argument(
        "--downsample_factor", 
        type=float, 
        default=1.0,
        help="Factor to downsample images (e.g., 2.0 = half resolution, default: 1.0)"
    )
    parser.add_argument(
        "--append", 
        action="store_true",
        help="Append new versions to existing dataset (reuses G-buffers)"
    )
    parser.add_argument(
        "--force_resolution", 
        action="store_true",
        help="Force use cosmos1-diffusion-renderer's expected resolution (704×1280)"
    )
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume from existing G-buffers (skip G-buffer extraction if they exist)"
    )
    parser.add_argument(
        "--force_individual_relighting", 
        action="store_true",
        help="Force individual relighting instead of batch processing (slower but more reliable)"
    )
    
    return parser.parse_args()


def main():
    """Main function to generate multi-lights dataset."""
    args = parse_arguments()
    
    print("=== Cosmos1-Diffusion-Renderer Multi-Lights Dataset Generator ===")
    print("IMPORTANT: Make sure you have activated the 'cosmos-predict1' conda environment")
    print("before running this script:")
    print("  conda activate cosmos-predict1\n")
    
    # Set up paths
    colmap_path = Path(args.colmap_path).resolve()
    ibl_folder = Path(args.ibl_folder).resolve()
    output_path = Path(args.output_path).resolve()
    cosmos_path = setup_cosmos_environment(args.cosmos_path)
    
    # Check for append mode
    existing_dataset_info = None
    existing_gbuffer_dir = None
    start_version_idx = 0
    
    if args.append:
        print("=== Append Mode: Adding to existing dataset ===")
        is_valid, existing_dataset_info, existing_gbuffer_dir = check_existing_dataset(output_path)
        
        if not is_valid:
            if existing_dataset_info is None:
                print(f"\n❌ Cannot use --append: No existing multi-lights dataset found at {output_path}")
                print(f"\n💡 To fix this:")
                print(f"   1. First create the dataset (remove --append):")
                print(f"      python {sys.argv[0]} \\")
                print(f"          --colmap_path {args.colmap_path} \\")  
                print(f"          --ibl_folder {args.ibl_folder} \\")
                print(f"          --output_path {args.output_path} \\")
                if args.force_resolution:
                    print(f"          --force_resolution \\")
                if args.downsample_factor != 1.0:
                    print(f"          --downsample_factor {args.downsample_factor} \\")
                print(f"          --num_versions {args.num_versions}")
                print(f"")
                print(f"   2. Then append more versions:")
                print(f"      python {sys.argv[0]} \\")
                print(f"          --ibl_folder /path/to/more/ibls \\")
                print(f"          --output_path {args.output_path} \\")
                print(f"          --num_versions 3 \\")
                print(f"          --append")
                raise ValueError(f"No existing dataset found at {output_path}. Use without --append to create new dataset.")
            else:
                raise ValueError(f"Existing dataset found but G-buffers are missing. Cannot append without G-buffers.")
        
        start_version_idx = get_next_version_index(existing_dataset_info)
        print(f"Will add {args.num_versions} new versions starting from v{start_version_idx:03d}")
        
        # For append mode, we need the original COLMAP path from existing dataset
        if existing_dataset_info and 'original_colmap_path' in existing_dataset_info:
            colmap_path = Path(existing_dataset_info['original_colmap_path'])
    else:
        print("=== New Dataset Mode ===")
    
    # Validate inputs
    if not colmap_path.exists():
        raise ValueError(f"COLMAP dataset not found: {colmap_path}")
    if not ibl_folder.exists():
        raise ValueError(f"IBL folder not found: {ibl_folder}")
    if not cosmos_path.exists():
        raise ValueError(f"Cosmos1-diffusion-renderer not found: {cosmos_path}")
    
    # Handle resolution specification
    if args.force_resolution:
        # Force cosmos1-diffusion-renderer's expected resolution
        height, width = 704, 1280
        print("🔧 Forcing cosmos1-diffusion-renderer's expected resolution: 704×1280")
        if args.downsample_factor != 1.0:
            height = int(height / args.downsample_factor)
            width = int(width / args.downsample_factor)
            print(f"   After downsample factor {args.downsample_factor}: {width}×{height}")
    elif args.height is None or args.width is None:
        print("Auto-detecting image resolution from COLMAP dataset...")
        detected_height, detected_width = detect_image_resolution(colmap_path, args.downsample_factor)
        height = args.height if args.height is not None else detected_height
        width = args.width if args.width is not None else detected_width
    else:
        height = args.height
        width = args.width
        # Apply downsampling to manually specified resolution
        if args.downsample_factor != 1.0:
            height = int(height / args.downsample_factor)
            width = int(width / args.downsample_factor)
            print(f"Applying downsample factor {args.downsample_factor} to specified resolution")
    
    print(f"Generation settings:")
    print(f"  - Number of versions: {args.num_versions}")
    print(f"  - Rotation range: ±{args.rotation_range/2:.1f}°")
    print(f"  - Random seed: {args.seed if args.seed else 'None (random)'}")
    print(f"  - Downsample factor: {args.downsample_factor}")
    print(f"  - Output resolution: {width}x{height}")
    if args.height is None or args.width is None:
        print(f"    (auto-detected from input images)")
    print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create temporary working directory
    temp_dir = output_path / "temp_processing"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        if args.append:
            print("=== Starting Multi-Lights Dataset Append ===")
        else:
            print("=== Starting Multi-Lights Dataset Generation ===")
        
        # Handle G-buffers: extract new, use existing from append, or resume from failed run
        if args.append and existing_gbuffer_dir:
            print("\n1. Using existing G-buffers...")
            gbuffer_dir = existing_gbuffer_dir
            print(f"G-buffers found at: {gbuffer_dir}")
        elif args.resume or not args.append:
            # Check for existing G-buffers from previous run
            existing_gbuffer_dir = check_existing_gbuffers(output_path)
            
            if existing_gbuffer_dir and (args.resume or args.append):
                print("\n1. Using existing G-buffers from previous run...")
                gbuffer_dir = existing_gbuffer_dir
                print(f"G-buffers found at: {gbuffer_dir}")
                
                # Preserve G-buffers if they're still in temporary location
                if "temp_processing" in str(gbuffer_dir):
                    preserve_gbuffers(gbuffer_dir, output_path)
            else:
                # Step 1: Prepare COLMAP images
                print("\n1. Preparing COLMAP images...")
                images_dir = prepare_colmap_images(colmap_path, temp_dir)
                
                # Step 2: Extract G-buffers (only once for all versions)
                print("\n2. Extracting G-buffers...")
                gbuffer_output_dir = temp_dir / "gbuffer_output"
                gbuffer_dir = extract_gbuffers(
                    images_dir, cosmos_path, args.checkpoint_dir, 
                    gbuffer_output_dir, height, width, args.num_frames
                )
                
                # Preserve G-buffers for future reuse
                if not args.append:  # Only preserve for new datasets
                    preserve_gbuffers(gbuffer_dir, output_path)
        else:
            # Step 1: Prepare COLMAP images
            print("\n1. Preparing COLMAP images...")
            images_dir = prepare_colmap_images(colmap_path, temp_dir)
            
            # Step 2: Extract G-buffers (only once for all versions)
            print("\n2. Extracting G-buffers...")
            gbuffer_output_dir = temp_dir / "gbuffer_output"
            gbuffer_dir = extract_gbuffers(
                images_dir, cosmos_path, args.checkpoint_dir, 
                gbuffer_output_dir, height, width, args.num_frames
            )
            
            # Preserve G-buffers for future reuse
            if not args.append:  # Only preserve for new datasets
                preserve_gbuffers(gbuffer_dir, output_path)
        
        # Step 3: Find original IBL files
        print("\n3. Finding IBL files...")
        original_ibl_files = find_ibl_files(ibl_folder)
        
        # Adjust step numbers for append mode
        step_offset = 1 if args.append else 2
        
        # Step: Generate random IBL versions with rotations
        print(f"\n{step_offset + 2}. Generating {args.num_versions} random IBL versions...")
        rotated_ibls_dir = temp_dir / "rotated_ibls"
        
        # Set random seed if specified
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
        
        # Generate versions with correct starting index
        ibl_versions = []
        with tqdm(total=args.num_versions, desc="🎭 Generating IBL versions", unit="versions") as pbar:
            for i in range(args.num_versions):
                version_idx = start_version_idx + i
                selected_ibl = random.choice(original_ibl_files)
                rotation_degrees = random.uniform(-args.rotation_range/2, args.rotation_range/2)
                
                # Create output filename for rotated IBL
                original_name = selected_ibl.stem
                extension = selected_ibl.suffix
                rotated_filename = f"v{version_idx:03d}_{original_name}_rot{rotation_degrees:.1f}{extension}"
                rotated_path = rotated_ibls_dir / rotated_filename
                rotated_ibls_dir.mkdir(exist_ok=True)
                
                # Update progress description with current details
                pbar.set_postfix_str(f"v{version_idx:03d}: {selected_ibl.name} ({rotation_degrees:.1f}°)")
                
                # Rotate and save the IBL
                rotate_hdr_image(selected_ibl, rotated_path, rotation_degrees)
                
                ibl_versions.append((version_idx, rotated_path, selected_ibl.name, rotation_degrees))
                pbar.update(1)
        
        # Step 5: Relight with ALL versions in a single batch call (much more efficient!)
        final_step = step_offset + 3
        print(f"\n{final_step}. Batch relighting with {args.num_versions} IBL versions...")
        relighting_output_dir = temp_dir / "relighting_output"
        
        # Convert ibl_versions data to format expected by batch function
        ibl_versions_for_batch = []
        for version_idx, rotated_ibl_path, original_ibl_name, rotation_degrees in ibl_versions:
            ibl_versions_for_batch.append({
                'ibl_file': rotated_ibl_path,
                'version_idx': version_idx,
                'original_name': original_ibl_name,
                'rotation_degrees': rotation_degrees
            })
        
        # Choose relighting strategy
        if args.force_individual_relighting:
            print(f"💡 Using individual relighting (as requested with --force_individual_relighting)")
            # Use individual relighting
            relighting_output_dir.mkdir(parents=True, exist_ok=True)
            
            successful_versions = []
            failed_versions = []
            
            with tqdm(total=len(ibl_versions), desc="💡 Individual relighting", unit="IBL versions") as pbar:
                for i, (version_idx, rotated_ibl_path, original_ibl_name, rotation_degrees) in enumerate(ibl_versions):
                    pbar.set_postfix_str(f"v{version_idx:03d}: {original_ibl_name}")
                    
                    version_output_dir = relighting_output_dir / f"version_{version_idx:03d}"
                    version_output_dir.mkdir(exist_ok=True)
                    
                    try:
                        relight_with_custom_ibl(
                            gbuffer_dir, rotated_ibl_path, cosmos_path, args.checkpoint_dir,
                            version_output_dir, height, width, args.num_frames, i
                        )
                        successful_versions.append((version_idx, original_ibl_name))
                    except Exception as e:
                        print(f"\n⚠️  Failed to relight with {original_ibl_name}: {e}")
                        failed_versions.append((version_idx, original_ibl_name, str(e)))
                    
                    pbar.update(1)
            
            # Update ibl_versions to only include successful ones
            ibl_versions = [(v, p, n, r) for v, p, n, r in ibl_versions 
                           if v in [sv[0] for sv in successful_versions]]
            
            print(f"✅ Individual relighting completed for {len(successful_versions)} IBL versions")
            if failed_versions:
                print(f"⚠️  {len(failed_versions)} versions failed:")
                for version_idx, name, error in failed_versions[:3]:
                    print(f"   ❌ v{version_idx:03d}: {name} - {error}")
                if len(failed_versions) > 3:
                    print(f"   ... and {len(failed_versions) - 3} more")
        else:
            # Try batch relighting first, fall back to individual if it fails
            try:
                print(f"💡 Attempting batch relighting (faster, but may fail with some G-buffer formats)...")
                relight_with_all_ibls(
                    gbuffer_dir, ibl_versions_for_batch, cosmos_path, args.checkpoint_dir,
                    relighting_output_dir, height, width, args.num_frames
                )
            except RuntimeError as e:
                if "Batch relighting failed" in str(e):
                    print(f"⚠️  Batch relighting failed, falling back to individual relighting...")
                    print(f"💡 This will be slower but more reliable.")
                    
                    # Fall back to individual relighting
                    relighting_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    successful_versions = []
                    failed_versions = []
                    
                    with tqdm(total=len(ibl_versions), desc="💡 Individual relighting", unit="IBL versions") as pbar:
                        for i, (version_idx, rotated_ibl_path, original_ibl_name, rotation_degrees) in enumerate(ibl_versions):
                            pbar.set_postfix_str(f"v{version_idx:03d}: {original_ibl_name}")
                            
                            version_output_dir = relighting_output_dir / f"version_{version_idx:03d}"
                            version_output_dir.mkdir(exist_ok=True)
                            
                            try:
                                relight_with_custom_ibl(
                                    gbuffer_dir, rotated_ibl_path, cosmos_path, args.checkpoint_dir,
                                    version_output_dir, height, width, args.num_frames, i
                                )
                                successful_versions.append((version_idx, original_ibl_name))
                            except Exception as e:
                                print(f"\n⚠️  Failed to relight with {original_ibl_name}: {e}")
                                failed_versions.append((version_idx, original_ibl_name, str(e)))
                            
                            pbar.update(1)
                    
                    # Update ibl_versions to only include successful ones
                    ibl_versions = [(v, p, n, r) for v, p, n, r in ibl_versions 
                                   if v in [sv[0] for sv in successful_versions]]
                    
                    print(f"✅ Individual relighting completed for {len(successful_versions)} IBL versions")
                    if failed_versions:
                        print(f"⚠️  {len(failed_versions)} versions failed:")
                        for version_idx, name, error in failed_versions[:3]:
                            print(f"   ❌ v{version_idx:03d}: {name} - {error}")
                        if len(failed_versions) > 3:
                            print(f"   ... and {len(failed_versions) - 3} more")
                else:
                    raise
        
        # Final step: Organize dataset
        final_step = step_offset + 4
        action_str = "Updating" if args.append else "Organizing"
        print(f"\n{final_step}. {action_str} final dataset...")
        organize_output_dataset(
            output_path, colmap_path, ibl_versions, temp_dir, 
            original_ibl_files, args.num_versions, existing_dataset_info, args.downsample_factor
        )
        
        if args.append:
            print(f"\n=== Multi-Lights Dataset Append Complete ===")
            print(f"Added {args.num_versions} new lighting versions to existing dataset!")
        else:
            print(f"\n=== Multi-Lights Dataset Generation Complete ===")
            print(f"Generated {args.num_versions} lighting versions with random sampling and rotation!")
        
        print(f"Dataset location: {output_path}")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        raise
    
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
