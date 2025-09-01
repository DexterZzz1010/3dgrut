#!/usr/bin/env python3
"""
Script to extract G-buffers from a COLMAP dataset using cosmos1-diffusion-renderer.

This script takes a COLMAP dataset and extracts G-buffers (basecolor, depth, metallic, 
normal, roughness) using the cosmos1-diffusion-renderer inverse renderer. The extracted
G-buffers are organized by image name and saved to a specified output directory.

Features:
- Auto-detects image resolution from COLMAP dataset
- Supports downsampling for faster processing
- Compatible with cosmos1-diffusion-renderer checkpoints
- Organized output with metadata

Usage:
    # Basic usage
    conda activate cosmos-predict1
    python extract_gbuffers.py \
        --colmap_path /path/to/colmap/dataset \
        --output_path /path/to/gbuffers/output

    # With downsampling for faster processing
    python extract_gbuffers.py \
        --colmap_path /path/to/colmap/dataset \
        --output_path /path/to/gbuffers/output \
        --downsample_factor 2.0

    # Force specific resolution
    python extract_gbuffers.py \
        --colmap_path /path/to/colmap/dataset \
        --output_path /path/to/gbuffers/output \
        --force_resolution \
        --height 704 \
        --width 1280
"""

import argparse
import os
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract G-buffers from COLMAP dataset using cosmos1-diffusion-renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--colmap_path", 
        type=str, 
        required=True,
        help="Path to COLMAP dataset directory"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Path to output directory for G-buffers"
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
        "--downsample_factor", 
        type=float, 
        default=1.0,
        help="Factor to downsample images (e.g., 2.0 = half resolution, default: 1.0)"
    )
    parser.add_argument(
        "--force_resolution", 
        action="store_true",
        help="Force use cosmos1-diffusion-renderer's expected resolution (704×1280)"
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


def prepare_colmap_images(colmap_path, temp_dir, downsample_factor=1.0):
    """
    Prepare COLMAP images for processing by cosmos1-diffusion-renderer.
    
    Args:
        colmap_path: Path to COLMAP dataset
        temp_dir: Temporary directory for processing
        downsample_factor: Factor to downsample images
        
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
    
    # Copy and potentially downsample images to temp directory
    with tqdm(total=len(image_files), desc="📂 Preparing images", unit="images") as pbar:
        for img_file in image_files:
            pbar.set_postfix_str(img_file.name)
            output_path = temp_images_dir / img_file.name
            downsample_image(img_file, output_path, downsample_factor)
            pbar.update(1)
    
    return temp_images_dir


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


def organize_gbuffers(gbuffer_dir, output_path, colmap_path, height, width, downsample_factor):
    """
    Organize the extracted G-buffers in a structured format.
    
    Args:
        gbuffer_dir: Source G-buffer directory
        output_path: Final output directory
        colmap_path: Original COLMAP dataset path
        height, width: Processing resolution
        downsample_factor: Downsampling factor used
    """
    gbuffer_dir = Path(gbuffer_dir)
    output_path = Path(output_path)
    colmap_path = Path(colmap_path)
    
    # Create final output directory
    final_gbuffer_dir = output_path / "gbuffers"
    final_gbuffer_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Organizing G-buffers...")
    
    # Copy G-buffers to final location
    if gbuffer_dir != final_gbuffer_dir:
        shutil.copytree(gbuffer_dir, final_gbuffer_dir, dirs_exist_ok=True)
    
    # Count organized files
    organized_files = []
    for item in final_gbuffer_dir.rglob("*"):
        if item.is_file():
            file_size = item.stat().st_size
            if (item.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.exr'] or 
                (not item.suffix and file_size > 1000)):
                organized_files.append(item)
    
    # Create metadata
    metadata = {
        "type": "gbuffers_colmap",
        "original_colmap_path": str(colmap_path),
        "extraction_info": {
            "height": height,
            "width": width,
            "downsample_factor": downsample_factor,
            "num_gbuffer_files": len(organized_files),
            "extraction_tool": "cosmos1-diffusion-renderer",
            "gbuffer_types": ["basecolor", "depth", "metallic", "normal", "roughness"]
        },
        "structure": {
            "gbuffers/": "G-buffer files organized by image name",
            "dataset_info.json": "Metadata about the G-buffer extraction"
        }
    }
    
    # Save metadata
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ G-buffers organized successfully:")
    print(f"   - {len(organized_files)} G-buffer files")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - Downsample factor: {downsample_factor}")
    print(f"   - Output: {output_path}")


def main():
    """Main function to extract G-buffers from COLMAP dataset."""
    args = parse_arguments()
    
    print("=== Cosmos1-Diffusion-Renderer G-Buffer Extractor ===")
    print("IMPORTANT: Make sure you have activated the 'cosmos-predict1' conda environment")
    print("before running this script:")
    print("  conda activate cosmos-predict1\n")
    
    # Set up paths
    colmap_path = Path(args.colmap_path).resolve()
    output_path = Path(args.output_path).resolve()
    cosmos_path = setup_cosmos_environment(args.cosmos_path)
    
    # Validate inputs
    if not colmap_path.exists():
        raise ValueError(f"COLMAP dataset not found: {colmap_path}")
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
    
    print(f"Extraction settings:")
    print(f"  - Downsample factor: {args.downsample_factor}")
    print(f"  - Processing resolution: {width}x{height}")
    print(f"  - Number of frames: {args.num_frames}")
    if args.height is None or args.width is None:
        print(f"    (auto-detected from input images)")
    print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        try:
            print("=== Starting G-Buffer Extraction ===")
            
            # Step 1: Prepare COLMAP images
            print("\n1. Preparing COLMAP images...")
            images_dir = prepare_colmap_images(colmap_path, temp_dir, args.downsample_factor)
            
            # Step 2: Extract G-buffers
            print("\n2. Extracting G-buffers...")
            gbuffer_output_dir = temp_dir / "gbuffer_output"
            gbuffer_dir = extract_gbuffers(
                images_dir, cosmos_path, args.checkpoint_dir, 
                gbuffer_output_dir, height, width, args.num_frames
            )
            
            # Step 3: Organize final output
            print("\n3. Organizing final output...")
            organize_gbuffers(gbuffer_dir, output_path, colmap_path, height, width, args.downsample_factor)
            
            print(f"\n=== G-Buffer Extraction Complete ===")
            print(f"G-buffers extracted and organized at: {output_path}")
            
        except Exception as e:
            print(f"\nError during processing: {e}")
            raise


if __name__ == "__main__":
    main()
