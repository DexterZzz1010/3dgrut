#!/usr/bin/env python3
"""
Script to create a complete COLMAP G-buffer dataset from an existing COLMAP dataset.

This script takes a COLMAP dataset and creates a complete, standalone COLMAP G-buffer dataset
that includes:
- Sparse reconstruction data (cameras, images, points3D) copied from original
- Images folder with resized images (matching G-buffer extraction resolution)
- G-buffers folder with extracted G-buffers (basecolor, depth, metallic, normal, roughness)
- Complete metadata for the new dataset

The output dataset is completely independent from the original COLMAP dataset.

Features:
- Creates standalone COLMAP G-buffer dataset
- Auto-detects image resolution from COLMAP dataset
- Supports downsampling for faster processing
- Compatible with cosmos1-diffusion-renderer checkpoints
- Copies COLMAP sparse reconstruction data
- Resizes images to match G-buffer extraction resolution

Usage:
    # Basic usage
    conda activate cosmos-predict1
    python create_colmap_gbuffers.py \
        --input_colmap_path /path/to/original/colmap/dataset \
        --output_path /path/to/new/colmap_gbuffer/dataset

    # With downsampling for faster processing
    python create_colmap_gbuffers.py \
        --input_colmap_path /path/to/original/colmap/dataset \
        --output_path /path/to/new/colmap_gbuffer/dataset \
        --downsample_factor 2.0

    # Force specific resolution
    python create_colmap_gbuffers.py \
        --input_colmap_path /path/to/original/colmap/dataset \
        --output_path /path/to/new/colmap_gbuffer/dataset \
        --force_resolution \
        --height 704 \
        --width 1280

Output Structure:
    /output/path/
    ├── images/              # Resized images (matching G-buffer resolution)
    ├── sparse/             # COLMAP reconstruction (copied from original)
    │   └── 0/
    │       ├── cameras.bin
    │       ├── images.bin
    │       └── points3D.bin
    ├── gbuffers/           # Extracted G-buffers
    │   ├── image1/
    │   │   ├── 0000.0000.basecolor.jpg
    │   │   ├── 0000.0000.depth.jpg
    │   │   ├── 0000.0000.normal.jpg
    │   │   ├── 0000.0000.metallic.jpg
    │   │   └── 0000.0000.roughness.jpg
    │   └── ...
    └── dataset_info.json   # Complete dataset metadata
"""

import argparse
import os
import json
import shutil
import subprocess
import sys
import tempfile
import struct
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter, namedtuple
from tqdm import tqdm


def get_camera_model_params_count(model_id):
    """Get the number of parameters for a COLMAP camera model.
    
    Model IDs must match those expected by 3dgrut's dataset loader.
    """
    model_params = {
        0: 3,   # SIMPLE_PINHOLE: f, cx, cy
        1: 4,   # PINHOLE: fx, fy, cx, cy  
        2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
        3: 5,   # RADIAL: f, cx, cy, k1, k2
        4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        5: 8,   # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
        6: 12,  # FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        7: 5,   # FOV: fx, fy, cx, cy, omega
        8: 4,   # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k
        9: 5,   # RADIAL_FISHEYE: f, cx, cy, k1, k2
        10: 12, # THIN_PRISM_FISHEYE: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
    }
    return model_params.get(model_id, 0)


# COLMAP data structures
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
ColmapImage = namedtuple("ColmapImage", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "track"])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """Read cameras from COLMAP cameras.bin file."""
    cameras = {}
    try:
        with open(path_to_model_file, "rb") as fid:
            num_cameras = read_next_bytes(fid, 8, "Q")[0]
            print(f"📷 Reading {num_cameras} cameras from binary file...")
            
            for camera_idx in range(num_cameras):
                camera_properties = read_next_bytes(fid, 24, "iiQQ")
                camera_id = camera_properties[0]
                model_id = camera_properties[1]
                width = camera_properties[2]
                height = camera_properties[3]
                
                # Get number of parameters based on camera model (not read from file!)
                num_params = get_camera_model_params_count(model_id)
                
                if num_params == 0:
                    raise ValueError(f"Unknown camera model: {model_id}")
                
                print(f"  Camera {camera_id}: model={model_id}, size={width}×{height}, params={num_params}")
                
                params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
                cameras[camera_id] = Camera(
                    id=camera_id, model=model_id, width=width, height=height, params=params
                )
        return cameras
    except Exception as e:
        print(f"❌ Error reading cameras.bin: {e}")
        print(f"   File: {path_to_model_file}")
        print(f"   File size: {Path(path_to_model_file).stat().st_size} bytes")
        raise


def write_cameras_binary(cameras, path_to_model_file):
    """Write cameras to COLMAP cameras.bin file."""
    with open(path_to_model_file, "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras.values():
            fid.write(struct.pack("<iiQQ", camera.id, camera.model, camera.width, camera.height))
            # Note: Don't write num_params - it's implicit based on camera model
            for param in camera.params:
                fid.write(struct.pack("<d", param))


def read_images_binary(path_to_model_file):
    """Read images from COLMAP images.bin file."""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = ColmapImage(
                id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids
            )
    return images


def write_images_binary(images, path_to_model_file):
    """Write images to COLMAP images.bin file."""
    with open(path_to_model_file, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image in images.values():
            fid.write(struct.pack("<i", image.id))
            fid.write(struct.pack("<dddd", *image.qvec))
            fid.write(struct.pack("<ddd", *image.tvec))
            fid.write(struct.pack("<i", image.camera_id))
            fid.write(image.name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(image.point3D_ids)))
            for xy, point3D_id in zip(image.xys, image.point3D_ids):
                fid.write(struct.pack("<ddq", xy[0], xy[1], point3D_id))


def scale_colmap_cameras(cameras, original_width, original_height, target_width, target_height):
    """Scale camera intrinsic parameters for new image resolution with crop adjustment."""
    scaled_cameras = {}
    
    # Calculate crop parameters for aspect ratio preservation
    crop_x_offset, crop_y_offset, cropped_width, cropped_height = calculate_crop_parameters(
        original_width, original_height, target_width, target_height
    )
    
    # Calculate scale factors from cropped dimensions to target dimensions
    scale_x = target_width / cropped_width
    scale_y = target_height / cropped_height
    
    print(f"📐 Crop and scale analysis:")
    print(f"   Original: {original_width}×{original_height}")
    print(f"   Crop offset: ({crop_x_offset}, {crop_y_offset})")
    print(f"   Cropped: {cropped_width}×{cropped_height}")
    print(f"   Final scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
    
    for camera_id, camera in cameras.items():
        # Scale intrinsic parameters based on camera model
        scaled_params = list(camera.params)
        
        if camera.model == 0:  # SIMPLE_PINHOLE: f, cx, cy
            scaled_params[0] *= (scale_x + scale_y) / 2  # focal length (average scaling)
            scaled_params[1] = (scaled_params[1] - crop_x_offset) * scale_x  # cx (crop then scale)
            scaled_params[2] = (scaled_params[2] - crop_y_offset) * scale_y  # cy (crop then scale)
        elif camera.model == 1:  # PINHOLE: fx, fy, cx, cy
            scaled_params[0] *= scale_x  # fx
            scaled_params[1] *= scale_y  # fy
            scaled_params[2] = (scaled_params[2] - crop_x_offset) * scale_x  # cx (crop then scale)
            scaled_params[3] = (scaled_params[3] - crop_y_offset) * scale_y  # cy (crop then scale)
        elif camera.model == 2:  # SIMPLE_RADIAL: f, cx, cy, k
            scaled_params[0] *= (scale_x + scale_y) / 2  # focal length (average scaling)
            scaled_params[1] = (scaled_params[1] - crop_x_offset) * scale_x  # cx (crop then scale)
            scaled_params[2] = (scaled_params[2] - crop_y_offset) * scale_y  # cy (crop then scale)
            # k (distortion) remains the same
        elif camera.model == 3:  # RADIAL: f, cx, cy, k1, k2
            scaled_params[0] *= (scale_x + scale_y) / 2  # focal length (average scaling)
            scaled_params[1] = (scaled_params[1] - crop_x_offset) * scale_x  # cx (crop then scale)
            scaled_params[2] = (scaled_params[2] - crop_y_offset) * scale_y  # cy (crop then scale)
            # k1, k2 (distortion) remain the same
        elif camera.model == 4:  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
            scaled_params[0] *= scale_x  # fx
            scaled_params[1] *= scale_y  # fy
            scaled_params[2] = (scaled_params[2] - crop_x_offset) * scale_x  # cx (crop then scale)
            scaled_params[3] = (scaled_params[3] - crop_y_offset) * scale_y  # cy (crop then scale)
            # k1, k2, p1, p2 (distortion) remain the same
        else:
            print(f"⚠️ Warning: Unknown camera model {camera.model}, copying parameters as-is")
        
        scaled_cameras[camera_id] = Camera(
            id=camera.id,
            model=camera.model,
            width=target_width,
            height=target_height,
            params=tuple(scaled_params)
        )
    
    return scaled_cameras


def scale_colmap_images(images, original_width, original_height, target_width, target_height):
    """Scale 2D point observations in images for new image resolution with crop adjustment."""
    scaled_images = {}
    
    # Calculate crop parameters for aspect ratio preservation
    crop_x_offset, crop_y_offset, cropped_width, cropped_height = calculate_crop_parameters(
        original_width, original_height, target_width, target_height
    )
    
    # Calculate scale factors from cropped dimensions to target dimensions
    scale_x = target_width / cropped_width
    scale_y = target_height / cropped_height
    
    for image_id, image in images.items():
        # Adjust 2D point coordinates for crop and scale
        scaled_xys = image.xys.copy()
        scaled_xys[:, 0] = (scaled_xys[:, 0] - crop_x_offset) * scale_x  # x: crop then scale
        scaled_xys[:, 1] = (scaled_xys[:, 1] - crop_y_offset) * scale_y  # y: crop then scale
        
        scaled_images[image_id] = ColmapImage(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=scaled_xys,
            point3D_ids=image.point3D_ids
        )
    
    return scaled_images


def filter_colmap_data_by_images(cameras, images, selected_image_names):
    """
    Filter COLMAP cameras and images to only include data for selected images.
    
    Args:
        cameras: Dict of camera data from read_cameras_binary
        images: Dict of image data from read_images_binary  
        selected_image_names: Set of image names to keep
        
    Returns:
        Tuple of (filtered_cameras, filtered_images)
    """
    # Filter images to only include selected ones
    filtered_images = {}
    used_camera_ids = set()
    
    for image_id, image in images.items():
        if image.name in selected_image_names:
            filtered_images[image_id] = image
            used_camera_ids.add(image.camera_id)
    
    # Filter cameras to only include those referenced by filtered images
    filtered_cameras = {}
    for camera_id, camera in cameras.items():
        if camera_id in used_camera_ids:
            filtered_cameras[camera_id] = camera
    
    print(f"📊 Filtered COLMAP data:")
    print(f"   Original: {len(images)} images, {len(cameras)} cameras")
    print(f"   Filtered: {len(filtered_images)} images, {len(filtered_cameras)} cameras")
    
    return filtered_cameras, filtered_images


def get_limited_image_list(images_dir, max_images=None):
    """
    Get a limited list of image files for processing.
    
    Args:
        images_dir: Path to images directory
        max_images: Maximum number of images to include (None = all)
        
    Returns:
        List of image file paths, limited to max_images
    """
    images_dir = Path(images_dir)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(images_dir.glob(f"*{ext}"))
    
    # Sort for consistent ordering
    all_images = sorted(all_images)
    
    if max_images is not None and len(all_images) > max_images:
        selected_images = all_images[:max_images]
        print(f"📊 Image selection for quick testing:")
        print(f"   Total images available: {len(all_images)}")
        print(f"   Selected for processing: {len(selected_images)}")
        print(f"   Using first {max_images} images alphabetically")
        return selected_images
    else:
        print(f"📊 Processing all {len(all_images)} images")
        return all_images


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a complete COLMAP G-buffer dataset from an existing COLMAP dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input_colmap_path", 
        type=str, 
        required=True,
        help="Path to the input COLMAP dataset"
    )
    
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Path to create the output COLMAP G-buffer dataset"
    )
    
    parser.add_argument(
        "--cosmos_path", 
        type=str, 
        default="../cosmos1-diffusion-renderer",
        help="Path to cosmos1-diffusion-renderer repository (default: ../cosmos1-diffusion-renderer)"
    )
    
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="checkpoints",
        help="Directory containing model checkpoints relative to cosmos_path (default: checkpoints)"
    )
    
    parser.add_argument(
        "--downsample_factor", 
        type=float, 
        default=1.0,
        help="Factor to downsample images (default: 1.0, no downsampling)"
    )
    
    parser.add_argument(
        "--force_resolution", 
        action="store_true",
        help="Force specific resolution instead of auto-detecting from images"
    )
    
    parser.add_argument(
        "--height", 
        type=int, 
        help="Height for G-buffer extraction (only used with --force_resolution)"
    )
    
    parser.add_argument(
        "--width", 
        type=int, 
        help="Width for G-buffer extraction (only used with --force_resolution)"
    )
    
    parser.add_argument(
        "--num_frames", 
        type=int, 
        default=1,
        help="Number of frames per image for G-buffer extraction (default: 1)"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: None, process all images). Useful for quick testing."
    )
    
    return parser.parse_args()


def setup_cosmos_environment(cosmos_path):
    """Set up and validate the cosmos1-diffusion-renderer environment."""
    cosmos_path = Path(cosmos_path)
    
    if not cosmos_path.exists():
        raise FileNotFoundError(f"Cosmos path does not exist: {cosmos_path}")
    
    # Check for required scripts
    inverse_script = cosmos_path / "cosmos_predict1/diffusion/inference/inference_inverse_renderer.py"
    if not inverse_script.exists():
        raise FileNotFoundError(f"Inverse renderer script not found: {inverse_script}")
    
    print(f"✅ Cosmos environment validated: {cosmos_path}")
    return cosmos_path


def detect_image_resolution(colmap_path):
    """
    Auto-detect image resolution from COLMAP dataset images.
    
    Returns the most common resolution found in the images directory.
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
    
    print(f"🔍 Analyzing {len(image_files)} images to detect resolution...")
    
    # Sample a subset of images for resolution detection
    sample_size = min(10, len(image_files))
    sample_files = image_files[:sample_size]
    
    resolutions = []
    with tqdm(sample_files, desc="📏 Detecting resolution", unit="images") as pbar:
        for img_path in pbar:
            try:
                with Image.open(img_path) as img:
                    resolutions.append((img.width, img.height))
                pbar.set_postfix_str(f"{img.width}×{img.height}")
            except Exception as e:
                print(f"⚠️ Warning: Could not read {img_path}: {e}")
    
    if not resolutions:
        raise ValueError("Could not detect resolution from any images")
    
    # Find most common resolution
    resolution_counts = Counter(resolutions)
    most_common_resolution = resolution_counts.most_common(1)[0][0]
    width, height = most_common_resolution
    
    print(f"✅ Detected resolution: {width}×{height}")
    if len(set(resolutions)) > 1:
        print(f"⚠️ Warning: Multiple resolutions found. Using most common: {width}×{height}")
        for res, count in resolution_counts.items():
            print(f"   {res[0]}×{res[1]}: {count} images")
    
    return height, width


def downsample_image(input_path, output_path, downsample_factor):
    """Downsample an image by the given factor."""
    if downsample_factor == 1.0:
        # No downsampling needed, just copy
        shutil.copy2(input_path, output_path)
        return
    
    try:
        with Image.open(input_path) as img:
            new_width = int(img.width / downsample_factor)
            new_height = int(img.height / downsample_factor)
            
            # Use high-quality resampling
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(output_path, quality=95, optimize=True)
    except Exception as e:
        print(f"⚠️ Warning: Failed to process {input_path}: {e}")
        # Fallback: copy original
        shutil.copy2(input_path, output_path)


def calculate_crop_parameters(original_width, original_height, target_width, target_height):
    """
    Calculate crop parameters for aspect ratio preservation.
    
    Args:
        original_width, original_height: Original image dimensions
        target_width, target_height: Target image dimensions
        
    Returns:
        Tuple of (crop_x_offset, crop_y_offset, cropped_width, cropped_height)
    """
    target_aspect = target_width / target_height
    original_aspect = original_width / original_height
    
    if abs(target_aspect - original_aspect) < 0.001:
        # No crop needed
        return 0, 0, original_width, original_height
    
    if target_aspect > original_aspect:
        # Target is wider - crop height (keep full width)
        cropped_width = original_width
        cropped_height = int(original_width / target_aspect)
        crop_x_offset = 0
        crop_y_offset = (original_height - cropped_height) // 2
    else:
        # Target is taller - crop width (keep full height)
        cropped_width = int(original_height * target_aspect)
        cropped_height = original_height
        crop_x_offset = (original_width - cropped_width) // 2
        crop_y_offset = 0
    
    return crop_x_offset, crop_y_offset, cropped_width, cropped_height


def resize_image_to_target(input_path, output_path, target_width, target_height):
    """
    Resize an image to exact target dimensions with aspect ratio preservation (center crop).
    
    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        target_width: Target width in pixels
        target_height: Target height in pixels
    """
    try:
        with Image.open(input_path) as img:
            # Calculate target aspect ratio
            target_aspect = target_width / target_height
            original_aspect = img.width / img.height
            
            if abs(target_aspect - original_aspect) < 0.001:
                # Aspect ratios are already very close, just resize
                resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            else:
                # Need to crop to maintain aspect ratio
                if target_aspect > original_aspect:
                    # Target is wider - crop height (keep full width)
                    new_height = int(img.width / target_aspect)
                    crop_y = (img.height - new_height) // 2
                    crop_box = (0, crop_y, img.width, crop_y + new_height)
                else:
                    # Target is taller - crop width (keep full height)
                    new_width = int(img.height * target_aspect)
                    crop_x = (img.width - new_width) // 2
                    crop_box = (crop_x, 0, crop_x + new_width, img.height)
                
                # Crop then resize
                cropped_img = img.crop(crop_box)
                resized_img = cropped_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            resized_img.save(output_path, quality=95, optimize=True)
    except Exception as e:
        print(f"⚠️ Warning: Failed to process {input_path}: {e}")
        # Fallback: copy original
        shutil.copy2(input_path, output_path)


def copy_and_scale_sparse_reconstruction(input_colmap_path, output_path, original_width, original_height, new_width, new_height, selected_image_names=None):
    """
    Copy and scale COLMAP sparse reconstruction data for the new image resolution.
    
    Args:
        input_colmap_path: Path to input COLMAP dataset
        output_path: Path to output dataset
        original_width, original_height: Original image dimensions
        new_width, new_height: New image dimensions
        selected_image_names: Optional set of image names to include (None = all images)
    """
    input_colmap_path = Path(input_colmap_path)
    output_path = Path(output_path)
    
    input_sparse = input_colmap_path / "sparse"
    output_sparse = output_path / "sparse"
    
    if not input_sparse.exists():
        raise FileNotFoundError(f"Sparse reconstruction not found: {input_sparse}")
    
    print("📁 Copying and scaling COLMAP sparse reconstruction...")
    if output_sparse.exists():
        shutil.rmtree(output_sparse)
    
    print(f"📐 Processing camera parameters with crop-aware scaling:")
    print(f"   Original resolution: {original_width}×{original_height}")
    print(f"   Target resolution: {new_width}×{new_height}")
    
    # Process sparse/0 directory
    input_sparse_0 = input_sparse / "0"
    if input_sparse_0.exists():
        # Create output directory structure
        output_sparse_0 = output_sparse / "0"
        output_sparse_0.mkdir(parents=True, exist_ok=True)
        
        cameras_bin = input_sparse_0 / "cameras.bin"
        images_bin = input_sparse_0 / "images.bin"
        points3d_bin = input_sparse_0 / "points3D.bin"
        
        # Check if binary files exist
        if cameras_bin.exists() and images_bin.exists():
            try:
                print("📐 Reading, filtering, and scaling COLMAP data...")
                
                # Read original data from INPUT
                cameras = read_cameras_binary(cameras_bin)
                images = read_images_binary(images_bin)
                
                print(f"📊 Original data: {len(cameras)} cameras, {len(images)} images")
                
                # Filter data if image selection is specified
                if selected_image_names is not None:
                    cameras, images = filter_colmap_data_by_images(cameras, images, selected_image_names)
                
                # Scale camera parameters (with crop adjustment)
                scaled_cameras = scale_colmap_cameras(cameras, original_width, original_height, new_width, new_height)
                
                # Scale 2D point observations in images (with crop adjustment)
                scaled_images = scale_colmap_images(images, original_width, original_height, new_width, new_height)
                
                # Write scaled data to OUTPUT
                output_cameras_bin = output_sparse_0 / "cameras.bin"
                output_images_bin = output_sparse_0 / "images.bin"
                output_points3d_bin = output_sparse_0 / "points3D.bin"
                
                write_cameras_binary(scaled_cameras, output_cameras_bin)
                write_images_binary(scaled_images, output_images_bin)
                
                print(f"✅ Wrote {len(scaled_cameras)} cameras and {len(scaled_images)} images")
                
                # Copy points3D.bin if it exists (no scaling needed)
                if points3d_bin.exists():
                    shutil.copy2(points3d_bin, output_points3d_bin)
                    print("✅ Copied points3D.bin (no scaling needed)")
                else:
                    print("⚠️ Warning: points3D.bin not found, but this is optional")
                    
            except Exception as e:
                print(f"❌ Error processing binary format: {e}")
                print("🔄 Falling back to copy-only mode (no camera scaling)")
                print("   Note: Camera parameters will NOT be scaled for the new resolution!")
                print("   This may cause geometric inconsistencies in the dataset.")
                print()
                print("💡 Suggested fixes:")
                print("   1. Convert to text format first: colmap model_converter --input_path sparse/0 --output_path sparse/0 --output_type TXT")
                print("   2. Use original resolution (remove --force_resolution)")
                print(f"   3. Manually scale camera parameters for resolution change: {original_width}×{original_height} → {new_width}×{new_height}")
                
                # Fallback: just copy without modification
                if output_sparse.exists():
                    shutil.rmtree(output_sparse)
                shutil.copytree(input_sparse, output_sparse)
            
        else:
            # Try .txt files or just copy everything
            cameras_txt = input_sparse_0 / "cameras.txt"
            images_txt = input_sparse_0 / "images.txt"
            
            if cameras_txt.exists() and images_txt.exists():
                print("⚠️ Warning: Text format sparse reconstruction detected.")
                print("   Automatic scaling not implemented for .txt format.")
                print("   Consider converting to binary format first or manually scaling camera parameters.")
                print(f"   Resolution change needed: {original_width}×{original_height} → {new_width}×{new_height}")
            else:
                print("⚠️ No cameras.bin/txt or images.bin/txt found, copying as-is")
            
            # Copy everything as-is
            if output_sparse.exists():
                shutil.rmtree(output_sparse)
            shutil.copytree(input_sparse, output_sparse)
    
    else:
        print("⚠️ Warning: sparse/0 directory not found, copying entire sparse directory")
        if output_sparse.exists():
            shutil.rmtree(output_sparse)
        shutil.copytree(input_sparse, output_sparse)
    
    print(f"✅ Sparse reconstruction processed and saved to: {output_sparse}")


def copy_sparse_reconstruction(input_colmap_path, output_path):
    """
    Legacy function - copy sparse reconstruction without scaling.
    
    Args:
        input_colmap_path: Path to input COLMAP dataset
        output_path: Path to output dataset
    """
    input_colmap_path = Path(input_colmap_path)
    output_path = Path(output_path)
    
    # Copy sparse reconstruction directory
    input_sparse = input_colmap_path / "sparse"
    output_sparse = output_path / "sparse"
    
    if input_sparse.exists():
        print("📁 Copying COLMAP sparse reconstruction (no scaling)...")
        if output_sparse.exists():
            shutil.rmtree(output_sparse)
        shutil.copytree(input_sparse, output_sparse)
        print(f"✅ Sparse reconstruction copied to: {output_sparse}")
        
        # Validate copied sparse reconstruction
        sparse_0 = output_sparse / "0"
        if sparse_0.exists():
            required_files = ["cameras.bin", "images.bin", "points3D.bin"]
            missing_files = []
            for req_file in required_files:
                if not (sparse_0 / req_file).exists():
                    # Check for .txt versions
                    txt_file = req_file.replace(".bin", ".txt")
                    if not (sparse_0 / txt_file).exists():
                        missing_files.append(f"{req_file} or {txt_file}")
            
            if missing_files:
                print(f"⚠️ Warning: Missing sparse reconstruction files: {missing_files}")
            else:
                print("✅ Sparse reconstruction validation passed")
        else:
            print("⚠️ Warning: sparse/0 directory not found")
    else:
        raise FileNotFoundError(f"Sparse reconstruction not found: {input_sparse}")


def prepare_images_for_dataset(input_colmap_path, output_path, target_width, target_height, selected_images=None):
    """
    Copy and resize images from input COLMAP dataset to create standalone images folder.
    
    Args:
        input_colmap_path: Path to input COLMAP dataset
        output_path: Path to output dataset
        target_width: Target width for resized images
        target_height: Target height for resized images
        selected_images: Optional list of specific image files to process (None = all images)
        
    Returns:
        Path to the images directory in the output dataset
    """
    input_colmap_path = Path(input_colmap_path)
    output_path = Path(output_path)
    
    input_images_dir = input_colmap_path / "images"
    output_images_dir = output_path / "images"
    
    if not input_images_dir.exists():
        raise ValueError(f"Input images directory not found: {input_images_dir}")
    
    # Create output images directory
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Use provided selection or find all image files  
    if selected_images is not None:
        # Convert to full paths if they're just names
        image_files = []
        for img_path in selected_images:
            if isinstance(img_path, str) or hasattr(img_path, 'name'):
                # If it's just a name, find the full path
                img_name = img_path if isinstance(img_path, str) else img_path.name
                full_path = input_images_dir / img_name
                if full_path.exists():
                    image_files.append(full_path)
            else:
                image_files.append(img_path)
    else:
        # Find all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(input_images_dir.glob(f"*{ext}"))
    
    if not image_files:
        raise ValueError(f"No image files found in {input_images_dir}")
    
    print(f"📁 Copying and resizing {len(image_files)} images to {target_width}×{target_height}...")
    
    # Copy and resize images to exact target dimensions
    with tqdm(total=len(image_files), desc="📷 Processing images", unit="images") as pbar:
        for img_file in image_files:
            pbar.set_postfix_str(img_file.name)
            output_img_path = output_images_dir / img_file.name
            resize_image_to_target(img_file, output_img_path, target_width, target_height)
            pbar.update(1)
    
    print(f"✅ Images prepared in: {output_images_dir}")
    return output_images_dir


def prepare_colmap_images_for_extraction(images_dir, temp_dir, downsample_factor):
    """
    Prepare images for G-buffer extraction by copying to temp directory.
    This is needed for the cosmos renderer which expects a specific structure.
    """
    images_dir = Path(images_dir)
    temp_dir = Path(temp_dir)
    
    # Create temp directory for extraction
    temp_images_dir = temp_dir / "images"
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images to temp directory (they're already resized)
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    print(f"📂 Preparing {len(image_files)} images for G-buffer extraction...")
    
    with tqdm(total=len(image_files), desc="📂 Copying for extraction", unit="images") as pbar:
        for img_file in image_files:
            pbar.set_postfix_str(img_file.name)
            output_path = temp_images_dir / img_file.name
            shutil.copy2(img_file, output_path)
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
        "--group_mode", "webdataset",  # This groups images for batch processing
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
    
    print(f"🚀 Running G-buffer extraction for {len(input_images)} images:")
    print(f"   📁 Input directory: {images_dir}")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   🎯 Target resolution: {width}×{height}")
    print(f"   🔢 Frames per image: {num_frames}")
    print(f"   📦 Group mode: webdataset (for batch processing)")
    print()
    print(f"⚠️  NOTE: The cosmos renderer will process images individually")
    print(f"     Total inference calls: {len(input_images)} images × 5 G-buffer types = {len(input_images) * 5}")
    print()
    print(f"Command: CUDA_HOME=$CONDA_PREFIX PYTHONPATH={cosmos_path}")
    print(f"         {' '.join(cmd)}")
    print()
    print(f"⏱️  Starting cosmos1-diffusion-renderer...")
    
    # Run the inverse renderer - cosmos will handle all images in the directory
    # but internally processes each image individually with multiple G-buffer passes
    result = subprocess.run(cmd, cwd=cosmos_path, env=env, capture_output=False, text=True)
    
    print(f"✅ G-buffer extraction completed for all {len(input_images)} images!")
    
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


def organize_final_dataset(gbuffer_dir, output_path, input_colmap_path, height, width, downsample_factor, max_images=None):
    """
    Organize the extracted G-buffers and create final dataset metadata.
    
    Args:
        gbuffer_dir: Source G-buffer directory
        output_path: Final output dataset directory
        input_colmap_path: Original input COLMAP dataset path
        height, width: Processing resolution
        downsample_factor: Downsampling factor used
        max_images: Optional maximum number of images processed (for testing)
    """
    gbuffer_dir = Path(gbuffer_dir)
    output_path = Path(output_path)
    input_colmap_path = Path(input_colmap_path)
    
    # Create final output directory
    final_gbuffer_dir = output_path / "gbuffers"
    final_gbuffer_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Organizing G-buffers in final dataset...")
    
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
    
    # Count images in the dataset
    images_dir = output_path / "images"
    image_files = []
    if images_dir.exists():
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(images_dir.glob(f"*{ext}"))
    
    # Create comprehensive dataset metadata
    metadata = {
        "type": "gbuffers_colmap",
        "description": "Complete COLMAP G-buffer dataset with images, sparse reconstruction, and G-buffers",
        "source_info": {
            "original_colmap_path": str(input_colmap_path),
            "creation_tool": "create_colmap_gbuffers.py",
            "cosmos_renderer": "cosmos1-diffusion-renderer"
        },
        "dataset_structure": {
            "images/": "Resized images matching G-buffer extraction resolution",
            "sparse/": "COLMAP sparse reconstruction (cameras, images, points3D)",
            "gbuffers/": "Extracted G-buffer files organized by image name",
            "dataset_info.json": "This metadata file"
        },
        "statistics": {
            "num_images": len(image_files),
            "num_gbuffer_files": len(organized_files),
            "image_resolution": {
                "width": width,
                "height": height,
                "downsample_factor": downsample_factor
            },
            "dataset_limitation": {
                "is_limited_for_testing": max_images is not None,
                "max_images_processed": max_images
            }
        },
        "gbuffer_info": {
            "types": ["basecolor", "depth", "metallic", "normal", "roughness"],
            "format": "Individual files per type per image",
            "naming_pattern": "gbuffers/{image_name}/{frame}.{pass}.{type}.jpg",
            "extraction_resolution": {
                "width": width,
                "height": height
            }
        },
        "usage": {
            "dataset_type": "gbuffers_colmap",
            "compatible_with": "ColmapGBufferDataset class",
            "example_config": {
                "type": "gbuffers_colmap",
                "path": str(output_path),
                "gbuffer_config": {
                    "load_gbuffers": True,
                    "gbuffer_format": "concat"
                }
            }
        }
    }
    
    # Save metadata
    metadata_path = output_path / "dataset_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Dataset metadata saved: {metadata_path}")
    print(f"📊 Dataset statistics:")
    print(f"   - Images: {len(image_files)}")
    print(f"   - G-buffer files: {len(organized_files)}")
    print(f"   - Resolution: {width}×{height}")
    print(f"   - Downsample factor: {downsample_factor}")


def main():
    """Main function to create a complete COLMAP G-buffer dataset."""
    args = parse_arguments()
    
    # Validate paths
    input_colmap_path = Path(args.input_colmap_path)
    output_path = Path(args.output_path)
    
    if not input_colmap_path.exists():
        raise FileNotFoundError(f"Input COLMAP dataset not found: {input_colmap_path}")
    
    print(f"🎯 Creating COLMAP G-buffer dataset")
    print(f"📁 Input COLMAP dataset: {input_colmap_path}")
    print(f"📁 Output dataset: {output_path}")
    print()
    
    # Setup cosmos environment
    cosmos_path = setup_cosmos_environment(args.cosmos_path)
    
    # Detect original resolution from input dataset (needed for camera scaling)
    original_height, original_width = detect_image_resolution(input_colmap_path)
    
    # Determine final resolution for G-buffer extraction
    if args.force_resolution:
        if args.height is None or args.width is None:
            raise ValueError("Both --height and --width must be specified with --force_resolution")
        height = args.height
        width = args.width
        # Apply downsampling to manually specified resolution
        if args.downsample_factor != 1.0:
            height = int(height / args.downsample_factor)
            width = int(width / args.downsample_factor)
            print(f"Applying downsample factor {args.downsample_factor} to specified resolution")
    else:
        # Use detected resolution
        height = original_height
        width = original_width
        # Apply downsampling to detected resolution
        if args.downsample_factor != 1.0:
            height = int(height / args.downsample_factor)
            width = int(width / args.downsample_factor)
            print(f"Applying downsample factor {args.downsample_factor} to detected resolution")
    
    # Determine which images to process
    input_images_dir = input_colmap_path / "images"
    selected_images = get_limited_image_list(input_images_dir, args.max_images)
    selected_image_names = {img.name for img in selected_images}
    
    print(f"Dataset creation settings:")
    print(f"  - Original image resolution: {original_width}×{original_height}")
    print(f"  - Final image resolution: {width}×{height}")
    print(f"  - G-buffer extraction resolution: {width}×{height}")
    print(f"  - Downsample factor: {args.downsample_factor}")
    print(f"  - Number of frames: {args.num_frames}")
    if args.max_images is not None:
        print(f"  - Image limit (for testing): {args.max_images}")
    if args.height is None or args.width is None:
        print(f"    (resolution auto-detected from input images)")
    if original_width != width or original_height != height:
        print(f"  📐 Camera parameters will be scaled accordingly")
    print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        try:
            print("=== Creating Complete COLMAP G-Buffer Dataset ===")
            
            # Step 1: Copy and scale sparse reconstruction
            print("\n1. Copying and scaling COLMAP sparse reconstruction...")
            copy_and_scale_sparse_reconstruction(
                input_colmap_path, output_path, 
                original_width, original_height, width, height, selected_image_names
            )
            
            # Step 2: Prepare images for the dataset
            print("\n2. Preparing images for standalone dataset...")
            dataset_images_dir = prepare_images_for_dataset(
                input_colmap_path, output_path, width, height, selected_images
            )
            
            # Step 3: Prepare images for G-buffer extraction
            print("\n3. Preparing images for G-buffer extraction...")
            extraction_images_dir = prepare_colmap_images_for_extraction(
                dataset_images_dir, temp_dir, 1.0  # No additional downsampling
            )
            
            # Step 4: Extract G-buffers
            print("\n4. Extracting G-buffers...")
            gbuffer_output_dir = temp_dir / "gbuffer_output"
            gbuffer_dir = extract_gbuffers(
                extraction_images_dir, cosmos_path, args.checkpoint_dir, 
                gbuffer_output_dir, height, width, args.num_frames
            )
            
            # Step 5: Organize final dataset
            print("\n5. Finalizing dataset structure...")
            # Calculate the effective downsample factor that was applied
            effective_downsample_factor = original_width / width  # or original_height / height
            organize_final_dataset(
                gbuffer_dir, output_path, input_colmap_path, 
                height, width, effective_downsample_factor, args.max_images
            )
            
            print(f"\n=== COLMAP G-Buffer Dataset Creation Complete ===")
            print(f"📁 Complete dataset created at: {output_path}")
            print(f"")
            print(f"Dataset structure:")
            print(f"  {output_path}/")
            print(f"  ├── images/              # Resized images ({width}×{height})")
            print(f"  ├── sparse/             # COLMAP reconstruction")
            print(f"  ├── gbuffers/           # Extracted G-buffers")
            print(f"  └── dataset_info.json   # Dataset metadata")
            print(f"")
            print(f"Usage with 3DGRUT:")
            print(f"  dataset:")
            print(f"    type: colmap_gbuffers")
            print(f"    path: {output_path}")
            
        except Exception as e:
            print(f"\nError during processing: {e}")
            raise


if __name__ == "__main__":
    main()