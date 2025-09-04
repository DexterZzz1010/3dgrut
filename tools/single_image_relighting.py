#!/usr/bin/env python3
"""
Single Image Relighting Tool

This script takes a single image and a folder of HDRI images, extracts G-buffers
using cosmos1-renderer, and generates random relit images using randomly selected
and rotated HDRIs. The results are displayed in a visualization showing:
- Original image
- G-buffer components (albedo, depth, normals)
- For each relit image: the rotated HDRI and corresponding relit result

Usage:
    # Basic usage
    conda activate cosmos-predict1
    python single_image_relighting.py \
        --input_image /path/to/image.jpg \
        --hdri_folder /path/to/hdri/files \
        --num_relit 3

    # With custom cosmos path
    python single_image_relighting.py \
        --input_image /path/to/image.jpg \
        --hdri_folder /path/to/hdri/files \
        --num_relit 5 \
        --cosmos_path /path/to/cosmos1-diffusion-renderer
"""

import argparse
import os
import shutil
import subprocess
import sys
import random
import tempfile
import time
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


def resize_image_to_resolution(input_path, output_path, target_width, target_height):
    """
    Resize an image to target resolution with high-quality resampling.
    
    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        target_width: Target width in pixels
        target_height: Target height in pixels
    """
    # Load image
    img = Image.open(input_path)
    original_width, original_height = img.size
    
    # Check if resizing is needed
    if original_width == target_width and original_height == target_height:
        # No resizing needed, just copy
        shutil.copy2(input_path, output_path)
        return original_width, original_height
    
    # Resize using high-quality resampling
    resized_img = img.resize((target_width, target_height), Image.LANCZOS)
    
    # Save with same format and quality
    if input_path.suffix.lower() in ['.jpg', '.jpeg']:
        resized_img.save(output_path, 'JPEG', quality=95)
    else:
        resized_img.save(output_path)
    
    print(f"📐 Resized image: {original_width}×{original_height} → {target_width}×{target_height}")
    return original_width, original_height


def detect_and_validate_image_resolution(image_path, expected_width=1280, expected_height=704):
    """
    Detect image resolution and check against cosmos1-diffusion-renderer expected resolution.
    
    Args:
        image_path: Path to input image
        expected_width: Expected width for cosmos1 (default: 1280)
        expected_height: Expected height for cosmos1 (default: 704)
        
    Returns:
        tuple: (original_width, original_height, needs_resize, scale_factor)
    """
    image_path = Path(image_path)
    
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size
    except Exception as e:
        raise ValueError(f"Could not read image {image_path}: {e}")
    
    print(f"📏 Input image resolution: {original_width}×{original_height}")
    
    # Check if resolution matches expected
    needs_resize = (original_width != expected_width or original_height != expected_height)
    
    if needs_resize:
        # Calculate scale factor needed
        width_ratio = original_width / expected_width
        height_ratio = original_height / expected_height
        scale_factor = max(width_ratio, height_ratio)
        
        print(f"⚠️  Resolution mismatch: got {original_width}×{original_height}, expected {expected_width}×{expected_height}")
        print(f"🔧 Will resize to match cosmos1-diffusion-renderer expected resolution")
        if scale_factor > 1.0:
            print(f"💡 Original image is {scale_factor:.2f}x larger than expected")
        else:
            print(f"💡 Original image is {1/scale_factor:.2f}x smaller than expected")
    else:
        print(f"✅ Image resolution matches cosmos1 expected resolution")
        scale_factor = 1.0
    
    return original_width, original_height, needs_resize, scale_factor


class PerformanceTracker:
    """Track performance metrics for the relighting process."""
    
    def __init__(self):
        self.start_time = time.time()
        self.step_times = {}
        self.step_start = None
        self.current_step = None
        
    def start_step(self, step_name):
        """Start timing a processing step."""
        if self.current_step is not None:
            self.end_step()
        
        self.current_step = step_name
        self.step_start = time.time()
        print(f"⏱️  Starting: {step_name}")
        
    def end_step(self):
        """End timing the current step."""
        if self.current_step is not None and self.step_start is not None:
            duration = time.time() - self.step_start
            self.step_times[self.current_step] = duration
            print(f"✅ Completed: {self.current_step} ({duration:.2f}s)")
            self.current_step = None
            self.step_start = None
        else:
            print("⚠️  Warning: end_step() called but no current step active")
    
    def get_total_time(self):
        """Get total elapsed time since tracker creation."""
        return time.time() - self.start_time
    
    def print_summary(self):
        """Print a comprehensive performance summary."""
        total_time = self.get_total_time()
        
        print("\n" + "="*60)
        print("🚀 PERFORMANCE SUMMARY")
        print("="*60)
        
        if self.step_times:
            print("\n📊 Step-by-step timing:")
            longest_name = max(len(name) for name in self.step_times.keys()) if self.step_times else 0
            
            for step_name, duration in self.step_times.items():
                percentage = (duration / total_time) * 100 if total_time > 0 else 0
                bar_length = max(0, min(50, int(percentage / 2)))  # Scale bar to fit, clamp to 0-50
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"  {step_name:<{longest_name}} │ {duration:6.2f}s │ {percentage:5.1f}% │ {bar}")
            
            print(f"\n⚡ Performance insights:")
            if len(self.step_times) > 0:
                slowest_step = max(self.step_times.items(), key=lambda x: x[1])
                fastest_step = min(self.step_times.items(), key=lambda x: x[1])
                print(f"  • Slowest step: {slowest_step[0]} ({slowest_step[1]:.2f}s)")
                print(f"  • Fastest step: {fastest_step[0]} ({fastest_step[1]:.2f}s)")
                print(f"  • Average step time: {sum(self.step_times.values()) / len(self.step_times):.2f}s")
        else:
            print("\n⚠️  No performance data recorded")
            print("   This might happen if the script failed early or performance tracking wasn't enabled")
        
        print(f"\n🏁 Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        # Add some performance tips based on timing
        if self.step_times:
            gbuffer_time = self.step_times.get("G-buffer extraction", 0)
            relight_time = sum(v for k, v in self.step_times.items() if "Relighting" in k)
            
            print(f"\n💡 Performance tips:")
            if gbuffer_time > relight_time and relight_time > 0:
                print(f"  • G-buffer extraction took {gbuffer_time:.1f}s vs {relight_time:.1f}s for relighting")
                print(f"  • Consider reusing G-buffers for multiple relighting sessions")
            
            relight_steps = [k for k in self.step_times.keys() if "Relighting" in k]
            if relight_time > 0 and len(relight_steps) > 0:
                avg_relight = relight_time / len(relight_steps)
                print(f"  • Average relighting time: {avg_relight:.2f}s per HDRI")
                
            if total_time > 300:  # > 5 minutes
                print(f"  • For faster processing, consider reducing image resolution")
                print(f"  • GPU acceleration is recommended for large batches")
        
        print("="*60)


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


def find_hdri_files(hdri_folder):
    """Find all valid HDRI (.hdr/.exr) files in the specified folder."""
    hdri_folder = Path(hdri_folder)
    all_hdri_files = []
    
    for ext in ['.hdr', '.HDR', '.exr', '.EXR']:
        all_hdri_files.extend(hdri_folder.glob(f"*{ext}"))
    
    if not all_hdri_files:
        raise ValueError(f"No HDRI files found in {hdri_folder}")
    
    # Validate HDRI files and exclude corrupted ones
    valid_hdri_files = []
    corrupted_files = []
    
    print(f"Validating {len(all_hdri_files)} HDRI files...")
    for hdri_file in all_hdri_files:
        if validate_hdr_file(hdri_file):
            valid_hdri_files.append(hdri_file)
        else:
            corrupted_files.append(hdri_file)
    
    if corrupted_files:
        print(f"⚠️  Skipping {len(corrupted_files)} corrupted HDRI files:")
        for f in corrupted_files[:3]:  # Show first 3
            print(f"   ❌ {f.name}")
        if len(corrupted_files) > 3:
            print(f"   ... and {len(corrupted_files) - 3} more")
    
    if not valid_hdri_files:
        raise ValueError(f"No valid HDRI files found in {hdri_folder}")
    
    print(f"✅ Found {len(valid_hdri_files)} valid HDRI files")
    return sorted(valid_hdri_files)


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
    
    return image


def extract_gbuffers(input_image, cosmos_path, checkpoint_dir, output_dir, height=704, width=1280, perf_tracker=None, auto_resize=True):
    """
    Extract G-buffers from a single image using cosmos1-diffusion-renderer.
    
    Args:
        input_image: Path to input image
        cosmos_path: Path to cosmos1-diffusion-renderer
        checkpoint_dir: Directory containing model checkpoints
        output_dir: Directory to save G-buffer outputs
        height, width: Output dimensions
        perf_tracker: Optional PerformanceTracker instance
        auto_resize: Whether to automatically resize image to target resolution
        
    Returns:
        Path to the directory containing extracted G-buffers
    """
    input_image = Path(input_image)
    cosmos_path = Path(cosmos_path)
    output_dir = Path(output_dir)
    
    print(f"🔍 Processing image: {input_image.name}")
    if not input_image.exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")
    
    # Create temporary directory for single image
    temp_images_dir = output_dir.parent / "temp_single_image"
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_image_path = temp_images_dir / input_image.name
    
    if auto_resize:
        # Detect and validate image resolution
        original_width, original_height, needs_resize, scale_factor = detect_and_validate_image_resolution(
            input_image, width, height
        )
        
        # Resize image to expected resolution if needed
        if needs_resize:
            resize_start = time.time()
            print(f"🔧 Resizing image to cosmos1-diffusion-renderer expected resolution...")
            resize_image_to_resolution(input_image, temp_image_path, width, height)
            resize_time = time.time() - resize_start
            if perf_tracker:
                perf_tracker.step_times["Image resizing"] = resize_time
        else:
            # Image already at correct resolution, just copy
            shutil.copy2(input_image, temp_image_path)
    else:
        # Skip resizing, just copy original
        print(f"⚠️  Skipping image resize (keeping original resolution)")
        print(f"💡 This may cause issues if image resolution doesn't match cosmos1 expected {width}×{height}")
        shutil.copy2(input_image, temp_image_path)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare inverse renderer command
    inverse_script = cosmos_path / "cosmos_predict1/diffusion/inference/inference_inverse_renderer.py"
    
    cmd = [
        "python", str(inverse_script),
        "--checkpoint_dir", str(cosmos_path / checkpoint_dir),
        "--diffusion_transformer_dir", "Diffusion_Renderer_Inverse_Cosmos_7B",
        "--dataset_path", str(temp_images_dir),
        "--num_video_frames", "1",
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
    
    print(f"Running G-buffer extraction...")
    print(f"CUDA_HOME=$CONDA_PREFIX PYTHONPATH={cosmos_path} {' '.join(cmd)}")
    
    # Run the inverse renderer with timing
    cosmos_start = time.time()
    with tqdm(total=1, desc="🔍 Extracting G-buffers", unit="image") as pbar:
        result = subprocess.run(cmd, cwd=cosmos_path, env=env, capture_output=True, text=True)
        pbar.update(1)
    cosmos_time = time.time() - cosmos_start
    
    if perf_tracker:
        perf_tracker.step_times[f"Cosmos inference"] = cosmos_time
    
    if result.returncode != 0:
        print(f"❌ Error running inverse renderer:")
        print(f"Command: {' '.join(cmd)}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"💡 Troubleshooting tips:")
        print(f"   - Make sure you activated the 'cosmos-predict1' conda environment")
        print(f"   - Check that the checkpoint directory exists: {cosmos_path / checkpoint_dir}")
        print(f"   - Verify that the input image is valid: {input_image}")
        raise RuntimeError("G-buffer extraction failed")
    
    print("✅ G-buffer extraction completed successfully")
    
    # Clean up temp directory
    if temp_images_dir.exists():
        shutil.rmtree(temp_images_dir)
    
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


def relight_with_hdri(gbuffer_dir, hdri_file, cosmos_path, checkpoint_dir, output_dir, height=704, width=1280, perf_tracker=None, relit_index=0):
    """
    Relight G-buffers with a custom HDRI file.
    
    Args:
        gbuffer_dir: Directory containing G-buffer frames
        hdri_file: Path to HDRI file
        cosmos_path: Path to cosmos1-diffusion-renderer
        checkpoint_dir: Directory containing model checkpoints
        output_dir: Directory to save relit outputs
        height, width: Output dimensions
        perf_tracker: Optional PerformanceTracker instance
        relit_index: Index of this relighting for tracking
    """
    print(f"💡 Relighting with HDRI: {hdri_file.name}")
    
    # Validate G-buffer directory structure
    gbuffer_path = Path(gbuffer_dir)
    if not gbuffer_path.exists():
        raise FileNotFoundError(f"G-buffer directory not found: {gbuffer_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary modified forward renderer script
    forward_script = cosmos_path / "cosmos_predict1/diffusion/inference/inference_forward_renderer.py"
    temp_forward_script = cosmos_path / f"temp_inference_forward_renderer_single.py"
    
    # Read the original script
    with open(forward_script, 'r') as f:
        script_content = f.read()
    
    # Replace the ENV_LIGHT_PATH_LIST with our custom HDRI
    hdri_line = f'ENV_LIGHT_PATH_LIST = ["{str(hdri_file)}"]'
    
    # Find and replace the ENV_LIGHT_PATH_LIST definition
    lines = script_content.split('\n')
    new_lines = []
    in_env_list = False
    
    for line in lines:
        if line.strip().startswith('ENV_LIGHT_PATH_LIST = ['):
            new_lines.append(hdri_line)
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
        # Run the forward renderer with custom HDRI
        cmd = [
            "python", str(temp_forward_script),
            "--checkpoint_dir", str(cosmos_path / checkpoint_dir),
            "--diffusion_transformer_dir", "Diffusion_Renderer_Forward_Cosmos_7B",
            "--dataset_path", str(gbuffer_dir),
            "--num_video_frames", "1",
            "--envlight_ind", "0",  # Use index 0 since we only have one HDRI
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
        
        # Run the forward renderer with timing
        relight_start = time.time()
        with tqdm(total=1, desc=f"💡 Relighting", unit="HDRI") as pbar:
            result = subprocess.run(cmd, cwd=cosmos_path, env=env, capture_output=True, text=True)
            pbar.update(1)
        relight_time = time.time() - relight_start
        
        if perf_tracker:
            perf_tracker.step_times[f"Relighting #{relit_index+1}"] = relight_time
        
        if result.returncode != 0:
            print(f"❌ Error running forward renderer:")
            print(f"Command: {' '.join(cmd)}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            print(f"💡 Troubleshooting tips:")
            print(f"   - Make sure you activated the 'cosmos-predict1' conda environment")
            print(f"   - Check that G-buffer directory contains valid files: {gbuffer_dir}")
            print(f"   - Verify that the HDRI file exists and is valid: {hdri_file}")
            print(f"   - Check that the checkpoint directory exists: {cosmos_path / checkpoint_dir}")
            raise RuntimeError(f"Relighting with {hdri_file.name} failed")
        
        print(f"✅ Relighting completed successfully")
        
    finally:
        # Clean up temporary script
        if temp_forward_script.exists():
            temp_forward_script.unlink()


def load_gbuffer_components(gbuffer_dir):
    """
    Load G-buffer components (albedo, depth, normals) for visualization.
    
    Args:
        gbuffer_dir: Directory containing G-buffer files
        
    Returns:
        dict: Dictionary with 'albedo', 'depth', 'normals' components
    """
    gbuffer_dir = Path(gbuffer_dir)
    components = {}
    
    print(f"🔍 Looking for G-buffer files in: {gbuffer_dir}")
    print(f"💡 Expected patterns:")
    print(f"   • Albedo: files containing 'albedo', 'basecolor', or 'diffuse'")
    print(f"   • Depth: files containing 'depth', 'z', or 'distance'") 
    print(f"   • Normals: files containing 'normal', 'norm', or 'n_'")
    print(f"   • Cosmos1 format: 0000.0000.basecolor.jpg, 0000.0000.depth.jpg, etc.")
    print()
    
    # Check if directory exists
    if not gbuffer_dir.exists():
        print(f"❌ G-buffer directory does not exist: {gbuffer_dir}")
        return components
    
    # List all files for debugging
    all_files = list(gbuffer_dir.glob("*"))
    print(f"📂 Found {len(all_files)} items in G-buffer directory:")
    for file in all_files:
        if file.is_file():
            print(f"   📄 {file.name} ({file.stat().st_size} bytes)")
            # Show what this file would be classified as
            name_lower = file.name.lower()
            classification = "unknown"
            if 'basecolor' in name_lower or 'albedo' in name_lower or 'diffuse' in name_lower:
                classification = "albedo/basecolor"
            elif 'depth' in name_lower or 'z' in name_lower or 'distance' in name_lower:
                classification = "depth"
            elif 'normal' in name_lower or 'norm' in name_lower or 'n_' in name_lower:
                classification = "normals"
            elif 'metallic' in name_lower:
                classification = "metallic"
            elif 'roughness' in name_lower:
                classification = "roughness"
            print(f"      → Would be classified as: {classification}")
        else:
            print(f"   📁 {file.name}/ (directory)")
    
    # Also show subdirectory contents
    for subdir_path in gbuffer_dir.glob("*"):
        if subdir_path.is_dir():
            subdir_files = list(subdir_path.glob("*"))
            if subdir_files:
                print(f"   📁 {subdir_path.name}/ contents:")
                for file in subdir_files[:10]:  # Show first 10 files
                    if file.is_file():
                        name_lower = file.name.lower()
                        classification = "unknown"
                        if 'basecolor' in name_lower or 'albedo' in name_lower or 'diffuse' in name_lower:
                            classification = "albedo/basecolor"
                        elif 'depth' in name_lower or 'z' in name_lower or 'distance' in name_lower:
                            classification = "depth"
                        elif 'normal' in name_lower or 'norm' in name_lower or 'n_' in name_lower:
                            classification = "normals"
                        elif 'metallic' in name_lower:
                            classification = "metallic"
                        elif 'roughness' in name_lower:
                            classification = "roughness"
                        print(f"      📄 {file.name} ({file.stat().st_size} bytes) → {classification}")
                if len(subdir_files) > 10:
                    print(f"      ... and {len(subdir_files) - 10} more files")
    
    # Look for G-buffer component files
    # cosmos1-diffusion-renderer outputs files like: 0000.0000.basecolor.jpg, 0000.0000.depth.jpg, etc.
    all_files = list(gbuffer_dir.glob("*"))
    
    def load_gbuffer_image(file_path, component_name):
        """Helper function to load a G-buffer image."""
        try:
            print(f"📷 Loading {component_name} from: {file_path.name}")
            if file_path.suffix.lower() == '.exr':
                os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
                img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.array(Image.open(file_path))
            
            if img is not None:
                print(f"✅ Loaded {component_name}: shape {img.shape}")
                return img
            else:
                print(f"⚠️  Failed to load image from {file_path.name} - image is None")
                return None
        except Exception as e:
            print(f"⚠️  Failed to load {component_name} from {file_path.name}: {e}")
            return None
    
    for gbuffer_file in all_files:
        if gbuffer_file.is_file():
            name_lower = gbuffer_file.name.lower()
            
            # cosmos1 patterns: 0000.0000.basecolor.jpg, 0000.0000.depth.jpg, 0000.0000.normal.jpg
            # Also check for standalone patterns: basecolor.jpg, depth.jpg, normal.jpg
            # And alternative patterns: albedo.exr, diffuse.png, etc.
            
            # Check for albedo/basecolor/diffuse
            if (('basecolor' in name_lower or 'albedo' in name_lower or 'diffuse' in name_lower) and 
                'albedo' not in components):
                img = load_gbuffer_image(gbuffer_file, 'albedo')
                if img is not None:
                    components['albedo'] = img
                    
            # Check for depth (various patterns)
            elif (('depth' in name_lower or 'z' in name_lower or 'distance' in name_lower) and 
                  'depth' not in components):
                img = load_gbuffer_image(gbuffer_file, 'depth')
                if img is not None:
                    components['depth'] = img
                    
            # Check for normals (various patterns)
            elif (('normal' in name_lower or 'norm' in name_lower or 'n_' in name_lower) and 
                  'normals' not in components):
                img = load_gbuffer_image(gbuffer_file, 'normals')
                if img is not None:
                    components['normals'] = img
                    
            # Check for metallic (bonus)
            elif ('metallic' in name_lower and 'metallic' not in components):
                img = load_gbuffer_image(gbuffer_file, 'metallic')
                if img is not None:
                    components['metallic'] = img
                    
            # Check for roughness (bonus)
            elif ('roughness' in name_lower and 'roughness' not in components):
                img = load_gbuffer_image(gbuffer_file, 'roughness')
                if img is not None:
                    components['roughness'] = img
    
    # Check for subdirectories (cosmos1 might put files in subdirs)
    for subdir in gbuffer_dir.glob("*"):
        if subdir.is_dir():
            print(f"📁 Checking subdirectory: {subdir.name}")
            subdir_files = list(subdir.glob("*"))
            print(f"   Found {len(subdir_files)} files in subdirectory")
            
            for gbuffer_file in subdir_files:
                if gbuffer_file.is_file():
                    name_lower = gbuffer_file.name.lower()
                    print(f"   📄 Checking: {gbuffer_file.name}")
                    
                    # Use same pattern matching as main directory
                    if (('basecolor' in name_lower or 'albedo' in name_lower or 'diffuse' in name_lower) and 
                        'albedo' not in components):
                        img = load_gbuffer_image(gbuffer_file, 'albedo (from subdir)')
                        if img is not None:
                            components['albedo'] = img
                            
                    elif (('depth' in name_lower or 'z' in name_lower or 'distance' in name_lower) and 
                          'depth' not in components):
                        img = load_gbuffer_image(gbuffer_file, 'depth (from subdir)')
                        if img is not None:
                            components['depth'] = img
                            
                    elif (('normal' in name_lower or 'norm' in name_lower or 'n_' in name_lower) and 
                          'normals' not in components):
                        img = load_gbuffer_image(gbuffer_file, 'normals (from subdir)')
                        if img is not None:
                            components['normals'] = img
                            
                    elif ('metallic' in name_lower and 'metallic' not in components):
                        img = load_gbuffer_image(gbuffer_file, 'metallic (from subdir)')
                        if img is not None:
                            components['metallic'] = img
                            
                    elif ('roughness' in name_lower and 'roughness' not in components):
                        img = load_gbuffer_image(gbuffer_file, 'roughness (from subdir)')
                        if img is not None:
                            components['roughness'] = img
    
    # If we didn't find specific components, try to load files as generic components
    missing_components = []
    if 'albedo' not in components:
        missing_components.append('albedo')
    if 'depth' not in components:
        missing_components.append('depth')
    if 'normals' not in components:
        missing_components.append('normals')
    
    if missing_components:
        print(f"⚠️  Missing G-buffer components: {missing_components}")
        print(f"🔍 Attempting to load generic files as fallback...")
        gbuffer_files = []
        
        # Look in main directory for any image files
        for f in gbuffer_dir.glob("*"):
            if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.exr']:
                # Skip files we already loaded or that match known patterns
                name_lower = f.name.lower()
                already_loaded = False
                
                # Check if this file matches any known G-buffer patterns
                if ('basecolor' in name_lower or 'albedo' in name_lower or 'diffuse' in name_lower or
                    'depth' in name_lower or 'z' in name_lower or 'distance' in name_lower or
                    'normal' in name_lower or 'norm' in name_lower or 'n_' in name_lower or
                    'metallic' in name_lower or 'roughness' in name_lower):
                    # This file should have been processed in the specific pattern matching
                    continue
                
                # Check if we already loaded this file
                for comp in components.keys():
                    if comp.lower() in name_lower:
                        already_loaded = True
                        break
                        
                if not already_loaded:
                    gbuffer_files.append(f)
        
        # Look in subdirectories if main directory doesn't have enough files
        if len(gbuffer_files) < len(missing_components):
            for subdir in gbuffer_dir.glob("*"):
                if subdir.is_dir():
                    for f in subdir.glob("*"):
                        if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.exr']:
                            # Skip files we already loaded or that match known patterns
                            name_lower = f.name.lower()
                            already_loaded = False
                            
                            # Check if this file matches any known G-buffer patterns
                            if ('basecolor' in name_lower or 'albedo' in name_lower or 'diffuse' in name_lower or
                                'depth' in name_lower or 'z' in name_lower or 'distance' in name_lower or
                                'normal' in name_lower or 'norm' in name_lower or 'n_' in name_lower or
                                'metallic' in name_lower or 'roughness' in name_lower):
                                # This file should have been processed in the specific pattern matching
                                continue
                            
                            # Check if we already loaded this file
                            for comp in components.keys():
                                if comp.lower() in name_lower:
                                    already_loaded = True
                                    break
                                    
                            if not already_loaded and f not in gbuffer_files:
                                gbuffer_files.append(f)
        
        print(f"🔍 Found {len(gbuffer_files)} potential generic files for fallback loading")
        
        # Sort by filename for consistent loading order
        gbuffer_files = sorted(gbuffer_files, key=lambda x: x.name)
        
        # Assign files to missing components
        for i, component_name in enumerate(missing_components):
            if i < len(gbuffer_files):
                gbuffer_file = gbuffer_files[i]
                img = load_gbuffer_image(gbuffer_file, f'{component_name} (generic fallback)')
                if img is not None:
                    components[component_name] = img
                    print(f"✅ Assigned {gbuffer_file.name} as {component_name}")
                else:
                    print(f"⚠️  Could not load {gbuffer_file.name} as {component_name}")
            else:
                print(f"❌ No more files available to assign as {component_name}")
    
    # Final check for files without extensions (cosmos1 sometimes outputs these)
    if len(components) < 2:  # If we still don't have enough components
        print(f"🔍 Checking for files without extensions (cosmos1 format)...")
        for f in gbuffer_dir.glob("*"):
            if f.is_file() and not f.suffix:  # File without extension
                file_size = f.stat().st_size
                if file_size > 1000:  # Non-empty file
                    print(f"   📄 Found file without extension: {f.name} ({file_size} bytes)")
                    
                    # Try to load it as a generic component if we're missing some
                    if len(components) < 3:
                        missing = ['albedo', 'depth', 'normals']
                        for comp in list(components.keys()):
                            if comp in missing:
                                missing.remove(comp)
                        
                        if missing:
                            comp_name = missing[0]
                            img = load_gbuffer_image(f, f'{comp_name} (no extension)')
                            if img is not None:
                                components[comp_name] = img
                                print(f"✅ Assigned {f.name} (no ext) as {comp_name}")
        
        # Also check subdirectories for files without extensions
        for subdir in gbuffer_dir.glob("*"):
            if subdir.is_dir():
                for f in subdir.glob("*"):
                    if f.is_file() and not f.suffix:  # File without extension
                        file_size = f.stat().st_size
                        if file_size > 1000:  # Non-empty file
                            print(f"   📄 Found file without extension in subdir: {f.name} ({file_size} bytes)")
                            
                            # Try to load it as a generic component if we're missing some
                            if len(components) < 3:
                                missing = ['albedo', 'depth', 'normals']
                                for comp in list(components.keys()):
                                    if comp in missing:
                                        missing.remove(comp)
                                
                                if missing:
                                    comp_name = missing[0]
                                    img = load_gbuffer_image(f, f'{comp_name} (no extension, subdir)')
                                    if img is not None:
                                        components[comp_name] = img
                                        print(f"✅ Assigned {f.name} (subdir, no ext) as {comp_name}")
                                        break
    
    print(f"📊 Final G-buffer components loaded: {list(components.keys())}")
    return components


def load_hdri_for_display(hdri_path, max_size=(256, 128)):
    """
    Load and tone-map an HDRI for display.
    
    Args:
        hdri_path: Path to HDRI file
        max_size: Maximum size for display (width, height)
        
    Returns:
        numpy array: Tone-mapped image for display
    """
    # Load HDRI
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    hdri = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED)
    if hdri is None:
        raise ValueError(f"Failed to load HDRI: {hdri_path}")
    
    # Convert BGR to RGB
    if len(hdri.shape) == 3 and hdri.shape[2] == 3:
        hdri = cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB)
    
    # Resize for display if needed
    height, width = hdri.shape[:2]
    if width > max_size[0] or height > max_size[1]:
        scale = min(max_size[0] / width, max_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        hdri = cv2.resize(hdri, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Simple tone mapping for display
    # Clamp extreme values and apply gamma correction
    hdri_display = np.clip(hdri, 0, None)
    hdri_display = hdri_display / (hdri_display.max() + 1e-6)  # Normalize to [0, 1]
    hdri_display = np.power(hdri_display, 1/2.2)  # Gamma correction
    
    return hdri_display


def visualize_results(input_image, gbuffer_components, relit_results, output_path=None):
    """
    Create a comprehensive visualization of the relighting results.
    
    Args:
        input_image: Path to original input image
        gbuffer_components: Dictionary with G-buffer components
        relit_results: List of tuples (hdri_path, relit_image_path, rotation_degrees)
        output_path: Optional path to save the visualization
        
    Returns:
        str: Path where visualization was saved (if saved), None otherwise
    """
    num_relit = len(relit_results)
    
    # Calculate grid layout
    # Top row: original image + G-buffer components (4 images)
    # Bottom rows: HDRI and relit pairs (2 images per relit result)
    num_cols = 4  # Original, albedo, depth, normals
    num_rows = 1 + num_relit  # Top row + one row per relit result
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(16, 4 * num_rows))
    gs = GridSpec(num_rows, num_cols, figure=fig, hspace=0.3, wspace=0.2)
    
    # Load and display original image
    original_img = np.array(Image.open(input_image))
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original_img)
    ax_orig.set_title(f"Original Image\n{Path(input_image).name}", fontsize=10)
    ax_orig.axis('off')
    
    # Display G-buffer components
    component_names = ['albedo', 'depth', 'normals']
    for i, comp_name in enumerate(component_names):
        ax_comp = fig.add_subplot(gs[0, i + 1])
        
        if comp_name in gbuffer_components:
            comp_img = gbuffer_components[comp_name]
            
            # Handle different data types and normalize for display
            if comp_name == 'depth':
                if len(comp_img.shape) == 3:
                    comp_img = comp_img[:, :, 0]  # Take first channel for depth
                comp_img = (comp_img - comp_img.min()) / (comp_img.max() - comp_img.min() + 1e-6)
                ax_comp.imshow(comp_img, cmap='plasma')
            else:
                if comp_img.dtype != np.uint8:
                    comp_img = np.clip(comp_img * 255, 0, 255).astype(np.uint8)
                ax_comp.imshow(comp_img)
                
            ax_comp.set_title(f"G-buffer: {comp_name.capitalize()}", fontsize=10)
        else:
            ax_comp.text(0.5, 0.5, f"No {comp_name}\ndata found", 
                        ha='center', va='center', transform=ax_comp.transAxes)
            ax_comp.set_title(f"G-buffer: {comp_name.capitalize()}", fontsize=10)
        
        ax_comp.axis('off')
    
    # Display relit results
    for relit_idx, (hdri_path, relit_image_path, rotation_degrees) in enumerate(relit_results):
        row = relit_idx + 1
        
        # Display HDRI (spans 2 columns)
        ax_hdri = fig.add_subplot(gs[row, :2])
        try:
            hdri_display = load_hdri_for_display(hdri_path)
            ax_hdri.imshow(hdri_display)
            ax_hdri.set_title(f"HDRI {relit_idx + 1}: {Path(hdri_path).stem}\n(rotated {rotation_degrees:.1f}°)", fontsize=10)
        except Exception as e:
            ax_hdri.text(0.5, 0.5, f"Failed to load\nHDRI: {e}", 
                        ha='center', va='center', transform=ax_hdri.transAxes)
            ax_hdri.set_title(f"HDRI {relit_idx + 1}: Error", fontsize=10)
        ax_hdri.axis('off')
        
        # Display relit image (spans 2 columns)
        ax_relit = fig.add_subplot(gs[row, 2:])
        try:
            relit_img = np.array(Image.open(relit_image_path))
            ax_relit.imshow(relit_img)
            ax_relit.set_title(f"Relit Result {relit_idx + 1}", fontsize=10)
        except Exception as e:
            ax_relit.text(0.5, 0.5, f"Failed to load\nrelit image: {e}", 
                         ha='center', va='center', transform=ax_relit.transAxes)
            ax_relit.set_title(f"Relit Result {relit_idx + 1}: Error", fontsize=10)
        ax_relit.axis('off')
    
    plt.suptitle("Single Image Relighting Results", fontsize=16, y=0.98)
    
    saved_path = None
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to: {output_path}")
        saved_path = output_path
    
    plt.show()
    return saved_path


def save_all_images_to_directory(input_image, gbuffer_components, relit_results, output_dir, visualization_path=None):
    """
    Save all images (input, G-buffers, HDRIs, relit results) to organized output directory.
    
    Args:
        input_image: Path to original input image
        gbuffer_components: Dictionary with G-buffer components
        relit_results: List of tuples (hdri_path, relit_image_path, rotation_degrees)
        output_dir: Directory to save all images
        visualization_path: Optional path to visualization image to copy
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Saving all images to: {output_dir}")
    
    # Create organized subdirectories
    input_dir = output_dir / "01_input"
    gbuffers_dir = output_dir / "02_gbuffers"
    hdris_dir = output_dir / "03_hdris"
    relit_dir = output_dir / "04_relit_results"
    
    input_dir.mkdir(exist_ok=True)
    gbuffers_dir.mkdir(exist_ok=True)
    hdris_dir.mkdir(exist_ok=True)
    relit_dir.mkdir(exist_ok=True)
    
    # 1. Save original input image
    input_image_path = Path(input_image)
    saved_input = input_dir / f"original_input{input_image_path.suffix}"
    shutil.copy2(input_image_path, saved_input)
    print(f"💾 Saved original image: {saved_input.name}")
    
    # 2. Save G-buffer components
    for component_name, component_data in gbuffer_components.items():
        try:
            if component_data is not None:
                # Convert to uint8 if needed
                if component_data.dtype != np.uint8:
                    if component_name == 'depth':
                        # Normalize depth for saving
                        if len(component_data.shape) == 3:
                            component_data = component_data[:, :, 0]  # Take first channel
                        normalized = (component_data - component_data.min()) / (component_data.max() - component_data.min() + 1e-6)
                        component_data = (normalized * 255).astype(np.uint8)
                    else:
                        component_data = np.clip(component_data * 255, 0, 255).astype(np.uint8)
                
                # Save as PNG for lossless quality
                output_path = gbuffers_dir / f"gbuffer_{component_name}.png"
                if len(component_data.shape) == 2:
                    # Grayscale image
                    Image.fromarray(component_data, mode='L').save(output_path)
                else:
                    # RGB image
                    Image.fromarray(component_data, mode='RGB').save(output_path)
                print(f"💾 Saved G-buffer {component_name}: {output_path.name}")
        except Exception as e:
            print(f"⚠️  Failed to save G-buffer {component_name}: {e}")
    
    # 3. Save rotated HDRIs and relit results
    for i, (hdri_path, relit_image_path, rotation_degrees) in enumerate(relit_results):
        version_name = f"v{i+1:03d}"
        
        # Save rotated HDRI (copy the already rotated version)
        hdri_output_path = hdris_dir / f"{version_name}_hdri_rot{rotation_degrees:.1f}.hdr"
        try:
            shutil.copy2(hdri_path, hdri_output_path)
            print(f"💾 Saved rotated HDRI {i+1}: {hdri_output_path.name}")
        except Exception as e:
            print(f"⚠️  Failed to save HDRI {i+1}: {e}")
        
        # Save relit result
        relit_output_path = relit_dir / f"{version_name}_relit_result{Path(relit_image_path).suffix}"
        try:
            shutil.copy2(relit_image_path, relit_output_path)
            print(f"💾 Saved relit result {i+1}: {relit_output_path.name}")
        except Exception as e:
            print(f"⚠️  Failed to save relit result {i+1}: {e}")
    
    # 4. Save visualization if provided
    if visualization_path and Path(visualization_path).exists():
        viz_output_path = output_dir / f"00_visualization_summary.png"
        try:
            shutil.copy2(visualization_path, viz_output_path)
            print(f"💾 Saved visualization: {viz_output_path.name}")
        except Exception as e:
            print(f"⚠️  Failed to save visualization: {e}")
    
    # Create a summary text file
    summary_path = output_dir / "relighting_summary.txt"
    try:
        with open(summary_path, 'w') as f:
            f.write("Single Image Relighting Results Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Original Image: {input_image_path.name}\n")
            f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Relit Versions: {len(relit_results)}\n\n")
            
            f.write("G-Buffer Components Found:\n")
            for component_name in gbuffer_components.keys():
                f.write(f"  • {component_name}\n")
            f.write("\n")
            
            f.write("Relit Versions:\n")
            for i, (hdri_path, relit_image_path, rotation_degrees) in enumerate(relit_results):
                f.write(f"  {i+1:2d}. {Path(hdri_path).stem} (rotated {rotation_degrees:.1f}°)\n")
            f.write("\n")
            
            f.write("Directory Structure:\n")
            f.write("  01_input/           - Original input image\n")
            f.write("  02_gbuffers/        - Extracted G-buffer components\n") 
            f.write("  03_hdris/           - Rotated HDRI environment maps\n")
            f.write("  04_relit_results/   - Final relit images\n")
            f.write("  00_visualization_summary.png - Combined visualization\n")
        
        print(f"📄 Created summary: {summary_path.name}")
    except Exception as e:
        print(f"⚠️  Failed to create summary: {e}")
    
    print(f"✅ All images saved to: {output_dir}")
    print(f"📊 Directory structure:")
    print(f"   📁 01_input/          - Original input image")
    print(f"   📁 02_gbuffers/       - G-buffer components ({len(gbuffer_components)} files)")
    print(f"   📁 03_hdris/          - Rotated HDRIs ({len(relit_results)} files)")
    print(f"   📁 04_relit_results/  - Relit images ({len(relit_results)} files)")
    if visualization_path:
        print(f"   📄 00_visualization_summary.png - Combined visualization")
    print(f"   📄 relighting_summary.txt - Process summary")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Single image relighting with random HDRI selection and rotation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--hdri_folder",
        type=str,
        required=True,
        help="Path to folder containing HDRI files (.hdr/.exr)"
    )
    parser.add_argument(
        "--num_relit",
        type=int,
        required=True,
        help="Number of random relit images to generate"
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
        default=704,
        help="Output height (default: 704, cosmos1 expected resolution)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Output width (default: 1280, cosmos1 expected resolution)"
    )
    parser.add_argument(
        "--rotation_range",
        type=float,
        default=360.0,
        help="Maximum rotation range in degrees for random HDRI rotation (default: 360.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible HDRI sampling and rotation (default: None = random)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory to save all images and results (default: temp directory, cleaned up after)"
    )
    parser.add_argument(
        "--save_visualization",
        type=str,
        default=None,
        help="Path to save visualization image (default: display only)"
    )
    parser.add_argument(
        "--keep_original_resolution",
        action="store_true", 
        help="Keep original image resolution instead of auto-resizing to 704×1280 (may cause issues)"
    )
    
    return parser.parse_args()


def main():
    """Main function for single image relighting."""
    args = parse_arguments()
    
    print("=== Single Image Relighting Tool ===")
    print("IMPORTANT: Make sure you have activated the 'cosmos-predict1' conda environment")
    print("before running this script:")
    print("  conda activate cosmos-predict1\n")
    
    # Initialize performance tracker
    perf_tracker = PerformanceTracker()
    
    # Set up paths
    input_image = Path(args.input_image).resolve()
    hdri_folder = Path(args.hdri_folder).resolve()
    cosmos_path = setup_cosmos_environment(args.cosmos_path)
    
    # Validate inputs
    if not input_image.exists():
        raise ValueError(f"Input image not found: {input_image}")
    if not hdri_folder.exists():
        raise ValueError(f"HDRI folder not found: {hdri_folder}")
    if not cosmos_path.exists():
        raise ValueError(f"Cosmos1-diffusion-renderer not found: {cosmos_path}")
    
    # No conflicting arguments to validate now
    
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Using random seed: {args.seed}")
    
    # Handle resolution settings
    if args.keep_original_resolution:
        # Use original image resolution
        with Image.open(input_image) as img:
            original_width, original_height = img.size
        height, width = original_height, original_width
        print(f"🔧 Using original image resolution: {width}×{height}")
        auto_resize = False
    else:
        # Use specified resolution (default is cosmos1 expected)
        height, width = args.height, args.width
        auto_resize = True  # Auto-resize by default
        print(f"🎯 Target resolution for cosmos1: {width}×{height}")
    
    print(f"Settings:")
    print(f"  - Input image: {input_image.name}")
    print(f"  - Number of relit versions: {args.num_relit}")
    print(f"  - Rotation range: ±{args.rotation_range/2:.1f}°")
    print(f"  - Target resolution: {width}x{height}")
    print(f"  - Auto-resize: {'enabled' if auto_resize else 'disabled'}")
    print()
    
    # Create working directory
    if args.output_dir:
        work_dir = Path(args.output_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="single_relight_"))
    
    try:
        print("=== Starting Single Image Relighting ===")
        
        # Step 1: Extract G-buffers
        perf_tracker.start_step("G-buffer extraction")
        print("\n1. Extracting G-buffers...")
        gbuffer_output_dir = work_dir / "gbuffer_output"
        gbuffer_dir = extract_gbuffers(
            input_image, cosmos_path, args.checkpoint_dir,
            gbuffer_output_dir, height, width, perf_tracker, auto_resize
        )
        perf_tracker.end_step()
        
        # Step 2: Find HDRI files
        perf_tracker.start_step("HDRI file validation")
        print("\n2. Finding HDRI files...")
        hdri_files = find_hdri_files(hdri_folder)
        perf_tracker.end_step()
        
        # Step 3: Generate random relit versions
        perf_tracker.start_step("HDRI rotation and relighting")
        print(f"\n3. Generating {args.num_relit} random relit versions...")
        relit_results = []
        rotated_hdris_dir = work_dir / "rotated_hdris"
        rotated_hdris_dir.mkdir(exist_ok=True)
        
        for i in range(args.num_relit):
            # Randomly select HDRI
            selected_hdri = random.choice(hdri_files)
            
            # Generate random rotation
            rotation_degrees = random.uniform(-args.rotation_range/2, args.rotation_range/2)
            
            # Create rotated HDRI
            rotated_filename = f"relit_{i:03d}_{selected_hdri.stem}_rot{rotation_degrees:.1f}{selected_hdri.suffix}"
            rotated_hdri_path = rotated_hdris_dir / rotated_filename
            
            print(f"  Relit {i+1}: {selected_hdri.name} (rotation: {rotation_degrees:.1f}°)")
            rotate_hdr_image(selected_hdri, rotated_hdri_path, rotation_degrees)
            
            # Relight with this HDRI
            relit_output_dir = work_dir / f"relit_{i:03d}"
            relight_with_hdri(
                gbuffer_dir, rotated_hdri_path, cosmos_path, args.checkpoint_dir,
                relit_output_dir, height, width, perf_tracker, i
            )
            
            # Find the relit image
            relit_images = list(relit_output_dir.glob("*.png")) + list(relit_output_dir.glob("*.jpg"))
            if relit_images:
                relit_results.append((rotated_hdri_path, relit_images[0], rotation_degrees))
            else:
                print(f"⚠️  No relit image found for version {i+1}")
        
        perf_tracker.end_step()
        
        # Step 4: Load G-buffer components for visualization
        perf_tracker.start_step("G-buffer loading and processing")
        print("\n4. Loading G-buffer components...")
        gbuffer_components = load_gbuffer_components(gbuffer_dir)
        print(f"✅ Loaded {len(gbuffer_components)} G-buffer components: {list(gbuffer_components.keys())}")
        perf_tracker.end_step()
        
        # Step 5: Create visualization
        perf_tracker.start_step("Visualization generation")
        print("\n5. Creating visualization...")
        
        # Determine visualization save path
        viz_save_path = args.save_visualization
        if args.output_dir and not args.save_visualization:
            # If output_dir is specified but no specific visualization path, save to temp for copying later
            viz_save_path = work_dir / "temp_visualization.png"
        
        visualization_path = visualize_results(
            input_image, 
            gbuffer_components, 
            relit_results,
            viz_save_path
        )
        perf_tracker.end_step()
        
        # Step 6: Save all images to output directory if requested
        if args.output_dir:
            perf_tracker.start_step("Saving all images to output directory")
            print("\n6. Saving all images to output directory...")
            save_all_images_to_directory(
                input_image,
                gbuffer_components,
                relit_results,
                args.output_dir,
                visualization_path
            )
            perf_tracker.end_step()
        
        print(f"\n=== Single Image Relighting Complete ===")
        print(f"Generated {len(relit_results)} relit versions!")
        if args.output_dir:
            print(f"All results saved to: {args.output_dir}")
        else:
            print(f"Temporary files were in: {work_dir} (cleaned up)")
        
        # Print performance summary
        perf_tracker.print_summary()
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        # Still show performance info for partial runs
        if perf_tracker and perf_tracker.step_times:
            print(f"\nPartial performance data before error:")
            perf_tracker.print_summary()
        raise
    
    finally:
        # Clean up temporary directory if we created one and user didn't specify output_dir
        if not args.output_dir and work_dir.exists() and 'single_relight_' in str(work_dir):
            shutil.rmtree(work_dir)
            print(f"Cleaned up temporary directory: {work_dir}")
        elif args.output_dir:
            # Clean up the working directory since we copied everything to output_dir
            if work_dir.exists() and work_dir != Path(args.output_dir) and 'single_relight_' in str(work_dir):
                shutil.rmtree(work_dir)
                print(f"Cleaned up temporary working directory: {work_dir}")


if __name__ == "__main__":
    main()
