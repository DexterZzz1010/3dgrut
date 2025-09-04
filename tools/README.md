# Tools Directory

This directory contains utility scripts for working with the 3DGRUT relighting system.

## Single Image Relighting Tool

### `single_image_relighting.py`

A tool for relighting a single image using random HDRI environment maps. This script:

1. **Extracts G-buffers** from a single input image using cosmos1-diffusion-renderer
2. **Randomly selects HDRIs** from a specified folder
3. **Randomly rotates the HDRIs** for variation
4. **Generates relit images** using the rotated HDRIs
5. **Displays comprehensive results** showing:
   - Original input image
   - G-buffer components (albedo, depth, normals)
   - For each relit version: the rotated HDRI and corresponding relit result

### Prerequisites

```bash
# Activate the cosmos-predict1 conda environment
conda activate cosmos-predict1
```

### Usage

#### Basic Usage
```bash
python tools/single_image_relighting.py \
    --input_image /path/to/your/image.jpg \
    --hdri_folder /path/to/hdri/files \
    --num_relit 3
```

#### Advanced Usage
```bash
python tools/single_image_relighting.py \
    --input_image /path/to/your/image.jpg \
    --hdri_folder /path/to/hdri/files \
    --num_relit 5 \
    --cosmos_path /path/to/cosmos1-diffusion-renderer \
    --rotation_range 180.0 \
    --seed 42 \
    --output_dir /path/to/results \
    --save_visualization /path/to/visualization.png
```

### Arguments

**Required:**
- `--input_image`: Path to the input image to relight
- `--hdri_folder`: Path to folder containing HDRI files (.hdr/.exr)
- `--num_relit`: Number of random relit images to generate

**Optional:**
- `--cosmos_path`: Path to cosmos1-diffusion-renderer repository (default: `/mnt/dev/cosmos1-diffusion-renderer`)
- `--checkpoint_dir`: Directory containing model checkpoints (default: `checkpoints`)
- `--height`: Output height in pixels (default: 704, cosmos1 expected resolution)
- `--width`: Output width in pixels (default: 1280, cosmos1 expected resolution)
- `--rotation_range`: Maximum rotation range in degrees (default: 360.0)
- `--seed`: Random seed for reproducible results (default: None = random)
- `--output_dir`: Output directory to save all images and results (default: temp directory, cleaned up after)
- `--save_visualization`: Path to save the visualization image (default: display only)
- `--keep_original_resolution`: Keep original image resolution instead of auto-resizing to 704×1280 (may cause issues)

### Automatic Image Rescaling

The tool **automatically handles image resolution** to ensure compatibility with cosmos1-diffusion-renderer:

#### Default Behavior (Recommended)
- **Auto-detects** input image resolution
- **Automatically resizes** to cosmos1's expected 704×1280 resolution if needed
- **High-quality resampling** using Lanczos interpolation
- **Preserves aspect ratio** by resizing to exact target dimensions
- **Performance tracking** for resize operations

#### Example Resolution Handling
```
📏 Input image resolution: 1920×1080
⚠️  Resolution mismatch: got 1920×1080, expected 1280×704
🔧 Will resize to match cosmos1-diffusion-renderer expected resolution
💡 Original image is 2.73x larger than expected
📐 Resized image: 1920×1080 → 1280×704
```

#### Override Options
- Use `--keep_original_resolution` to skip auto-resizing (not recommended unless image is already 704×1280)
- Manually specify `--height` and `--width` for custom target resolution
- Resolution validation warnings help identify potential issues

#### Why This Matters
- **cosmos1-diffusion-renderer expects 704×1280** for optimal performance
- Wrong resolution can cause model inference errors or poor quality results
- Automatic rescaling ensures consistent, reliable processing

### Output Directory Structure

When you specify `--output_dir`, the tool saves all images in an organized directory structure:

```
output_directory/
├── 00_visualization_summary.png    # Combined visualization grid
├── 01_input/
│   └── original_input.jpg          # Original input image  
├── 02_gbuffers/
│   ├── gbuffer_albedo.png          # Extracted albedo/basecolor
│   ├── gbuffer_depth.png           # Extracted depth information
│   ├── gbuffer_normals.png         # Extracted surface normals
│   └── gbuffer_metallic.png        # Additional G-buffer components (if available)
├── 03_hdris/
│   ├── v001_hdri_rot45.2.hdr       # Rotated HDRI environment maps
│   ├── v002_hdri_rot-23.7.hdr
│   └── v003_hdri_rot180.0.hdr
├── 04_relit_results/
│   ├── v001_relit_result.png       # Final relit images
│   ├── v002_relit_result.png
│   └── v003_relit_result.png
└── relighting_summary.txt          # Process summary and metadata
```

#### Usage Example with Output Directory
```bash
# Save all results to a directory
python tools/single_image_relighting.py \
    --input_image /path/to/image.jpg \
    --hdri_folder /path/to/hdris \
    --num_relit 3 \
    --output_dir /path/to/results
```

The directory structure makes it easy to:
- **Browse results systematically** with numbered folders
- **Reuse G-buffers** for further processing  
- **Compare different HDRI rotations** side by side
- **Archive complete relighting sessions** with metadata

### Example Output

The tool creates a comprehensive visualization grid showing:

**Top row:**
- Original input image
- G-buffer albedo component
- G-buffer depth component  
- G-buffer normals component

**Subsequent rows (one per relit version):**
- Rotated HDRI environment map (left half)
- Corresponding relit result image (right half)

### Dependencies

The script is based on functions extracted from `scripts/generate_multilights_dataset.py` and requires:
- cosmos1-diffusion-renderer
- OpenCV with OpenEXR support
- PIL/Pillow
- NumPy
- Matplotlib
- tqdm

### Troubleshooting

1. **Make sure you've activated the cosmos-predict1 environment** before running
2. **Check that HDRIs are valid** - corrupted files will be automatically skipped
3. **Verify cosmos1-diffusion-renderer path** and checkpoint directories exist
4. **Ensure sufficient disk space** for temporary files during processing

### Debugging Features

The script includes comprehensive debugging to help identify common issues:

#### G-Buffer Loading Debug
- **File discovery**: Lists all files found in G-buffer directories
- **Loading attempts**: Shows which files are being processed and why they succeed/fail  
- **Subdirectory search**: Automatically checks subdirectories for G-buffer files
- **Fallback loading**: Attempts to load generic files when specific components aren't found
- **Final summary**: Reports exactly which components were successfully loaded

#### Performance Tracking Debug
- **Initialization confirmation**: Verifies performance tracker is created
- **Step progress**: Shows when each processing step starts and completes
- **Error handling**: Displays partial performance data even when script fails
- **Comprehensive summary**: Detailed timing breakdown with visual progress bars

#### Example Debug Output

When G-buffers are found:
```
🔍 Looking for G-buffer files in: /tmp/.../gbuffer_frames
📂 Found 12 items in G-buffer directory:
   📄 albedo.exr (2048576 bytes)
   📄 depth.exr (1048576 bytes) 
   📄 normal.exr (2048576 bytes)
📷 Loading albedo from: albedo.exr
✅ Loaded albedo: shape (704, 1280, 3)
📊 Final G-buffer components loaded: ['albedo', 'depth', 'normals']
```

When performance tracking works:
```
⏱️  Starting: G-buffer extraction
✅ Completed: G-buffer extraction (45.23s)
🔧 Debug: Total steps tracked so far: 1

============================================================
🚀 PERFORMANCE SUMMARY
============================================================
📊 Step-by-step timing:
  G-buffer extraction       │  45.23s │  67.2% │ ████████████████████
⚡ Performance insights:
  • Slowest step: G-buffer extraction (45.23s)
🏁 Total execution time: 67.37s (1.1 minutes)
============================================================
```

See `DEBUG_TEST_EXAMPLE.md` for complete debugging examples and solutions to common issues.

### Performance Tracking

The script includes comprehensive performance monitoring that tracks timing for each major step:

- **G-buffer extraction**: Time spent extracting G-buffers from input image
- **HDRI file validation**: Time spent validating HDRI files in folder
- **HDRI rotation and relighting**: Combined time for all HDRI rotations and relighting
- **Individual relighting**: Separate timing for each HDRI relighting operation
- **G-buffer loading**: Time spent loading G-buffer components for visualization
- **Visualization generation**: Time spent creating the final visualization

At the end of processing, you'll see a detailed performance summary like:
```
============================================================
🚀 PERFORMANCE SUMMARY
============================================================

📊 Step-by-step timing:
  G-buffer extraction       │  45.23s │  67.2% │ ██████████████████████████████████████████████████
  Relighting #1            │  12.34s │  18.3% │ ████████████████████████████████████
  Relighting #2            │   8.91s │  13.2% │ ████████████████████████████
  Visualization generation │   0.89s │   1.3% │ ███

⚡ Performance insights:
  • Slowest step: G-buffer extraction (45.23s)
  • Fastest step: Visualization generation (0.89s)
  • Average step time: 16.84s
  • Average relighting time: 10.63s per HDRI

🏁 Total execution time: 67.37s (1.1 minutes)

💡 Performance tips:
  • G-buffer extraction took 45.2s vs 21.3s for relighting
  • Consider reusing G-buffers for multiple relighting sessions
============================================================
```

### Performance Notes

- G-buffer extraction happens once at the beginning
- Each HDRI relighting requires a separate model inference
- Performance metrics help identify bottlenecks in your workflow
- For better performance with many relit versions, consider using the batch processing in `scripts/generate_multilights_dataset.py` instead
