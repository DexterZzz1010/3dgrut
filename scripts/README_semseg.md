# NARadio Semantic Segmentation Scripts

This directory contains scripts for testing semantic segmentation with the NARadio feature extractor.

## Scripts

### 1. Interactive Web App (`naradio_semseg_app.py`)

A Gradio-based web interface similar to the RayFronts semantic segmentation demo.

**Requirements:**
```bash
pip install gradio matplotlib pillow
```

**Usage:**
```bash
python scripts/naradio_semseg_app.py
```

**Options:**
- `--model_name`: RADIO model version (radio_v2.5-b, radio_v2.5-l, radio_v2.5-h, radio_v2.5-g)
- `--lang_model`: Language model for alignment (siglip, clip)
- `--chunk_size`: Chunk size for similarity computation (reduce if OOM)

**Features:**
- Upload images via web interface
- Add multiple text prompts interactively
- Adjust resolution and processing settings
- Real-time visualization of segmentation results
- Support for both raw and language-aligned features

### 2. Command-Line Tool (`naradio_semseg_simple.py`)

A lightweight command-line tool for quick testing and batch processing.

**Requirements:**
```bash
pip install matplotlib pillow
```

**Usage:**
```bash
python scripts/naradio_semseg_simple.py \
    --image path/to/your/image.jpg \
    --prompts "car,road,sky,building"
```

**Options:**
- `--image`: Path to input image (required)
- `--prompts`: Comma-separated list of prompts (required)
- `--model_name`: RADIO model version (default: radio_v2.5-b)
- `--lang_model`: Language model (default: siglip)
- `--resolution`: Specific target resolution (default: None = native resolution)
- `--max_size`: Maximum dimension for memory control (default: None = no limit)
- `--preserve_aspect`: Preserve aspect ratio when resizing (default: True)
- `--force_square`: Force square resolution (may distort, default: False)
- `--output_dir`: Directory to save results (optional)
- `--chunk_size`: Chunk size for similarity computation (default: 10000)
- `--threshold`: Similarity threshold for prompt assignment (default: 0.7)

## Examples

### Example 1: Native Resolution (Default)
```bash
python scripts/naradio_semseg_simple.py \
    --image examples/street.jpg \
    --prompts "car,road,sidewalk,building,sky,person"
# Uses full native resolution with no size limit
```

### Example 2: Large Image with Memory Control
```bash
python scripts/naradio_semseg_simple.py \
    --image examples/huge_image.jpg \
    --prompts "tree,grass,water,rock,sky,mountain" \
    --max_size 2048 \
    --output_dir results/
```

### Example 3: Force Specific Resolution
```bash
python scripts/naradio_semseg_simple.py \
    --image examples/detailed.jpg \
    --prompts "person,face,clothing,background" \
    --resolution 1024 \
    --preserve_aspect
```

### Example 4: Custom Threshold
```bash
python scripts/naradio_semseg_simple.py \
    --image examples/complex_scene.jpg \
    --prompts "person,car,building" \
    --threshold 0.5  # Lower threshold = more pixels assigned to classes
```

### Example 5: High Confidence Segmentation
```bash
python scripts/naradio_semseg_simple.py \
    --image examples/clear_scene.jpg \
    --prompts "car,road,sky" \
    --threshold 0.8  # Higher threshold = more background pixels
```

## How It Works

1. **Feature Extraction**: The NARadio model extracts spatial features from the input image
2. **Language Alignment**: Features are optionally projected to language-aligned space using SIGLIP/CLIP
3. **Text Encoding**: Input prompts are encoded using the same language model
4. **Similarity Computation**: Cosine similarity is computed between text embeddings and spatial features
5. **Threshold Filtering**: Pixels with similarity below threshold (0.7) are assigned to "background" class
6. **Segmentation Map**: Creates a combined map where each pixel is colored by its best matching prompt
7. **Visualization**: Shows both individual similarity maps and a combined segmentation map

## Key Features

- **Dynamic Resolution Support**: Unlike the original NARadio, these scripts support varying input resolutions
- **Efficient Processing**: Chunked similarity computation to handle large images without OOM
- **Language Alignment**: Optional use of language-aligned features for better text-image matching
- **Combined Segmentation Map**: Color-coded map showing the best matching prompt for each pixel
- **Multiple Visualizations**: Shows original image, segmentation map, and individual similarity heatmaps
- **Automatic Saving**: Saves both full results and standalone segmentation maps

## Performance Tips

- Use smaller `--chunk_size` if you encounter out-of-memory errors
- Lower resolution for faster processing during development
- Use `radio_v2.5-b` for speed, `radio_v2.5-h` or `radio_v2.5-g` for quality
- Enable language alignment for better text-image correspondence
- Adjust `--threshold` based on your needs:
  - **Higher threshold (0.8-0.9)**: More conservative, cleaner segmentation
  - **Lower threshold (0.5-0.6)**: More aggressive, assigns more pixels to classes
  - **Default (0.7)**: Good balance for most images

## Troubleshooting

**Out of Memory:**
- Reduce `--resolution` or `--chunk_size`
- Use a smaller model version

**Poor Segmentation Quality:**
- Try language-aligned features (`use_language_alignment=True` in web app)
- Experiment with different prompts (more specific vs. more general)
- Use higher resolution for fine details

**Slow Processing:**
- Use smaller resolution
- Use `radio_v2.5-b` instead of larger models
- Increase chunk size (if memory allows)
