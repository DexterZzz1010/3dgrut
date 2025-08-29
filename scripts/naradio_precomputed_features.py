"""NARadio semantic segmentation using pre-computed features.

This script allows you to:
1. Pre-compute RADIO features from images and save them
2. Load pre-computed features and run semantic segmentation with different text prompts

This is useful for efficiency when testing multiple prompts on the same image.

Usage Examples:
    # Extract and save features
    python scripts/naradio_precomputed_features.py \
        --mode extract \
        --image path/to/image.jpg \
        --output features.pt

    # Use pre-computed features for segmentation
    python scripts/naradio_precomputed_features.py \
        --mode segment \
        --features features.pt \
        --prompts "car,road,sky,building"
"""

import sys
import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from omegaconf import DictConfig

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from threedgrut.features.extractors import make_extractor


def norm_img_01(x):
    """Normalize image to [0, 1] range per channel."""
    B, C, H, W = x.shape
    x = x - torch.min(x.reshape(B, C, H*W), dim=-1).values.reshape(B, C, 1, 1)
    x_max = torch.max(x.reshape(B, C, H*W), dim=-1).values.reshape(B, C, 1, 1)
    x_max = torch.where(x_max == 0, torch.ones_like(x_max), x_max)
    return x / x_max


def load_and_preprocess_image(image_path, max_size=None):
    """Load and preprocess image."""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    print(f"Original image size: {original_size[0]}x{original_size[1]}")
    
    if max_size is not None:
        # Limit maximum dimension while preserving aspect ratio
        w, h = original_size
        if max(w, h) > max_size:
            if w > h:
                new_w, new_h = max_size, int(h * max_size / w)
            else:
                new_w, new_h = int(w * max_size / h), max_size
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            print(f"Resized to: {new_w}x{new_h} (max_size={max_size})")
        else:
            print("No resizing needed (within max_size limit)")
    else:
        print("Using native resolution (no size limit)")
    
    # Convert to tensor in BHWC format
    image_array = np.array(image).astype(np.float32) / 255.0
    tensor_image = torch.from_numpy(image_array).unsqueeze(0)  # Add batch dim
    
    return tensor_image, image


def create_naradio_extractor(model_name="radio_v2.5-b", lang_model="siglip", device="cuda"):
    """Create NARadio feature extractor."""
    conf = DictConfig({
        'type': 'naradio',
        'model_name': model_name,
        'gauss_std': 7.0,
        'lang_model': lang_model,
        'return_radio_features': True,  # We'll handle language alignment separately
        'upscale_factor': 1,
        'compile': False,
        'amp': True,
        'device': device
    })
    
    return make_extractor(conf)


def extract_features(image_path, output_path, model_name="radio_v2.5-b", 
                    lang_model="siglip", max_size=None, device="cuda"):
    """Extract features from image and save them."""
    print(f"Extracting features from: {image_path}")
    
    # Create extractor
    extractor = create_naradio_extractor(model_name, lang_model, device)
    print(f"Model loaded. Feature dimension: {extractor.features_dim}")
    
    # Load and preprocess image
    image_tensor, original_image = load_and_preprocess_image(image_path, max_size)
    
    # Extract features
    print("Extracting features...")
    with torch.no_grad():
        features = extractor(image_tensor.to(device))  # BHWC format
        print(f"Features extracted: {features.shape}")
    
    # Save features
    extractor.save_features(features, output_path)
    
    # Also save original image info for reference
    info_path = output_path.replace('.pt', '_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Original image: {image_path}\n")
        f.write(f"Image size: {original_image.size}\n")
        f.write(f"Feature shape: {features.shape}\n")
        f.write(f"Model version: {model_name}\n")
        f.write(f"Language model: {lang_model}\n")
    
    print(f"Info saved to: {info_path}")


def create_segmentation_map(similarity_maps, prompts, threshold=0.7):
    """Create a colored segmentation map with one color per prompt."""
    H, W, N = similarity_maps.shape
    
    # Get the best matching prompt and its score for each pixel
    best_scores = np.max(similarity_maps, axis=2)  # (H, W)
    best_matches = np.argmax(similarity_maps, axis=2)  # (H, W)
    
    # Create mask for pixels below threshold - these become "background"
    below_threshold = best_scores < threshold
    
    # Create distinct colors for prompts + background
    background_color = np.array([0.3, 0.3, 0.3])  # Dark gray for background
    
    if N <= 9:  # Leave room for background color
        prompt_colors = plt.cm.tab10(np.linspace(0, 1, N))[:, :3]  # Remove alpha
    else:
        prompt_colors = plt.cm.hsv(np.linspace(0, 1, N))[:, :3]
    
    # Combine colors: background first, then prompts
    colors = np.vstack([background_color.reshape(1, -1), prompt_colors])
    labels = ["background"] + prompts
    
    # Create RGB segmentation map
    colored_map = np.zeros((H, W, 3))
    
    # Assign background color to low-confidence pixels
    colored_map[below_threshold] = background_color
    
    # Assign prompt colors to high-confidence pixels
    for i in range(N):
        mask = (best_matches == i) & (~below_threshold)  # High confidence for this prompt
        colored_map[mask] = prompt_colors[i]
    
    return colored_map, colors, labels


def visualize_similarity_results(similarity_maps, prompts, threshold=0.7, output_dir=None):
    """Visualize similarity results."""
    n_prompts = len(prompts)
    
    # Create segmentation map
    segmentation_map, legend_colors, legend_labels = create_segmentation_map(
        similarity_maps, prompts, threshold)
    
    # Create subplot grid
    cols = min(4, n_prompts + 1)  # +1 for segmentation map
    rows = (n_prompts + 1 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Add main title
    fig.suptitle(f'NARadio Semantic Segmentation (Pre-computed Features)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Show segmentation map
    seg_h, seg_w = segmentation_map.shape[:2]
    seg_resolution = f"{seg_w}×{seg_h}"
    
    axes[0, 0].imshow(segmentation_map)
    axes[0, 0].set_title(f"Segmentation Map (threshold={threshold})\n{seg_resolution}")
    axes[0, 0].axis('off')
    
    # Add legend
    legend_elements = []
    for i, (label, color) in enumerate(zip(legend_labels, legend_colors)):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label))
    axes[0, 0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
                     fontsize=8, framealpha=0.8)
    
    # Show individual similarity maps
    for i, (prompt, sim_map) in enumerate(zip(prompts, similarity_maps.transpose(2, 0, 1))):
        subplot_idx = i + 1
        if subplot_idx >= rows * cols:
            break  # Skip if we run out of subplot space
        row = subplot_idx // cols
        col = subplot_idx % cols
        
        # Get similarity map resolution
        sim_h, sim_w = sim_map.shape
        sim_resolution = f"{sim_w}×{sim_h}"
        
        im = axes[row, col].imshow(sim_map, cmap='viridis', vmin=0, vmax=1)
        axes[row, col].set_title(f'"{prompt}"\n{sim_resolution}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046)
    
    # Hide unused subplots
    for i in range(n_prompts + 1, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if output_dir:
        output_path = Path(output_dir) / f"precomputed_segmentation_{seg_resolution}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to: {output_path}")
        
        # Also save just the segmentation map
        seg_path = Path(output_dir) / f"precomputed_segmentation_map_{seg_resolution}.png"
        seg_image = Image.fromarray((segmentation_map * 255).astype(np.uint8))
        seg_image.save(seg_path)
        print(f"Segmentation map saved to: {seg_path}")
    
    plt.show()


def segment_from_features(features_path, prompts, model_name="radio_v2.5-b", 
                         lang_model="siglip", threshold=0.7, output_dir=None, device="cuda"):
    """Perform segmentation using pre-computed features."""
    print(f"Loading features from: {features_path}")
    
    # Create extractor (needed for text encoding and language alignment)
    extractor = create_naradio_extractor(model_name, lang_model, device)
    extractor.return_radio_features = False  # We want language-aligned features
    
    # Load pre-computed features
    features = extractor.load_features(features_path, device)
    
    print(f"Computing similarity for prompts: {prompts}")
    
    # Compute text similarity
    with torch.no_grad():
        similarity_maps = extractor.compute_text_similarity(features, prompts)
        similarity_maps = similarity_maps.squeeze(0).cpu().numpy()  # Remove batch dim and move to CPU
    
    print(f"Similarity maps computed: {similarity_maps.shape}")
    
    # Visualize results
    visualize_similarity_results(similarity_maps, prompts, threshold, output_dir)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='NARadio Pre-computed Features')
    parser.add_argument('--mode', required=True, choices=['extract', 'segment'],
                       help='Mode: extract features or segment using pre-computed features')
    
    # Common options
    parser.add_argument('--model_name', default='radio_v2.5-b',
                       choices=['radio_v2.5-b', 'radio_v2.5-l', 'radio_v2.5-h', 'radio_v2.5-g'],
                       help='RADIO model version')
    parser.add_argument('--lang_model', default='siglip', choices=['siglip', 'clip'],
                       help='Language model for alignment')
    parser.add_argument('--output_dir', help='Directory to save results')
    
    # Extract mode options
    parser.add_argument('--image', help='Path to input image (for extract mode)')
    parser.add_argument('--output', help='Output path for features (for extract mode)')
    parser.add_argument('--max_size', type=int, default=None,
                       help='Maximum image dimension (None = no limit)')
    
    # Segment mode options
    parser.add_argument('--features', help='Path to pre-computed features (for segment mode)')
    parser.add_argument('--prompts', help='Comma-separated list of prompts (for segment mode)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Similarity threshold for prompt assignment (default: 0.7)')
    
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if args.mode == 'extract':
        if not args.image or not args.output:
            parser.error("Extract mode requires --image and --output")
        
        extract_features(
            args.image, args.output, args.model_name, 
            args.lang_model, args.max_size, device)
    
    elif args.mode == 'segment':
        if not args.features or not args.prompts:
            parser.error("Segment mode requires --features and --prompts")
        
        prompts = [p.strip() for p in args.prompts.split(',') if p.strip()]
        
        segment_from_features(
            args.features, prompts, args.model_name,
            args.lang_model, args.threshold, args.output_dir, device)
    
    print("Done!")


if __name__ == "__main__":
    main()
