"""Simple command-line semantic segmentation with NARadio.

A lightweight script to test NARadio semantic segmentation without Gradio.

Usage:
    python scripts/naradio_semseg_simple.py --image path/to/image.jpg --prompts "car,road,sky"
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


def compute_cos_sim(vec1: torch.FloatTensor, vec2: torch.FloatTensor, softmax: bool = False):
    """Compute cosine similarity between two batches of vectors."""
    N, C1 = vec1.shape
    M, C2 = vec2.shape
    if C1 != C2:
        raise ValueError(f"Feature dimension mismatch: {C1} vs {C2}")

    # Ensure both tensors have the same dtype (convert to float32)
    vec1 = vec1.float()
    vec2 = vec2.float()

    vec1 = vec1 / vec1.norm(dim=-1, keepdim=True)
    vec1 = vec1.reshape(1, N, 1, C1)

    vec2 = vec2 / vec2.norm(dim=-1, keepdim=True)
    vec2 = vec2.reshape(M, 1, C2, 1)

    sim = (vec1 @ vec2).reshape(M, N)
    if softmax:
        return torch.softmax(100 * sim, dim=-1)
    else:
        return sim


def simple_text_encoder(text_list, lang_adaptor=None, device="cuda"):
    """Simple text encoding."""
    if lang_adaptor is None:
        # Fallback: create simple embeddings
        embeddings = []
        for text in text_list:
            hash_val = hash(text.lower()) % 1000000
            embedding = torch.randn(512) * 0.1
            embedding[hash_val % 512] += 1.0
            embeddings.append(embedding)
        return torch.stack(embeddings).to(device)
    else:
        # Use actual language adaptor
        with torch.autocast(device, dtype=torch.float16):
            text = lang_adaptor.tokenizer(text_list).to(device)
            text_features = lang_adaptor.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features


def load_and_preprocess_image(image_path, target_size=None, max_size=None, preserve_aspect=True):
    """Load and preprocess image with flexible resizing options.
    
    Args:
        image_path: Path to the image file
        target_size: Specific target size (H, W) or None for native resolution
        max_size: Maximum dimension size (applies to longest edge if preserve_aspect=True)
        preserve_aspect: Whether to preserve aspect ratio when resizing
    """
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    print(f"Original image size: {original_size[0]}x{original_size[1]}")
    
    if target_size is not None:
        # Use specific target size
        if preserve_aspect:
            # Resize while preserving aspect ratio
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
        else:
            # Force exact size (may distort)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        print(f"Resized to: {image.size[0]}x{image.size[1]}")
    elif max_size is not None:
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
        'return_radio_features': False,
        'upscale_factor': 1,
        'compile': False,
        'amp': True,
        'device': device
    })
    
    return make_extractor(conf)


def generate_segmentation(image_tensor, prompts, extractor, device="cuda", chunk_size=10000):
    """Generate semantic segmentation."""
    print("Extracting features...")
    
    # Extract features
    with torch.no_grad():
        feat_map = extractor(image_tensor.to(device))  # BHWC format
        
        # Try to align with language space
        try:
            feat_map = extractor.align_spatial_features_with_language(feat_map)
            print("Using language-aligned features")
        except (AttributeError, ValueError):
            print("Using raw RADIO features (language alignment not available)")
    
    print("Encoding text prompts...")
    
    # Encode prompts
    lang_adaptor = getattr(extractor, 'lang_adaptor', None)
    prompt_embeddings = simple_text_encoder(prompts, lang_adaptor, device)
    
    print("Computing similarity maps...")
    
    # Compute cosine similarity
    B, H, W, C = feat_map.shape
    feat_map_flat = feat_map.reshape(-1, C)
    
    # Process in chunks to avoid OOM
    num_chunks = int(np.ceil(feat_map_flat.shape[0] / chunk_size))
    cos_sim_chunks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, feat_map_flat.shape[0])
        chunk_features = feat_map_flat[start_idx:end_idx]
        
        chunk_sim = compute_cos_sim(prompt_embeddings, chunk_features, softmax=False)
        cos_sim_chunks.append(chunk_sim)
    
    cos_sim = torch.cat(cos_sim_chunks, dim=0)
    cos_sim = cos_sim.reshape(H, W, len(prompts))
    
    # Normalize similarity maps
    cos_sim = norm_img_01(cos_sim.permute(2, 0, 1).unsqueeze(0))
    cos_sim = cos_sim.squeeze(0).permute(1, 2, 0)
    
    return cos_sim.detach().cpu().numpy()


def create_segmentation_map(similarity_maps, prompts, threshold=0.7):
    """Create a colored segmentation map with one color per prompt.
    
    Args:
        similarity_maps: (H, W, N) array of similarity scores
        prompts: List of prompt strings
        threshold: Minimum similarity threshold for assignment (default: 0.7)
        
    Returns:
        colored_map: (H, W, 3) RGB image
        legend_colors: List of colors used for each prompt + background
        legend_labels: List of labels including background class
    """
    H, W, N = similarity_maps.shape
    
    # Get the best matching prompt and its score for each pixel
    best_scores = np.max(similarity_maps, axis=2)  # (H, W)
    best_matches = np.argmax(similarity_maps, axis=2)  # (H, W)
    
    # Create mask for pixels below threshold - these become "background"
    below_threshold = best_scores < threshold
    
    # Create distinct colors for prompts + background
    # Reserve gray/black for background class
    background_color = np.array([0.3, 0.3, 0.3])  # Dark gray for background
    
    if N <= 9:  # Leave room for background color
        # Use tab10 colormap for prompts
        prompt_colors = plt.cm.tab10(np.linspace(0, 1, N))[:, :3]  # Remove alpha
    else:
        # Use hsv colormap for more prompts
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


def visualize_results(original_image, similarity_maps, prompts, output_dir=None, threshold=0.7):
    """Visualize segmentation results."""
    n_prompts = len(prompts)
    
    # Get image resolution info
    img_height, img_width = original_image.height, original_image.width
    resolution_info = f"{img_width}×{img_height}"
    
    # Create combined segmentation map with threshold
    segmentation_map, legend_colors, legend_labels = create_segmentation_map(
        similarity_maps, prompts, threshold)
    
    # Create subplot grid (add space for segmentation map)
    cols = min(4, n_prompts + 2)  # +2 for original and segmentation map
    rows = (n_prompts + 2 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Add main title with resolution info
    fig.suptitle(f'NARadio Semantic Segmentation - Resolution: {resolution_info}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Show original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title(f"Original Image\n{resolution_info}")
    axes[0, 0].axis('off')
    
    # Show combined segmentation map
    seg_h, seg_w = segmentation_map.shape[:2]
    seg_resolution = f"{seg_w}×{seg_h}"
    
    axes[0, 1].imshow(segmentation_map)
    axes[0, 1].set_title(f"Segmentation Map (threshold={threshold})\n{seg_resolution}")
    axes[0, 1].axis('off')
    
    # Add legend for segmentation map (including background class)
    legend_elements = []
    for i, (label, color) in enumerate(zip(legend_labels, legend_colors)):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label))
    axes[0, 1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
                     fontsize=8, framealpha=0.8)
    
    # Show individual similarity maps
    for i, (prompt, sim_map) in enumerate(zip(prompts, similarity_maps.transpose(2, 0, 1))):
        subplot_idx = i + 2  # +2 to account for original image and segmentation map
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
    for i in range(n_prompts + 2, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Adjust layout to accommodate suptitle
    plt.subplots_adjust(top=0.92)
    
    if output_dir:
        output_path = Path(output_dir) / f"segmentation_results_{resolution_info}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to: {output_path}")
        
        # Also save just the segmentation map
        seg_path = Path(output_dir) / f"segmentation_map_{resolution_info}.png"
        seg_image = Image.fromarray((segmentation_map * 255).astype(np.uint8))
        seg_image.save(seg_path)
        print(f"Segmentation map saved to: {seg_path}")
    
    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='NARadio Semantic Segmentation')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--prompts', required=True, 
                       help='Comma-separated list of prompts (e.g., "car,road,sky")')
    parser.add_argument('--model_name', default='radio_v2.5-b',
                       choices=['radio_v2.5-b', 'radio_v2.5-l', 'radio_v2.5-h', 'radio_v2.5-g'],
                       help='RADIO model version')
    parser.add_argument('--lang_model', default='siglip', choices=['siglip', 'clip'],
                       help='Language model for alignment')
    parser.add_argument('--resolution', type=int, default=None,
                       help='Target resolution for processing (None = native resolution)')
    parser.add_argument('--max_size', type=int, default=None,
                       help='Maximum image dimension (None = no limit, use native resolution)')
    parser.add_argument('--preserve_aspect', action='store_true', default=True,
                       help='Preserve aspect ratio when resizing')
    parser.add_argument('--force_square', action='store_true', default=False,
                       help='Force square resolution (may distort image)')
    parser.add_argument('--output_dir', help='Directory to save results')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Chunk size for similarity computation')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Similarity threshold for prompt assignment (default: 0.7)')
    
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Parse prompts
    prompts = [p.strip() for p in args.prompts.split(',') if p.strip()]
    print(f"Prompts: {prompts}")
    
    # Load model
    print(f"Loading NARadio model: {args.model_name} with {args.lang_model}")
    extractor = create_naradio_extractor(args.model_name, args.lang_model, device)
    print(f"Model loaded. Feature dimension: {extractor.features_dim}")
    
    # Load and preprocess image
    print(f"Loading image: {args.image}")
    
    if args.resolution is not None:
        # Use specific resolution
        if args.force_square:
            target_size = (args.resolution, args.resolution)
            preserve_aspect = False
        else:
            target_size = (args.resolution, args.resolution)
            preserve_aspect = args.preserve_aspect
        image_tensor, original_image = load_and_preprocess_image(
            args.image, target_size=target_size, preserve_aspect=preserve_aspect)
    else:
        # Use native resolution with optional max_size limit
        image_tensor, original_image = load_and_preprocess_image(
            args.image, max_size=args.max_size, preserve_aspect=True)
    
    # Generate segmentation
    similarity_maps = generate_segmentation(
        image_tensor, prompts, extractor, device, args.chunk_size)
    
    # Visualize results
    print("Generating visualization...")
    visualize_results(original_image, similarity_maps, prompts, args.output_dir, args.threshold)
    
    print("Done!")


if __name__ == "__main__":
    main()
