# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from omegaconf import DictConfig
import torch.hub
import math
from timm.layers import use_fused_attn


class GaussKernelAttn(nn.Module):
    """Custom attention mechanism with Gaussian kernel enhancement.
    
    This implements the NACLIP attention mechanism that applies a Gaussian kernel
    to the attention weights to improve spatial structure understanding.
    Supports dynamic input resolutions by computing attention matrices on-the-fly.
    """

    def __init__(
        self,
        orig_attn,
        gauss_std: float,
        device,
        chosen_cls_id: int,
        dim: int,
        patch_size: int = 16,
        qk_norm: bool = False,
        num_prefix_tokens: int = 8,
    ) -> None:
        """Initialize the Gaussian kernel attention.
        
        Args:
            orig_attn: Original attention module to replace
            gauss_std: Standard deviation for Gaussian kernel
            device: Device to store tensors on
            chosen_cls_id: Index of chosen class token
            dim: Attention dimension
            patch_size: Size of patches used by the model
            qk_norm: Whether to apply query-key normalization
            num_prefix_tokens: Number of prefix tokens (e.g., CLS tokens)
        """
        super().__init__()
        num_heads = orig_attn.num_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        
        # Store parameters for dynamic computation
        self.patch_size = patch_size
        self.chosen_cls_id = chosen_cls_id
        self.gauss_std = gauss_std
        self.device = device
        self.num_prefix_tokens = num_prefix_tokens
        
        # Cache for attention addition matrices by resolution
        self._attn_cache = {}

        # Copy components from original attention
        self.qkv = orig_attn.qkv
        self.q_norm = orig_attn.q_norm if qk_norm else nn.Identity()
        self.k_norm = orig_attn.k_norm if qk_norm else nn.Identity()
        self.attn_drop = orig_attn.attn_drop
        self.proj = orig_attn.proj
        self.proj_drop = orig_attn.proj_drop

    def forward(self, x: torch.Tensor, current_resolution: Tuple[int, int] = None, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the attention mechanism.
        
        Args:
            x: Input tensor
            current_resolution: Current input resolution for dynamic attention
        """
        B, N, C = x.shape
        
        # If no resolution provided, try to infer from input tensor dimensions
        if current_resolution is None:
            # During initialization or when resolution not available,
            # try to infer from the number of tokens
            # For Vision Transformer: N = H*W + num_prefix_tokens
            spatial_tokens = N - self.num_prefix_tokens
            if spatial_tokens > 0:
                # Assume square patches for simplicity
                H_patches = W_patches = int(spatial_tokens ** 0.5)
                if H_patches * W_patches == spatial_tokens:
                    # Assume patch_size=16 for default resolution
                    current_resolution = (H_patches * self.patch_size, W_patches * self.patch_size)
                else:
                    # Non-square or unknown layout, use a reasonable default
                    current_resolution = (224, 224)
            else:
                # Fallback to default resolution
                current_resolution = (224, 224)
        
        x_out = self.custom_attn(x.permute(1, 0, 2), current_resolution)
        x_out = x_out.permute(1, 0, 2)
        return x_out

    @staticmethod
    def gaussian_window(dim1, dim2, std=5., device="cuda"):
        """Create a Gaussian window for attention enhancement.
        
        Args:
            dim1, dim2: Dimensions of the window
            std: Standard deviation of the Gaussian
            device: Device to store tensors on
            
        Returns:
            torch.Tensor: Gaussian window tensor
        """
        constant = 1 / (std * math.sqrt(2))
        start = -(dim1 - 1) / 2.0
        k1 = torch.linspace(start=start * constant,
                            end=(start + (dim1 - 1)) * constant,
                            steps=dim1,
                            dtype=torch.float, device=device)
        start = -(dim2 - 1) / 2.0
        k2 = torch.linspace(start=start * constant,
                            end=(start + (dim2 - 1)) * constant,
                            steps=dim2,
                            dtype=torch.float, device=device)
        dist_square_to_mu = (torch.stack(torch.meshgrid(
            k1, k2, indexing="ij")) ** 2).sum(0)

        return torch.exp(-dist_square_to_mu)

    @staticmethod
    def get_attention_addition(dim1, dim2, window, num_prefix_tokens=8):
        """Create attention addition matrix for spatial enhancement.
        
        Args:
            dim1, dim2: Spatial dimensions
            window: Gaussian window tensor
            num_prefix_tokens: Number of prefix tokens
            
        Returns:
            torch.Tensor: Attention addition matrix
        """
        d = window.device
        m = torch.einsum("ij,kl->ijkl",
                         torch.eye(dim1, device=d),
                         torch.eye(dim2, device=d))
        m = m.permute((0, 3, 1, 2)).contiguous()
        out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1),
                       window.unsqueeze(0).unsqueeze(1),
                       padding='same').squeeze(1)

        out = out.view(dim1 * dim2, dim1 * dim2)
        if num_prefix_tokens > 0:
            v_adjusted = torch.vstack(
                [torch.zeros((num_prefix_tokens, dim1 * dim2), device=d), out])
            out = torch.hstack([torch.zeros(
                (dim1 * dim2 + num_prefix_tokens, num_prefix_tokens), device=d),
                v_adjusted])

        return out

    def _get_attn_addition(self, current_resolution: Tuple[int, int]) -> torch.Tensor:
        """Get or compute attention addition matrix for given resolution.
        
        Args:
            current_resolution: (height, width) of current input
            
        Returns:
            torch.Tensor: Attention addition matrix
        """
        h, w = current_resolution
        cache_key = (h, w)
        
        if cache_key not in self._attn_cache:
            n_patches = (w // self.patch_size, h // self.patch_size)
            window_size = [side * 2 - 1 for side in n_patches]
            window = GaussKernelAttn.gaussian_window(*window_size, std=self.gauss_std, device=self.device)
            attn_addition = GaussKernelAttn.get_attention_addition(
                *n_patches, window, self.num_prefix_tokens
            ).unsqueeze(0)
            self._attn_cache[cache_key] = attn_addition
            
        return self._attn_cache[cache_key]

    def custom_attn(self, x, current_resolution: Tuple[int, int]):
        """Custom attention computation with Gaussian enhancement.
        
        Args:
            x: Input tensor
            current_resolution: Current input resolution (height, width)
        """
        num_heads = self.num_heads
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # Key modification: k.T @ k instead of q.T @ k
        attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale

        # Add Gaussian attention enhancement (computed dynamically)
        attn_addition = self._get_attn_addition(current_resolution)
        attn_weights += attn_addition
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            -1, bsz, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        return attn_output

    def clear_cache(self):
        """Clear the attention addition cache."""
        self._attn_cache.clear()


class NARadioFeatureExtractor(nn.Module):
    """NARadio feature extractor based on RADIO with custom attention enhancement.
    
    This extractor implements the NARadio approach which combines NVIDIA's RADIO model
    with a custom Gaussian kernel attention mechanism inspired by NACLIP. The model
    modifies the attention of the last layer of RADIO to improve spatial structure
    understanding.
    
    Key features:
    - Uses NVIDIA's RADIO as the backbone
    - Applies custom Gaussian kernel attention in the last layer
    - Supports dynamic input resolutions (like regular RADIO)
    - Gaussian attention matrices computed dynamically and cached for efficiency
    - Supports language alignment capabilities (optional)
    - Mixed precision inference using bfloat16
    - Automatic resolution adjustment to nearest supported size
    
    Attributes:
        model_name (str): RADIO model version to use
        gauss_std (float): Standard deviation for Gaussian kernel
        lang_model (str): Language model for alignment ("siglip" or "clip")
        return_radio_features (bool): Whether to return raw RADIO features
    """
    
    def __init__(
        self,
        conf: DictConfig,
        device: Optional[str] = None,
    ):
        """Initialize the NARadio feature extractor.
        
        Args:
            conf: Hydra configuration containing NARadio parameters
            device: Device to store tensors on (default: same as input tensors)
        """
        super().__init__()
        self.device = device if device is not None else conf.device
        
        # Parse configuration
        self.model_name = conf.model_name
        self.gauss_std = conf.gauss_std
        self.lang_model = conf.get("lang_model", "siglip")
        self.return_radio_features = conf.get("return_radio_features", True)
        self.upscale_factor = conf.get("upscale_factor", 1)
        self.compile_model = conf.get("compile", False)
        self.amp = conf.get("amp", True)
        
        # Initialize RADIO model
        self._init_radio_model()
        
        # Get output feature dimension
        self._features_dim = self._get_features_dim()
    
    def _init_radio_model(self):
        """Initialize RADIO model with custom attention."""
        # Load RADIO model
        adaptors = [self.lang_model] if not self.return_radio_features else []
        self.model = torch.hub.load("NVlabs/RADIO", "radio_model",
                                    version=self.model_name, progress=True,
                                    skip_validation=True,
                                    adaptor_names=adaptors)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model.make_preprocessor_external()
        
        # Store language adaptor if available
        self.lang_adaptor = None
        if hasattr(self.model, 'adaptors') and self.model.adaptors is not None:
            if self.lang_model in self.model.adaptors:
                self.lang_adaptor = self.model.adaptors[self.lang_model]
                # Remove adaptors from model to control when they're applied
                self.model.adaptors = None
        
        # Replace the last attention layer with custom Gaussian kernel attention
        last_block = self.model.model.blocks[-1]
        chosen_cls_id = 0
        if self.lang_adaptor is not None:
            chosen_cls_id = self.lang_adaptor.head_idx
            
        last_block.attn = GaussKernelAttn(
            last_block.attn,
            gauss_std=self.gauss_std,
            device=self.device,
            chosen_cls_id=chosen_cls_id,
            dim=self.model.model.embed_dim,
            patch_size=self.model.patch_size,
            num_prefix_tokens=self.model.num_summary_tokens)
        
        # Store model properties
        self.patch_size = self.model.patch_size
        
        # Compile model if requested
        if self.compile_model:
            self.model.compile(fullgraph=True, options={"triton.cudagraphs": True})
            if self.lang_adaptor is not None:
                self.lang_adaptor.compile(fullgraph=True, options={"triton.cudagraphs": True})
    
    def _get_features_dim(self) -> int:
        """Determine the output feature dimension of the model."""
        # Create minimal dummy input
        dummy_size = 224
        nearest_res = self.model.get_nearest_supported_resolution(dummy_size, dummy_size)
        x = torch.zeros(1, 3, nearest_res[0], nearest_res[1]).to(self.device)
        
        # Temporarily store the current resolution for the attention layer
        self._temp_resolution = nearest_res
        
        # Run forward pass to get feature dimension
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.amp):
            out = self.model(x)
            features = out.features
            
            # Apply language adaptor if available and not returning raw RADIO features
            if not self.return_radio_features and self.lang_adaptor is not None:
                features = self.lang_adaptor.head_mlp(features)
        
        # Clean up temporary resolution
        delattr(self, '_temp_resolution')
                
        return features.shape[-1]  # Last dimension contains features
    
    @property
    def features_dim(self) -> int:
        """Returns the output feature dimension of the model."""
        return self._features_dim
    
    def get_output_shape(
        self, 
        input_shape: Union[Tuple[int, ...], torch.Size]
    ) -> Tuple[int, int]:
        """Compute the expected output shapes for a given input shape.
        
        Args:
            input_shape: Tuple of (batch_size, height, width, channels) in BHWC format
            
        Returns:
            Tuple[int, int]: (H_out, W_out) output spatial dimensions
        """
        B, H, W, _ = input_shape
        
        # Apply upscaling
        upscaled_H = H * self.upscale_factor
        upscaled_W = W * self.upscale_factor
        
        # Get nearest supported resolution
        H_valid, W_valid = self.model.get_nearest_supported_resolution(upscaled_H, upscaled_W)
        
        # Compute output spatial dimensions
        H_out = H_valid // self.patch_size
        W_out = W_valid // self.patch_size
        
        return H_out, W_out
    
    def clear_attention_cache(self):
        """Clear the attention cache (useful for memory management)."""
        self.model.model.blocks[-1].attn.clear_cache()
    
    def is_compatible_size(self, h: int, w: int) -> bool:
        """Check if the given size is compatible with the model."""
        nearest_h, nearest_w = self.model.get_nearest_supported_resolution(h, w)
        return nearest_h == h and nearest_w == w
    
    def get_nearest_size(self, h: int, w: int) -> Tuple[int, int]:
        """Get the nearest supported size for the given dimensions."""
        return self.model.get_nearest_supported_resolution(h, w)
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C) in BHWC format
                where B=batch size, H=height, W=width, C=3 (RGB channels).
                Values should be in range [0, 1]
                
        Returns:
            torch.Tensor: Spatial features in BHWC format
        """
        # Convert from BHWC to BCHW
        x = x.permute(0, 3, 1, 2)
        
        # Apply upscaling if specified
        if self.upscale_factor > 1:
            upscaled_shape = (x.shape[2] * self.upscale_factor, x.shape[3] * self.upscale_factor)
            x = torch.nn.functional.interpolate(x, upscaled_shape, mode='bicubic', align_corners=False)
        
        # Get nearest supported resolution and resize if needed
        current_h, current_w = x.shape[2], x.shape[3]
        nearest_res = self.model.get_nearest_supported_resolution(current_h, current_w)
        
        if nearest_res != (current_h, current_w):
            x = torch.nn.functional.interpolate(
                x, nearest_res, mode='bicubic', align_corners=False
            )
        
        # Store final processed dimensions for reshaping
        final_h, final_w = x.shape[2], x.shape[3]
        
        # Monkey patch the attention forward to pass current resolution
        original_forward = self.model.model.blocks[-1].attn.forward
        current_resolution = (final_h, final_w)
        
        def patched_forward(x_attn, *args, **kwargs):
            return original_forward(x_attn, current_resolution=current_resolution, *args, **kwargs)
        
        self.model.model.blocks[-1].attn.forward = patched_forward
        
        try:
            # Process with autocast for mixed precision
            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.amp):
                out = self.model(x)
                features = out.features
                
                # Apply language adaptor if available and not returning raw RADIO features
                if not self.return_radio_features and self.lang_adaptor is not None:
                    features = self.lang_adaptor.head_mlp(features)
        finally:
            # Restore original forward method
            self.model.model.blocks[-1].attn.forward = original_forward
        
        # Convert from (B, T, D) to (B, H, W, D) format
        B, T, D = features.shape
        H_out = final_h // self.patch_size
        W_out = final_w // self.patch_size
        
        # Verify dimensions match
        expected_tokens = H_out * W_out
        if T != expected_tokens:
            raise RuntimeError(
                f"Dimension mismatch: got {T} spatial tokens but expected {expected_tokens} "
                f"(H_out={H_out}, W_out={W_out}, patch_size={self.patch_size})"
            )
        
        spatial_features = features.view(B, H_out, W_out, D)
        
        return spatial_features.to(torch.float32)
    
    def encode_image_to_vector(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """Extract global image features as a vector.
        
        Args:
            rgb_image: Input image tensor in BCHW format
            
        Returns:
            torch.Tensor: Global feature vector
        """
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.amp):
            out = self.model(rgb_image)
            
            # Extract global features from summary tokens
            if hasattr(out, 'summary') and out.summary is not None:
                C = out.summary.shape[-1] // 3
                if self.lang_adaptor is not None:
                    i = self.lang_adaptor.head_idx
                    global_features = out.summary[:, C*i: C*(i+1)]
                else:
                    global_features = out.summary[:, :C]  # Use first set of summary tokens
                
                # Apply language adaptor if available and not returning raw RADIO features
                if not self.return_radio_features and self.lang_adaptor is not None:
                    global_features = self.lang_adaptor.head_mlp(global_features)
                    
                return global_features
            else:
                # Fallback: use global average pooling of spatial features
                features = out.features
                if not self.return_radio_features and self.lang_adaptor is not None:
                    features = self.lang_adaptor.head_mlp(features)
                return features.mean(dim=1)  # Average over spatial dimension
    
    def align_spatial_features_with_language(self, features: torch.Tensor) -> torch.Tensor:
        """Align spatial features with language space.
        
        Args:
            features: Spatial features tensor in BHWC format
            
        Returns:
            torch.Tensor: Language-aligned spatial features
        """
        if self.lang_adaptor is None:
            raise ValueError("Cannot align to language without a lang model")
        if not self.return_radio_features:
            return features  # Already aligned
            
        B, H, W, C = features.shape
        features = features.view(B, -1, C)  # Reshape to (B, H*W, C)
        
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.amp):
            aligned_features = self.lang_adaptor.head_mlp(features)
            
        return aligned_features.view(B, H, W, -1)  # Reshape back to BHWC
    
    def align_global_features_with_language(self, features: torch.Tensor) -> torch.Tensor:
        """Align global features with language space.
        
        Args:
            features: Global features tensor
            
        Returns:
            torch.Tensor: Language-aligned global features
        """
        if self.lang_adaptor is None:
            raise ValueError("Cannot align to language without a lang model")
        if not self.return_radio_features:
            return features  # Already aligned
            
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.amp):
            return self.lang_adaptor.head_mlp(features)
    
    def compute_text_similarity(self, features: torch.FloatTensor, text_prompts: list) -> torch.FloatTensor:
        """Compute similarity between pre-computed features and text prompts.
        
        Args:
            features: Pre-computed spatial features in BHWC format of shape (B, H, W, C)
            text_prompts: List of text prompts to compare against
            
        Returns:
            torch.Tensor: Similarity maps of shape (B, H, W, N) where N is number of prompts
        """
        # Ensure features are on the right device
        features = features.to(self.device)
        
        # Align features with language space if needed
        if self.return_radio_features and self.lang_adaptor is not None:
            features = self.align_spatial_features_with_language(features)
        
        # Encode text prompts
        if self.lang_adaptor is not None:
            with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
                text = self.lang_adaptor.tokenizer(text_prompts).to(self.device)
                text_features = self.lang_adaptor.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
        else:
            # Fallback text encoding (basic implementation)
            text_features = self._simple_text_encode(text_prompts)
        
        # Compute cosine similarity
        B, H, W, C = features.shape
        feat_flat = features.reshape(B * H * W, C)  # (B*H*W, C)
        
        # Normalize features
        feat_flat = feat_flat / feat_flat.norm(dim=-1, keepdim=True)
        
        # Compute similarity: (B*H*W, C) x (N, C) -> (B*H*W, N)
        similarities = torch.mm(feat_flat, text_features.t())
        
        # Reshape back to spatial format
        similarities = similarities.reshape(B, H, W, len(text_prompts))
        
        return similarities
    
    def _simple_text_encode(self, text_prompts: list) -> torch.FloatTensor:
        """Simple fallback text encoding when no language adaptor is available."""
        # This is a basic implementation - you might want to use a more sophisticated method
        embeddings = []
        for text in text_prompts:
            # Create a simple hash-based embedding (not ideal, but works for demo)
            hash_val = hash(text.lower()) % 1000000
            embedding = torch.randn(self.features_dim) * 0.1  # Random base
            embedding[hash_val % self.features_dim] += 1.0   # Add signal based on text
            embeddings.append(embedding)
        return torch.stack(embeddings).to(self.device)
    
    def save_features(self, features: torch.FloatTensor, save_path: str):
        """Save pre-computed features to disk.
        
        Args:
            features: Features tensor in BHWC format
            save_path: Path to save the features (.pt file)
        """
        torch.save({
            'features': features.cpu(),
            'shape': features.shape,
            'model_name': self.model_name,
            'features_dim': self.features_dim,
            'return_radio_features': self.return_radio_features
        }, save_path)
        print(f"Features saved to: {save_path}")
    
    @staticmethod
    def load_features(load_path: str, device: str = 'cuda') -> torch.FloatTensor:
        """Load pre-computed features from disk.
        
        Args:
            load_path: Path to the saved features (.pt file)
            device: Device to load features on
            
        Returns:
            torch.Tensor: Loaded features in BHWC format
        """
        data = torch.load(load_path, map_location=device)
        features = data['features'].to(device)
        print(f"Features loaded from: {load_path}")
        print(f"  - Shape: {data['shape']}")
        print(f"  - Model: {data.get('model_name', 'unknown')}")
        print(f"  - Feature dim: {data.get('features_dim', 'unknown')}")
        return features
