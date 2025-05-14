#%%writefile arc_solver_vicreg.py
"""
Main script for training and evaluating a Vision Transformer (ViT) model
with VICReg self-supervised pre-training for the Abstraction and Reasoning Corpus (ARC).

This script supports:
1. VICReg pre-training of the ViT encoder.
2. Supervised training of a decoder head on top of the pre-trained encoder.
3. Solving ARC tasks using the trained model and generating submission files.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from einops import rearrange, repeat # Not strictly used in the final version, can be removed if not needed
# from einops.layers.torch import Rearrange # Not strictly used, can be removed
try:
    from tqdm.notebook import tqdm # For Jupyter/Colab environments
except ImportError:
    from tqdm import tqdm # For terminal environments
import random
import math
import os
import copy
from pathlib import Path
import argparse
import sys
import time
import traceback

# --- Configuration Constants ---
# These can be overridden by command-line arguments.

# File Paths
DATA_PATH_DEFAULT = Path("./") # Base path for data files
TRAINING_CHALLENGES_FILE_DEFAULT = DATA_PATH_DEFAULT / "arc-agi_training_challenges.json"
TRAINING_SOLUTIONS_FILE_DEFAULT = DATA_PATH_DEFAULT / "arc-agi_training_solutions.json"
EVALUATION_CHALLENGES_FILE_DEFAULT = DATA_PATH_DEFAULT / "arc-agi_evaluation_challenges.json"
TEST_CHALLENGES_FILE_DEFAULT = DATA_PATH_DEFAULT / "arc-agi_test_challenges.json"
SUBMISSION_FILE_DEFAULT = DATA_PATH_DEFAULT / "submission.json"
MODEL_SAVE_PATH_DEFAULT = DATA_PATH_DEFAULT / "arc_vicreg_vit_model.pth"

# ARC Grid Parameters
MAX_GRID_SIZE = 30  # Maximum dimension (height/width) for ARC grids after padding
PATCH_SIZE = 5      # Size of patches (patch_size x patch_size)
NUM_COLORS = 10     # Number of possible colors in ARC grids (0-9)
PAD_VALUE = 0       # Value used for padding ARC grids

# Derived Grid and Patch Parameters
IMG_HEIGHT = MAX_GRID_SIZE
IMG_WIDTH = MAX_GRID_SIZE
CONCAT_IMG_HEIGHT = IMG_HEIGHT * 2 # For VICReg pre-training (input|output)
CONCAT_IMG_WIDTH = IMG_WIDTH

# Ensure image dimensions are divisible by patch size
if IMG_HEIGHT % PATCH_SIZE != 0 or IMG_WIDTH % PATCH_SIZE != 0:
    raise ValueError(f"Image dimensions ({IMG_HEIGHT}x{IMG_WIDTH}) must be divisible by patch size ({PATCH_SIZE}).")

NUM_PATCHES_H = IMG_HEIGHT // PATCH_SIZE
NUM_PATCHES_W = IMG_WIDTH // PATCH_SIZE
NUM_PATCHES = NUM_PATCHES_H * NUM_PATCHES_W # For a single 30x30 grid

CONCAT_NUM_PATCHES_H = CONCAT_IMG_HEIGHT // PATCH_SIZE
CONCAT_NUM_PATCHES_W = CONCAT_IMG_WIDTH // PATCH_SIZE
CONCAT_NUM_PATCHES = CONCAT_NUM_PATCHES_H * CONCAT_NUM_PATCHES_W # For a concatenated 60x30 grid

# Model Hyperparameters (Defaults)
EMBED_DIM_DEFAULT = 192         # Embedding dimension for ViT
ENCODER_DEPTH_DEFAULT = 6       # Number of Transformer blocks in the encoder
ENCODER_HEADS_DEFAULT = 6       # Number of attention heads in the encoder
MLP_DIM_DEFAULT = EMBED_DIM_DEFAULT * 4 # Dimension of the MLP hidden layer in ViT blocks
DECODER_DIM_DEFAULT = 192       # Embedding dimension for the decoder
MASK_RATIO_DEFAULT = 0.75       # Default mask ratio for VICReg pre-training views (patches to mask out)

# VICReg Projector Hyperparameters (Defaults)
VICREG_PROJECTOR_HIDDEN_DIM_DEFAULT = 2048
VICREG_PROJECTOR_OUTPUT_DIM_DEFAULT = 2048

# VICReg Loss Coefficients (Defaults)
VICREG_SIM_COEFF_DEFAULT = 25.0  # Coefficient for the similarity (invariance) term
VICREG_STD_COEFF_DEFAULT = 25.0  # Coefficient for the standard deviation (variance) term
VICREG_COV_COEFF_DEFAULT = 1.0   # Coefficient for the covariance term
VICREG_EPSILON = 1e-4            # Epsilon for numerical stability in variance calculation

# Training Hyperparameters (Defaults)
EPOCHS_DEFAULT = 50                 # Default epochs for VICReg pre-training
BATCH_SIZE_DEFAULT = 16             # Default batch size
LR_DEFAULT = 3e-4                   # Default learning rate for VICReg pre-training
WEIGHT_DECAY_DEFAULT = 0.05         # Default weight decay
DECODER_EPOCHS_DEFAULT = 30         # Default epochs for decoder training
DECODER_LR_DEFAULT = 1e-4           # Default learning rate for decoder training

# System and Utility Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FREQ = 50                     # Frequency of printing training stats (every N batches)
DEBUG_PRINT_BATCH_ZERO_DEFAULT = False # Default for enabling detailed debug prints for batch 0, epoch 0
GRAD_CLIP_VALUE = 1.0               # Gradient clipping value

# Sanity checks for model dimensions
if EMBED_DIM_DEFAULT % ENCODER_HEADS_DEFAULT != 0:
    raise ValueError(f"EMBED_DIM ({EMBED_DIM_DEFAULT}) must be divisible by ENCODER_HEADS ({ENCODER_HEADS_DEFAULT})")
if DECODER_DIM_DEFAULT % ENCODER_HEADS_DEFAULT != 0: # Assuming decoder uses same number of heads for simplicity
    raise ValueError(f"DECODER_DIM ({DECODER_DIM_DEFAULT}) must be divisible by ENCODER_HEADS ({ENCODER_HEADS_DEFAULT})")

# Global variable for controlling debug prints, will be set by args
DEBUG_PRINT_BATCH_ZERO = DEBUG_PRINT_BATCH_ZERO_DEFAULT


# --- Data Loading and Preprocessing ---

def load_arc_data(filepath: Path) -> dict | None:
    """Loads ARC challenge or solution data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def grid_to_tensor(grid: list[list[int]], max_h: int = MAX_GRID_SIZE, max_w: int = MAX_GRID_SIZE, pad_value: int = PAD_VALUE) -> torch.Tensor:
    """
    Converts a 2D list representation of an ARC grid to a padded PyTorch tensor.
    The grid is centered within the padded tensor.
    """
    if not grid or not grid[0]: # Handle empty or malformed grid
        return torch.full((max_h, max_w), pad_value, dtype=torch.long)

    grid_np = np.array(grid, dtype=np.int64)
    h, w = grid_np.shape

    padded_grid_np = np.full((max_h, max_w), pad_value, dtype=np.int64)

    # Clip grid if it's larger than max dimensions
    h_clip = min(h, max_h)
    w_clip = min(w, max_w)
    grid_np_clipped = grid_np[:h_clip, :w_clip]

    # Calculate padding for centering
    pad_h_total = max_h - h_clip
    pad_w_total = max_w - w_clip
    pad_top = pad_h_total // 2
    pad_left = pad_w_total // 2

    # Place the clipped grid into the center of the padded grid
    padded_grid_np[pad_top : pad_top + h_clip, pad_left : pad_left + w_clip] = grid_np_clipped
    
    return torch.tensor(padded_grid_np, dtype=torch.long)

def tensor_to_grid(tensor: torch.Tensor, original_h: int, original_w: int, max_h: int = MAX_GRID_SIZE, max_w: int = MAX_GRID_SIZE) -> list[list[int]]:
    """
    Converts a padded PyTorch tensor back to a 2D list representation of an ARC grid,
    cropping it to the original dimensions by removing centered padding.
    """
    if original_h <= 0 or original_w <= 0:
        return [] # Return empty list for invalid original dimensions

    # Calculate padding that was added
    pad_h_total = max_h - original_h
    pad_w_total = max_w - original_w
    pad_top = pad_h_total // 2
    pad_left = pad_w_total // 2

    # Determine slice indices to extract the original grid
    start_row = max(0, pad_top) # Ensure non-negative
    start_col = max(0, pad_left) # Ensure non-negative
    end_row = min(max_h, start_row + original_h)
    end_col = min(max_w, start_col + original_w)

    # Handle cases where original dimensions might be larger than tensor after padding (should not happen with correct padding)
    if start_row >= end_row or start_col >= end_col:
        return [] 

    grid_np = tensor.cpu().numpy()[start_row:end_row, start_col:end_col]
    return grid_np.tolist()


# --- Positional Embedding Utilities ---

def get_2d_sincos_pos_embed(embed_dim: int, grid_h_patches: int, grid_w_patches: int, cls_token: bool = False) -> np.ndarray:
    """
    Generates 2D sinusoidal positional embeddings.
    Args:
        embed_dim: The embedding dimension.
        grid_h_patches: Number of patches in the height dimension.
        grid_w_patches: Number of patches in the width dimension.
        cls_token: If True, prepends a zero embedding for a CLS token.
    Returns:
        A numpy array of shape (num_patches (+1 if cls_token), embed_dim).
    """
    grid_h_coords = np.arange(grid_h_patches, dtype=np.float32)
    grid_w_coords = np.arange(grid_w_patches, dtype=np.float32)
    # Create a meshgrid of coordinates
    grid_coords = np.stack(np.meshgrid(grid_w_coords, grid_h_coords), axis=0)  # Shape: (2, grid_h_patches, grid_w_patches)
    grid_coords = grid_coords.reshape([2, 1, grid_h_patches, grid_w_patches]) # Reshape for get_2d_sincos_pos_embed_from_grid

    pos_embed_patches = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_coords) # Shape: (grid_h_patches * grid_w_patches, embed_dim)

    if cls_token:
        cls_pos_embed = np.zeros([1, embed_dim])
        pos_embed = np.concatenate([cls_pos_embed, pos_embed_patches], axis=0)
    else:
        pos_embed = pos_embed_patches
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Helper function for 2D sinusoidal embeddings from a grid of coordinates."""
    assert embed_dim % 2 == 0, "Embedding dimension must be even for sinusoidal embeddings."
    # Calculate embeddings for height and width dimensions separately
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) # grid[0] is W-coords if meshgrid(W,H)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) # grid[1] is H-coords
    emb = np.concatenate([emb_h, emb_w], axis=1) # Concatenate to form final embedding
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Helper function for 1D sinusoidal embeddings from a grid of positions."""
    assert embed_dim % 2 == 0, "Embedding dimension must be even for sinusoidal embeddings."
    omega = 1. / (10000 ** (np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.)))
    pos_flat = pos.reshape(-1) # Flatten position array
    out = np.einsum('m,d->md', pos_flat, omega) # Outer product
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# --- Vision Transformer Components ---

class PatchEmbed(nn.Module):
    """Image to Patch Embedding layer."""
    def __init__(self, img_h: int = IMG_HEIGHT, img_w: int = IMG_WIDTH, 
                 patch_size: int = PATCH_SIZE, in_chans: int = 1, embed_dim: int = EMBED_DIM_DEFAULT):
        super().__init__()
        self.img_h, self.img_w = img_h, img_w
        self.patch_size = patch_size
        self.num_patches = (img_h // patch_size) * (img_w // patch_size)
        # Convolutional layer to project patches to embeddings
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).
        Returns:
            Tensor of shape (B, num_patches, embed_dim).
        """
        B, C, H, W = x.shape
        # Optional: Add check for H, W consistency with self.img_h, self.img_w if strictness is needed
        # if H != self.img_h or W != self.img_w:
        #     warnings.warn(f"Input image size ({H}x{W}) differs from PatchEmbed expected size ({self.img_h}x{self.img_w}).")
        x = self.proj(x)       # (B, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)       # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class Attention(nn.Module):
    """Multi-Head Self-Attention layer."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Attention dimension ({dim}) must be divisible by number of heads ({num_heads})")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 # Scaling factor for dot products

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # Combined Q, K, V projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) # Output projection
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape # Batch size, Num patches (sequence length), Channels (embedding dim)
        # Reshape QKV for multi-head attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # Separate Q, K, V: (B, num_heads, N, head_dim)

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """MLP (FeedForward) layer for Transformer blocks."""
    def __init__(self, in_features: int, hidden_features: int | None = None, 
                 out_features: int | None = None, act_layer=nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x) # Dropout after activation
        x = self.fc2(x)
        x = self.drop(x) # Dropout after second linear layer
        return x

class ViTBlock(nn.Module):
    """A single Vision Transformer block."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False, 
                 drop: float = 0., attn_drop: float = 0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # Note: drop_path (stochastic depth) could be added here for regularization if needed
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x)) # Apply attention with pre-normalization
        x = x + self.mlp(self.norm2(x))  # Apply MLP with pre-normalization
        return x

class Projector(nn.Module):
    """Projector MLP for VICReg (and other SSL methods)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3, norm_layer=nn.BatchNorm1d):
        super().__init__()
        layers = []
        current_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if norm_layer: # Typically BatchNorm1d for SSL projectors
                layers.append(norm_layer(hidden_dim))
            layers.append(nn.GELU()) # Activation function
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim)) # Final layer to output dimension
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# --- Main ARC ViT Model with VICReg and Decoder ---

class ARCVicregViT(nn.Module):
    """
    Vision Transformer for ARC, supporting VICReg pre-training and a task-specific decoder.
    """
    def __init__(self, 
                 img_h: int = IMG_HEIGHT, img_w: int = IMG_WIDTH, patch_size: int = PATCH_SIZE, in_chans: int = 1,
                 embed_dim: int = EMBED_DIM_DEFAULT, encoder_depth: int = ENCODER_DEPTH_DEFAULT, 
                 encoder_heads: int = ENCODER_HEADS_DEFAULT, decoder_dim: int = DECODER_DIM_DEFAULT, 
                 mlp_ratio: float = 4., norm_layer=nn.LayerNorm, num_colors: int = NUM_COLORS,
                 decoder_depth_ratio: float = 0.5, # Ratio of encoder depth for decoder depth
                 vicreg_projector_hidden_dim: int = VICREG_PROJECTOR_HIDDEN_DIM_DEFAULT,
                 vicreg_projector_output_dim: int = VICREG_PROJECTOR_OUTPUT_DIM_DEFAULT, 
                 mask_ratio: float = MASK_RATIO_DEFAULT):
        super().__init__()

        self.patch_h, self.patch_w = patch_size, patch_size # Store patch dimensions
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.mask_ratio = mask_ratio # Mask ratio for VICReg pre-training views

        # --- Encoder Components ---
        # Patch embed for VICReg pre-training (operates on concatenated 60x30 grids)
        self.pretrain_patch_embed = PatchEmbed(CONCAT_IMG_HEIGHT, CONCAT_IMG_WIDTH, patch_size, in_chans, embed_dim)
        self.pretrain_num_patches = self.pretrain_patch_embed.num_patches

        # CLS token for VICReg pre-training (learnable parameter)
        self.pretrain_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embedding for VICReg pre-training (1 for CLS + num_patches)
        self.pretrain_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.pretrain_num_patches, embed_dim), requires_grad=False)

        # Patch embed for single 30x30 grids (used by encoder during decoder training and inference)
        self.infer_patch_embed = PatchEmbed(IMG_HEIGHT, IMG_WIDTH, patch_size, in_chans, embed_dim)
        self.infer_num_patches = self.infer_patch_embed.num_patches # Should be NUM_PATCHES
        # Positional embedding for single 30x30 grids (no CLS token for this stage's encoder input)
        self.infer_pos_embed = nn.Parameter(torch.zeros(1, self.infer_num_patches, embed_dim), requires_grad=False)

        # ViT Encoder Blocks (shared by pre-training and decoder's conditioning)
        self.encoder_blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=encoder_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(embed_dim) # Final normalization for encoder output

        # --- VICReg Projector (only for VICReg pre-training) ---
        self.projector = Projector(embed_dim, vicreg_projector_hidden_dim, vicreg_projector_output_dim)

        # --- Decoder Components (trained in a second stage, used for inference) ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True) # Projects encoder output to decoder dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim)) # Learnable mask token for output patches
        # Positional embedding for the decoder's output grid (30x30)
        self.decoder_pos_embed_output = nn.Parameter(torch.zeros(1, self.infer_num_patches, decoder_dim), requires_grad=False)

        decoder_actual_depth = max(1, int(encoder_depth * decoder_depth_ratio)) # Ensure at least 1 decoder block
        self.decoder_blocks = nn.ModuleList([
            ViTBlock(dim=decoder_dim, num_heads=encoder_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_actual_depth)])
        self.decoder_norm = norm_layer(decoder_dim) # Final normalization for decoder output
        # Prediction head: maps decoder output to patch logits (patch_size*patch_size*num_colors)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size*patch_size*num_colors, bias=True)

        self.initialize_weights() # Call weight initialization

    def initialize_weights(self):
        """Initializes weights for the model, including positional embeddings and CLS token."""
        # --- VICReg Pre-training related embeddings (with CLS token) ---
        pretrain_num_h_patches = CONCAT_IMG_HEIGHT // self.patch_h
        pretrain_num_w_patches = CONCAT_IMG_WIDTH // self.patch_w
        
        # Sin-cos positional embeddings for patch tokens (excluding CLS)
        patch_pos_embed_pretrain_data = get_2d_sincos_pos_embed(
            self.embed_dim, pretrain_num_h_patches, pretrain_num_w_patches, cls_token=False
        )
        # Assign to the part of pretrain_pos_embed corresponding to patches
        self.pretrain_pos_embed.data[:, 1:, :].copy_(torch.from_numpy(patch_pos_embed_pretrain_data).float().unsqueeze(0))
        # Initialize CLS token's positional embedding (e.g., with small normal noise, or keep as zeros)
        torch.nn.init.normal_(self.pretrain_pos_embed.data[:, 0, :], std=.02)
        # Initialize the CLS token itself
        torch.nn.init.normal_(self.pretrain_cls_token, std=.02)

        # --- Single grid (inference/decoder training) related embeddings for encoder ---
        infer_num_h_patches = IMG_HEIGHT // self.patch_h
        infer_num_w_patches = IMG_WIDTH // self.patch_w
        pos_embed_infer_data = get_2d_sincos_pos_embed(
            self.embed_dim, infer_num_h_patches, infer_num_w_patches, cls_token=False
        )
        self.infer_pos_embed.data.copy_(torch.from_numpy(pos_embed_infer_data).float().unsqueeze(0))

        # --- Decoder output positional embedding (for 30x30 output grid) ---
        decoder_out_pos_embed_data = get_2d_sincos_pos_embed(
            self.decoder_dim, infer_num_h_patches, infer_num_w_patches, cls_token=False # Decoder dim for these
        )
        self.decoder_pos_embed_output.data.copy_(torch.from_numpy(decoder_out_pos_embed_data).float().unsqueeze(0))
        
        # --- Initialize Patch Embedding Convolutional Layers ---
        for patch_embed_module in [self.pretrain_patch_embed, self.infer_patch_embed]:
            # Xavier uniform initialization for convolutional weights
            w = patch_embed_module.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if patch_embed_module.proj.bias is not None:
                torch.nn.init.constant_(patch_embed_module.proj.bias, 0)

        # --- Initialize Decoder Mask Token ---
        torch.nn.init.normal_(self.mask_token, std=.02)

        # --- Apply _init_weights to other layers (Linear, LayerNorm, Conv2d biases) ---
        self.apply(self._init_weights_generic) # Renamed for clarity

    def _init_weights_generic(self, m: nn.Module):
        """Generic weight initialization for Linear, LayerNorm, BatchNorm, and Conv2d biases."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight) # Xavier uniform for linear layers
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)): # LayerNorm and BatchNorm
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # Convolutional weights for PatchEmbed are handled in initialize_weights.
            # This handles biases for any other Conv2d layers if they were added.
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def unpatchify(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """
        Converts a sequence of patch predictions back into a grid image.
        Args:
            x: Tensor of shape (B, num_patches, patch_h*patch_w*num_colors).
            h_patches: Number of patches in height.
            w_patches: Number of patches in width.
        Returns:
            Tensor of shape (B, num_colors, H_grid, W_grid).
        """
        p_h, p_w = self.patch_h, self.patch_w
        B = x.shape[0]
        num_output_patches = h_patches * w_patches

        if x.shape[1] != num_output_patches:
            raise ValueError(f"Number of patches mismatch in unpatchify: expected {num_output_patches}, got {x.shape[1]}")
        
        expected_last_dim = p_h * p_w * self.num_colors
        if x.shape[-1] != expected_last_dim:
            raise ValueError(f"Unexpected last dimension size in unpatchify. Expected {expected_last_dim}, got {x.shape[-1]}")
        
        # Reshape and permute to reconstruct grid
        # x: (B, L, P_h*P_w*NumColors) -> (B, H_patch, W_patch, Patch_H, Patch_W, NumColors)
        x = x.reshape(B, h_patches, w_patches, p_h, p_w, self.num_colors)
        # Permute to (B, NumColors, H_patch, Patch_H, W_patch, Patch_W)
        x = x.permute(0, 5, 1, 3, 2, 4)
        # Reshape to (B, NumColors, H_grid, W_grid)
        imgs = x.reshape(B, self.num_colors, h_patches * p_h, w_patches * p_w)
        return imgs

    def _encode_view_for_vicreg(self, x_img: torch.Tensor, epoch_idx: int | None = None, 
                                batch_idx: int | None = None, view_id_for_debug: int = 0) -> torch.Tensor:
        """Encodes one view of concatenated grids for VICReg pre-training using a CLS token and random masking."""
        B = x_img.shape[0]
        
        # 1. Patchify the input image (concatenated input|output grid)
        x_patched = self.pretrain_patch_embed(x_img)  # Shape: (B, self.pretrain_num_patches, embed_dim)
        N_total_patches = x_patched.shape[1]

        # 2. Add positional embeddings to patch tokens (excluding CLS token's position for now)
        x_patched_pos = x_patched + self.pretrain_pos_embed[:, 1:, :] # Pos embeds for patches start at index 1

        # 3. Randomly mask (remove) some patch tokens
        num_patches_to_keep = int(N_total_patches * (1 - self.mask_ratio))
        if num_patches_to_keep == 0 and N_total_patches > 0: # Ensure at least one patch is kept
            num_patches_to_keep = 1
        
        kept_patches_with_pos: torch.Tensor
        ids_kept_patches_debug: torch.Tensor | None = None # For debugging

        if N_total_patches == 0: # Should not happen with valid input
            kept_patches_with_pos = torch.empty(B, 0, self.embed_dim, device=x_img.device)
        elif num_patches_to_keep < N_total_patches : # Apply masking
            noise = torch.rand(B, N_total_patches, device=x_patched_pos.device) # Random noise for shuffling
            ids_shuffle = torch.argsort(noise, dim=1) # Indices for shuffling
            ids_keep = ids_shuffle[:, :num_patches_to_keep] # Indices of patches to keep
            ids_kept_patches_debug = ids_keep # Store for debugging
            # Gather the kept patch tokens
            kept_patches_with_pos = torch.gather(x_patched_pos, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        else: # Keep all patches (e.g., if mask_ratio is 0 or num_patches_to_keep >= N_total_patches)
            ids_kept_patches_debug = torch.arange(N_total_patches, device=x_img.device).unsqueeze(0).repeat(B,1)
            kept_patches_with_pos = x_patched_pos

        # --- Debugging Print for Batch 0, Epoch 0 ---
        if DEBUG_PRINT_BATCH_ZERO and epoch_idx == 0 and batch_idx == 0:
            print(f"  DEBUG view {view_id_for_debug} (Epoch {epoch_idx}, Batch {batch_idx}):")
            print(f"    N_total_patches: {N_total_patches}, Mask Ratio: {self.mask_ratio:.2f}, Num_patches_to_keep: {kept_patches_with_pos.shape[1]}")
            if N_total_patches > 0 and kept_patches_with_pos.shape[1] > 0:
                print(f"    Norm of first kept patch (item 0 of batch): {torch.norm(kept_patches_with_pos[0,0,:]).item():.4f}")
            if ids_kept_patches_debug is not None:
                print(f"    ids_kept_patches_debug (first 5 of item 0): {ids_kept_patches_debug[0, :min(5, ids_kept_patches_debug.shape[1])].tolist()}")
        
        # 4. Prepare CLS token: expand to batch size and add its positional embedding
        cls_token_with_pos = self.pretrain_cls_token.expand(B, -1, -1) + self.pretrain_pos_embed[:, :1, :] # CLS pos embed is at index 0

        # 5. Concatenate CLS token with the kept (visible) patch tokens
        x_full_sequence = torch.cat((cls_token_with_pos, kept_patches_with_pos), dim=1) # (B, 1 + num_patches_to_keep, embed_dim)

        # 6. Pass the full sequence (CLS + visible patches) through encoder blocks
        encoded_sequence = x_full_sequence
        for blk in self.encoder_blocks:
            encoded_sequence = blk(encoded_sequence)
        encoded_sequence = self.encoder_norm(encoded_sequence)

        # 7. Extract the representation of the CLS token (it's at index 0)
        cls_representation = encoded_sequence[:, 0] # Shape: (B, embed_dim)
        return cls_representation

    def forward_vicreg_loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates the VICReg loss components."""
        B, D_proj = z_a.shape # Batch size, Projector output dimension

        # Similarity (Invariance) Loss: MSE between the two projected views
        sim_loss = F.mse_loss(z_a, z_b)

        # Standard Deviation (Variance) Loss: Encourages variance along each dimension to be 1
        std_z_a = torch.sqrt(z_a.var(dim=0) + VICREG_EPSILON) # Variance across batch, add epsilon for stability
        std_z_b = torch.sqrt(z_b.var(dim=0) + VICREG_EPSILON)
        std_loss_a = torch.mean(F.relu(1 - std_z_a)) # Penalize if std_dev < 1
        std_loss_b = torch.mean(F.relu(1 - std_z_b))
        std_loss = (std_loss_a + std_loss_b) / 2

        # Covariance Loss: Encourages off-diagonal elements of covariance matrix to be zero
        cov_loss = torch.tensor(0.0, device=z_a.device) # Default for B<=1 or D_proj=0
        if B > 1 and D_proj > 0:
            # Center representations
            z_a_norm = z_a - z_a.mean(dim=0)
            z_b_norm = z_b - z_b.mean(dim=0)
            # Covariance matrices
            cov_z_a = (z_a_norm.T @ z_a_norm) / (B - 1) # (D_proj, D_proj)
            cov_z_b = (z_b_norm.T @ z_b_norm) / (B - 1)
            # Sum of squared off-diagonal elements, normalized by D_proj
            cov_loss_a = (torch.triu(cov_z_a, diagonal=1).pow(2).sum() + torch.tril(cov_z_a, diagonal=-1).pow(2).sum()) / D_proj
            cov_loss_b = (torch.triu(cov_z_b, diagonal=1).pow(2).sum() + torch.tril(cov_z_b, diagonal=-1).pow(2).sum()) / D_proj
            cov_loss = (cov_loss_a + cov_loss_b) / 2
        
        # Total VICReg Loss
        total_loss = (VICREG_SIM_COEFF_DEFAULT * sim_loss +
                      VICREG_STD_COEFF_DEFAULT * std_loss +
                      VICREG_COV_COEFF_DEFAULT * cov_loss)
        
        return total_loss, sim_loss, std_loss, cov_loss

    def forward_pretrain(self, imgs_concat: torch.Tensor, batch_idx: int | None = None, epoch_idx: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for VICReg pre-training."""
        # Encode two views of the input image (random masking happens inside _encode_view_for_vicreg)
        h1 = self._encode_view_for_vicreg(imgs_concat, epoch_idx=epoch_idx, batch_idx=batch_idx, view_id_for_debug=1)
        h2 = self._encode_view_for_vicreg(imgs_concat, epoch_idx=epoch_idx, batch_idx=batch_idx, view_id_for_debug=2)
        
        # Project the representations
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # --- Debugging Print for Batch 0, Epoch 0 ---
        if DEBUG_PRINT_BATCH_ZERO and epoch_idx == 0 and batch_idx == 0:
            h_mse = F.mse_loss(h1, h2).item()
            sim_loss_val_for_debug = F.mse_loss(z1,z2).item() # This is what forward_vicreg_loss will calculate as sim_loss

            h_dim = h1.shape[1] if h1.ndim > 1 else 0
            z_dim = z1.shape[1] if z1.ndim > 1 else 0
            h_l2_dist_implied = torch.sqrt(torch.tensor(h_mse * h_dim)).item() if h_dim > 0 else 0.0
            z_l2_dist_implied = torch.sqrt(torch.tensor(sim_loss_val_for_debug * z_dim)).item() if z_dim > 0 else 0.0

            print(f"  DEBUG Encoder/Projector Outputs (Epoch {epoch_idx}, Batch {batch_idx}):")
            print(f"    h1 norm (mean L2 over batch): {torch.norm(h1, p=2, dim=1).mean().item():.4f}, h2 norm: {torch.norm(h2, p=2, dim=1).mean().item():.4f}")
            print(f"    MSE(h1, h2): {h_mse:.4f}")
            print(f"    L2_dist(h1,h2) implied by MSE (avg over batch): {h_l2_dist_implied:.4f}")
            print(f"    Max/Min h1: {h1.max().item():.4f}/{h1.min().item():.4f}, Max/Min h2: {h2.max().item():.4f}/{h2.min().item():.4f}")
            
            print(f"    z1 norm (mean L2 over batch): {torch.norm(z1, p=2, dim=1).mean().item():.4f}, z2 norm: {torch.norm(z2, p=2, dim=1).mean().item():.4f}")
            print(f"    SimLoss (MSE(z1,z2)) to be used: {sim_loss_val_for_debug:.4f}") # This is the raw sim_loss before coefficient
            print(f"    L2_dist(z1,z2) implied by SimLoss (avg over batch): {z_l2_dist_implied:.4f}")
            print(f"    Max/Min z1: {z1.max().item():.4f}/{z1.min().item():.4f}, Max/Min z2: {z2.max().item():.4f}/{z2.min().item():.4f}")
            print("DEBUG END\n")
        
        return self.forward_vicreg_loss(z1, z2)

    def _encode_input_grid(self, input_grid_img: torch.Tensor) -> torch.Tensor:
        """Encodes a single input grid (B, 1, 30, 30) for decoder conditioning or inference."""
        # Uses infer_patch_embed and infer_pos_embed (no CLS token here)
        x = self.infer_patch_embed(input_grid_img) # (B, infer_num_patches, embed_dim)
        x = x + self.infer_pos_embed             # Add positional embeddings
        
        # Pass through encoder blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        encoded_input_features = self.encoder_norm(x) # (B, infer_num_patches, embed_dim)
        return encoded_input_features

    def _decode_to_output_grid_logits(self, encoded_input_features: torch.Tensor) -> torch.Tensor:
        """Decodes from encoded input features to output grid logits."""
        B = encoded_input_features.shape[0]
        
        # Project the full sequence of encoded input features to decoder dimension
        projected_conditioning_tokens = self.decoder_embed(encoded_input_features) # (B, N_input_patches, D_decoder)

        # Prepare mask tokens as queries for the output grid
        output_mask_tokens = self.mask_token.repeat(B, self.infer_num_patches, 1) # (B, N_output_patches, D_decoder)
        # Add positional embeddings for the output grid to these mask tokens/queries
        output_decoder_input_queries = output_mask_tokens + self.decoder_pos_embed_output

        # Concatenate conditioning tokens and output queries for the decoder transformer
        full_decoder_sequence = torch.cat([projected_conditioning_tokens, output_decoder_input_queries], dim=1)
        
        # Apply decoder blocks
        x_decoded = full_decoder_sequence
        for blk in self.decoder_blocks:
            x_decoded = blk(x_decoded)
        x_decoded = self.decoder_norm(x_decoded)

        # We are interested in the part of the sequence corresponding to the output grid predictions
        # These are the tokens that were originally the `output_decoder_input_queries`
        predicted_output_sequence = x_decoded[:, self.infer_num_patches:, :] # (B, N_output_patches, D_decoder)
        
        # Final prediction head to get patch logits
        pred_patches_logits_flat = self.decoder_pred(predicted_output_sequence) # (B, N_output_patches, P*P*NumColors)
        
        # Unpatchify to get grid logits: (B, NumColors, H_grid, W_grid)
        pred_output_grid_logits = self.unpatchify(pred_patches_logits_flat, NUM_PATCHES_H, NUM_PATCHES_W)
        return pred_output_grid_logits

    def forward_train_decoder(self, input_grid_img: torch.Tensor, target_output_grid_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass for training the decoder."""
        # 1. Ensure encoder parts are in eval mode and frozen (gradients should not flow back)
        # This should be handled by set_encoder_trainable(False) before starting decoder training epoch.
        # For safety, explicitly set eval mode on encoder components here.
        self.pretrain_patch_embed.eval(); self.infer_patch_embed.eval()
        self.encoder_blocks.eval(); self.encoder_norm.eval()
        if hasattr(self, 'pretrain_cls_token'): # Ensure CLS token (if exists) is not involved/trained
             self.pretrain_cls_token.requires_grad_(False)

        with torch.no_grad(): # Encoder pass should not compute gradients
            encoded_input_features = self._encode_input_grid(input_grid_img)
        
        # 2. Decode to output grid logits (using trainable decoder weights)
        # Decoder parts should be in train mode (handled by model.train() call before epoch)
        pred_output_grid_logits = self._decode_to_output_grid_logits(encoded_input_features)

        # 3. Compute reconstruction loss
        loss = self.compute_reconstruction_loss(pred_output_grid_logits, target_output_grid_img)
        return loss, pred_output_grid_logits

    def compute_reconstruction_loss(self, pred_output_grid_logits: torch.Tensor, target_output_grid_img: torch.Tensor) -> torch.Tensor:
        """Computes pixel-wise CrossEntropyLoss for reconstruction."""
        # target_output_grid_img: (B, 1, H, W) - Squeeze channel dim for CrossEntropy
        target_pixels = target_output_grid_img.squeeze(1).long() # (B, H, W)
        loss = F.cross_entropy(pred_output_grid_logits, target_pixels) # pred: (B, NumColors, H, W)
        return loss

    def forward_solve(self, input_grid_tensor: torch.Tensor) -> torch.Tensor:
        """For inference: encodes input, decodes to output grid predictions."""
        self.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            encoded_input_features = self._encode_input_grid(input_grid_tensor) # (B, 1, H, W) -> (B, N_patches, E)
            predicted_output_logits = self._decode_to_output_grid_logits(encoded_input_features) # (B, NumColors, H, W)
            # Get predicted class (color) for each pixel
            predicted_output_grid = torch.argmax(predicted_output_logits, dim=1) # (B, H, W)
        return predicted_output_grid

    def _set_trainable_state_generic(self, modules_list: list[nn.Module], params_list: list[nn.Parameter], trainable: bool):
        """Helper to set requires_grad for lists of modules and parameters."""
        for module in modules_list:
            for param in module.parameters():
                param.requires_grad = trainable
        for param_obj in params_list:
            param_obj.requires_grad = trainable
        
    def set_encoder_trainable(self, trainable: bool = True):
        """Sets requires_grad for encoder components."""
        modules = [self.pretrain_patch_embed, self.infer_patch_embed, self.encoder_blocks, self.encoder_norm]
        params = [self.pretrain_pos_embed, self.infer_pos_embed, self.pretrain_cls_token]
        self._set_trainable_state_generic(modules, params, trainable)

    def set_decoder_trainable(self, trainable: bool = True):
        """Sets requires_grad for decoder components."""
        modules = [self.decoder_embed, self.decoder_blocks, self.decoder_norm, self.decoder_pred]
        params = [self.mask_token, self.decoder_pos_embed_output]
        self._set_trainable_state_generic(modules, params, trainable)

    def set_projector_trainable(self, trainable: bool = True):
        """Sets requires_grad for VICReg projector."""
        for param in self.projector.parameters():
            param.requires_grad = trainable


# --- ARC Dataset Class ---

class ARCDataset(Dataset):
    """PyTorch Dataset for ARC tasks."""
    def __init__(self, challenges_file: Path, solutions_file: Path | None = None, 
                 mode: str = 'vicreg_pretrain', max_h: int = MAX_GRID_SIZE, max_w: int = MAX_GRID_SIZE):
        """
        Args:
            challenges_file: Path to the ARC challenges JSON file.
            solutions_file: Path to the ARC solutions JSON file (used for training modes).
            mode: Dataset mode ('vicreg_pretrain', 'decoder_train', 'eval', 'test').
            max_h: Maximum height for padding.
            max_w: Maximum width for padding.
        """
        self.challenges_data = load_arc_data(challenges_file)
        self.solutions_data = load_arc_data(solutions_file) if solutions_file and solutions_file.exists() else None
        self.mode = mode
        self.max_h, self.max_w = max_h, max_w

        if not self.challenges_data:
            raise ValueError(f"Critical error: Failed to load challenges from {challenges_file}. Cannot proceed.")
        
        self.task_ids = list(self.challenges_data.keys()) # Store task IDs for consistent iteration
        self.samples = self._create_samples()

        if not self.samples:
            print(f"Warning: No samples created for mode '{self.mode}' from file {challenges_file}.")

    def _create_samples(self) -> list[dict]:
        """Creates a list of samples (individual input-output pairs or test inputs)."""
        samples = []
        if not self.challenges_data: return samples 

        for task_id in self.task_ids:
            if task_id not in self.challenges_data: continue # Should not happen if task_ids from challenges_data
            task_data = self.challenges_data[task_id]

            data_split_key = 'train' if self.mode in ['vicreg_pretrain', 'decoder_train'] else 'test'
            if data_split_key not in task_data: continue # Skip task if 'train' or 'test' split is missing

            for i, example_data in enumerate(task_data[data_split_key]):
                if 'input' not in example_data: continue # Skip example if 'input' is missing
                
                # For training modes, 'output' must also exist
                if self.mode in ['vicreg_pretrain', 'decoder_train'] and 'output' not in example_data:
                    continue # Skip example if 'output' is missing in training modes
                
                current_sample = {'task_id': task_id, 'example_idx': i, 'input': example_data['input']}
                if self.mode in ['vicreg_pretrain', 'decoder_train']:
                    current_sample['output'] = example_data['output']
                
                # For test/eval modes, store original dimensions for later reconstruction of submission
                if self.mode in ['test', 'eval']:
                    input_grid = example_data['input']
                    original_h = len(input_grid) if input_grid and isinstance(input_grid, list) else 0
                    original_w = len(input_grid[0]) if original_h > 0 and isinstance(input_grid[0], list) else 0
                    current_sample.update({'original_h': original_h, 'original_w': original_w})
                
                samples.append(current_sample)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict | tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        sample = self.samples[idx]
        
        # Use IMG_HEIGHT and IMG_WIDTH for consistent tensor sizes passed to the model's PatchEmbed layers
        input_tensor = grid_to_tensor(sample['input'], IMG_HEIGHT, IMG_WIDTH)
        
        if self.mode == 'vicreg_pretrain':
            output_tensor = grid_to_tensor(sample['output'], IMG_HEIGHT, IMG_WIDTH)
            # Concatenate along height dimension (dim=0 for 2D tensors before unsqueeze)
            concat_tensor = torch.cat((input_tensor, output_tensor), dim=0) # Shape: (CONCAT_IMG_HEIGHT, IMG_WIDTH)
            # Add channel dimension: (1, CONCAT_IMG_HEIGHT, IMG_WIDTH)
            return concat_tensor.unsqueeze(0).float() 

        elif self.mode == 'decoder_train':
            output_tensor = grid_to_tensor(sample['output'], IMG_HEIGHT, IMG_WIDTH)
            # Add channel dimension for both: (1, IMG_HEIGHT, IMG_WIDTH)
            return input_tensor.unsqueeze(0).float(), output_tensor.unsqueeze(0).float() 
        
        elif self.mode in ['test', 'eval']:
            # For model input during solving, it expects (B, C, H, W) = (1, 1, IMG_HEIGHT, IMG_WIDTH)
            # The solve_task function will handle the unsqueeze(0) for batch if DataLoader isn't used there.
            # If DataLoader is used for solving (batch_size=1), this is fine.
            return {
                'task_id': sample['task_id'],
                'input_tensor': input_tensor.unsqueeze(0).unsqueeze(0).float(), # (1,1,H,W) for model
                'original_input_grid': sample['input'], # Keep as list of lists for solve_task
                'original_h': sample['original_h'], 
                'original_w': sample['original_w'],
                'test_example_index': sample['example_idx'] # For submission file output_id
            }
        else:
            raise ValueError(f"Unknown dataset mode: {self.mode}")


# --- Training and Evaluation Functions ---

def train_one_epoch_unified(model: ARCVicregViT, dataloader: DataLoader, optimizer: optim.Optimizer, 
                            device: torch.device, epoch: int, training_stage_config: dict):
    """
    Unified training loop for one epoch, adaptable for VICReg or Decoder training.
    Args:
        model: The ARCVicregViT model.
        dataloader: PyTorch DataLoader for the current stage.
        optimizer: PyTorch optimizer.
        device: Device to train on (cuda or cpu).
        epoch: Current epoch number.
        training_stage_config: Dictionary containing stage-specific settings:
            'is_vicreg': bool,
            'pbar_desc': str (description for tqdm progress bar),
            'encoder_trainable': bool,
            'projector_trainable': bool,
            'decoder_trainable': bool
    Returns:
        Tuple of average losses for the epoch.
        For VICReg: (avg_total_loss, avg_sim_loss, avg_std_loss, avg_cov_loss)
        For Decoder: (avg_recon_loss,)
    """
    model.train() # Set model to training mode

    # Configure model trainability based on the current training stage
    model.set_encoder_trainable(training_stage_config['encoder_trainable'])
    model.set_projector_trainable(training_stage_config['projector_trainable'])
    model.set_decoder_trainable(training_stage_config['decoder_trainable'])

    # Accumulators for losses
    total_loss_agg, sim_loss_agg, std_loss_agg, cov_loss_agg = 0.0, 0.0, 0.0, 0.0
    batch_times, num_batches, skipped_batches = [], len(dataloader), 0
    
    pbar_desc = training_stage_config['pbar_desc']
    if num_batches == 0: 
        print(f"Warning: {pbar_desc} DataLoader empty for epoch {epoch+1}. Skipping.")
        return (0.0,0.0,0.0,0.0) if training_stage_config['is_vicreg'] else (0.0,)

    epoch_start_time = time.time()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} {pbar_desc}", total=num_batches, leave=True)
    
    for i, batch_data in enumerate(pbar):
        if device.type == 'cuda': torch.cuda.synchronize() # For accurate timing on GPU
        batch_start_time = time.time()
        optimizer.zero_grad(set_to_none=True) # More efficient way to zero gradients

        # Initialize loss variables for the current batch
        current_batch_total_loss = torch.tensor(0.0, device=device)
        current_batch_sim_loss, current_batch_std_loss, current_batch_cov_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        # --- Forward pass specific to training stage ---
        if training_stage_config['is_vicreg']:
            imgs_concat = batch_data.float().to(device) # Data is directly the concatenated image tensor
            current_batch_total_loss, current_batch_sim_loss, current_batch_std_loss, current_batch_cov_loss = \
                model.forward_pretrain(imgs_concat, batch_idx=i, epoch_idx=epoch)
        else: # Decoder training stage
            input_grids, target_output_grids = batch_data # Data is a tuple of (input_grid, target_grid)
            input_grids = input_grids.float().to(device)
            target_output_grids = target_output_grids.float().to(device)
            
            # Ensure target_output_grids has the channel dimension if it's missing (B,1,H,W)
            if target_output_grids.ndim == 3: # If (B,H,W), unsqueeze to (B,1,H,W)
                 target_output_grids = target_output_grids.unsqueeze(1)

            current_batch_total_loss, _ = model.forward_train_decoder(input_grids, target_output_grids)

        # --- Backward pass and optimization ---
        if not torch.isfinite(current_batch_total_loss):
            print(f"Warning: Non-finite loss at {pbar_desc} epoch {epoch+1} batch {i}: {current_batch_total_loss.item()}. Skipping batch.")
            skipped_batches += 1
            if device.type == 'cuda': torch.cuda.synchronize()
            batch_times.append(time.time() - batch_start_time); continue # Skip optimizer step
        
        current_batch_total_loss.backward()
        
        # Clip gradients for currently trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP_VALUE)
        
        optimizer.step()
        if device.type == 'cuda': torch.cuda.synchronize() # For accurate timing

        # --- Aggregate losses ---
        total_loss_agg += current_batch_total_loss.item()
        if training_stage_config['is_vicreg']:
            sim_loss_agg += current_batch_sim_loss.item()
            std_loss_agg += current_batch_std_loss.item()
            cov_loss_agg += current_batch_cov_loss.item()
        
        # --- Update progress bar ---
        valid_batches_so_far = i + 1 - skipped_batches
        avg_loss_so_far = total_loss_agg / valid_batches_so_far if valid_batches_so_far > 0 else 0.0
        
        postfix_dict = {'avg_loss': f'{avg_loss_so_far:.4f}'}
        if training_stage_config['is_vicreg']:
            postfix_dict.update({
                'sim': f'{sim_loss_agg/valid_batches_so_far if valid_batches_so_far > 0 else 0.0:.4f}',
                'std': f'{std_loss_agg/valid_batches_so_far if valid_batches_so_far > 0 else 0.0:.4f}',
                'cov': f'{cov_loss_agg/valid_batches_so_far if valid_batches_so_far > 0 else 0.0:.4f}'})
        if (i + 1) % PRINT_FREQ == 0 or i == num_batches -1 : # Print less frequently or at the end
            pbar.set_postfix(postfix_dict)
        
        batch_times.append(time.time() - batch_start_time)
    
    pbar.close() # Close progress bar for the epoch
    
    # --- Epoch Summary ---
    epoch_duration = time.time() - epoch_start_time
    avg_batch_time_ms = (np.mean(batch_times) * 1000) if batch_times else 0.0
    valid_batches_total = num_batches - skipped_batches
    avg_epoch_total_loss = total_loss_agg / valid_batches_total if valid_batches_total > 0 else 0.0
    
    print_msg = f"Epoch {epoch+1} Avg {pbar_desc} Loss: {avg_epoch_total_loss:.4f}"
    result_tuple: tuple = (avg_epoch_total_loss,) # Default result for decoder

    if training_stage_config['is_vicreg']:
        avg_epoch_sim_loss = sim_loss_agg / valid_batches_total if valid_batches_total > 0 else 0.0
        avg_epoch_std_loss = std_loss_agg / valid_batches_total if valid_batches_total > 0 else 0.0
        avg_epoch_cov_loss = cov_loss_agg / valid_batches_total if valid_batches_total > 0 else 0.0
        print_msg += f" (Sim: {avg_epoch_sim_loss:.4f}, Std: {avg_epoch_std_loss:.4f}, Cov: {avg_epoch_cov_loss:.4f})"
        result_tuple = (avg_epoch_total_loss, avg_epoch_sim_loss, avg_epoch_std_loss, avg_epoch_cov_loss)
        
    print(print_msg)
    print(f"Epoch {epoch+1} Duration: {epoch_duration:.2f}s, Avg Batch Time: {avg_batch_time_ms:.2f}ms")
    if skipped_batches > 0:
        print(f"Epoch {epoch+1} Skipped Batches: {skipped_batches} / {num_batches}")
    
    return result_tuple


def solve_task(model: ARCVicregViT, task_input_grid_list: list[list[int]], device: torch.device) -> list[list[int]]: 
    """Solves a single ARC task input grid using the trained model."""
    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad(): # Disable gradient calculations for inference
        original_h = len(task_input_grid_list) if task_input_grid_list else 0
        original_w = len(task_input_grid_list[0]) if original_h > 0 and task_input_grid_list[0] else 0
        
        # Convert list-of-lists grid to tensor (H,W)
        input_tensor_hw = grid_to_tensor(task_input_grid_list, IMG_HEIGHT, IMG_WIDTH)
        # Add batch and channel dimensions for model: (1, 1, H, W)
        input_tensor_bchw = input_tensor_hw.unsqueeze(0).unsqueeze(0).float().to(device)
        
        # Get model prediction (B, H, W)
        predicted_output_tensor_bhw = model.forward_solve(input_tensor_bchw)
        # Remove batch dimension: (H, W)
        predicted_output_tensor_hw = predicted_output_tensor_bhw.squeeze(0)
        
        # Convert tensor back to list-of-lists grid with original dimensions
        return tensor_to_grid(predicted_output_tensor_hw, original_h, original_w, IMG_HEIGHT, IMG_WIDTH)


# --- Main Execution Logic ---

def main(args: argparse.Namespace):
    """Main function to orchestrate pre-training, decoder training, and solving."""
    # Set global variables from args, e.g., for VICReg coefficients and debug prints
    global VICREG_SIM_COEFF_DEFAULT, VICREG_STD_COEFF_DEFAULT, VICREG_COV_COEFF_DEFAULT, DEBUG_PRINT_BATCH_ZERO
    VICREG_SIM_COEFF_DEFAULT = args.vicreg_sim_coeff
    VICREG_STD_COEFF_DEFAULT = args.vicreg_std_coeff
    VICREG_COV_COEFF_DEFAULT = args.vicreg_cov_coeff
    DEBUG_PRINT_BATCH_ZERO = args.debug_print_batch_zero # Set global debug flag

    print(f"--- Script Starting ---")
    print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print(f"Config:")
    print(f"  Model Save Path: {args.model_save_path}")
    print(f"  Submission File: {args.submission_file}")
    print(f"  Debug Batch 0 Prints: {DEBUG_PRINT_BATCH_ZERO}")
    print(f"  Mask Ratio for VICReg: {args.mask_ratio}")

    # Initialize the model
    model = ARCVicregViT(
        embed_dim=args.embed_dim, 
        encoder_depth=args.encoder_depth, 
        encoder_heads=args.encoder_heads,
        decoder_dim=args.decoder_dim, 
        num_colors=NUM_COLORS, 
        vicreg_projector_hidden_dim=args.vicreg_proj_hidden_dim,
        vicreg_projector_output_dim=args.vicreg_proj_output_dim, 
        mask_ratio=args.mask_ratio # Pass the mask_ratio from args to model constructor
    ).to(DEVICE)
    print(f"\n--- Model Initialized ---")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- VICReg Pre-training Phase ---
    if args.pretrain:
        print("\n--- Starting VICReg Pre-training Phase ---")
        if not Path(args.training_challenges).exists():
            print(f"ERROR: Training challenges file not found: {args.training_challenges}. Cannot pre-train.")
            return # Exit if data is missing
        try:
            dataset = ARCDataset(Path(args.training_challenges), 
                                 Path(args.training_solutions) if args.training_solutions else None, 
                                 mode='vicreg_pretrain')
            if len(dataset) == 0:
                print("ERROR: VICReg pre-training dataset is empty. Cannot pre-train."); return

            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, 
                                pin_memory=args.num_workers > 0 and DEVICE.type == 'cuda', 
                                persistent_workers=args.num_workers > 0 and args.num_workers > 0) # Check num_workers for persistent_workers
            
            # Optimizer for pre-training (only encoder and projector should be trainable)
            model.set_encoder_trainable(True)
            model.set_projector_trainable(True)
            model.set_decoder_trainable(False)
            opt_params = [p for p in model.parameters() if p.requires_grad]
            if not opt_params:
                print("ERROR: No trainable parameters found for VICReg pre-training."); return
            optimizer = optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
            
            vicreg_stage_config = {
                'is_vicreg': True, 'pbar_desc': "VICReg PreTrain",
                'encoder_trainable': True, 'projector_trainable': True, 'decoder_trainable': False
            }
            print(f"Running {args.epochs} VICReg pre-training epochs (LR: {args.lr}, Mask Ratio: {args.mask_ratio})...")
            for epoch in range(args.epochs):
                train_one_epoch_unified(model, loader, optimizer, DEVICE, epoch, vicreg_stage_config)
            
            Path(args.model_save_path).parent.mkdir(parents=True, exist_ok=True) # Ensure save directory exists
            torch.save(model.state_dict(), args.model_save_path)
            print(f"VICReg pre-training finished. Model saved to {args.model_save_path}")
        except Exception as e:
            print(f"Error during VICReg pre-training: {e}")
            traceback.print_exc() # Print full traceback for debugging

    # --- Decoder Training Phase ---
    if args.train_decoder:
        print("\n--- Starting Decoder Training Phase ---")
        # Load pre-trained model if not just pre-trained in this run
        if not args.pretrain and Path(args.model_save_path).exists():
            print(f"Loading model from {args.model_save_path} for decoder training...")
            try:
                model.load_state_dict(torch.load(args.model_save_path, map_location=DEVICE))
                print("Model loaded successfully for decoder training.")
            except Exception as e:
                print(f"Error loading model for decoder training: {e}. Training decoder from current model state (if any).")
        elif not args.pretrain and not Path(args.model_save_path).exists():
            print(f"Warning: Model file {args.model_save_path} not found. Decoder will train from scratch if encoder is also randomly initialized.")
        
        if not Path(args.training_challenges).exists():
            print(f"ERROR: Training challenges file not found: {args.training_challenges}. Cannot train decoder."); return
        try:
            dataset = ARCDataset(Path(args.training_challenges), 
                                 Path(args.training_solutions) if args.training_solutions else None, 
                                 mode='decoder_train')
            if len(dataset) == 0:
                print("ERROR: Decoder training dataset is empty. Cannot train decoder."); return
                
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                pin_memory=args.num_workers > 0 and DEVICE.type == 'cuda',
                                persistent_workers=args.num_workers > 0 and args.num_workers > 0)
            
            # Optimizer for decoder training (only decoder parts should be trainable)
            model.set_encoder_trainable(False)
            model.set_projector_trainable(False)
            model.set_decoder_trainable(True)
            opt_params = [p for p in model.parameters() if p.requires_grad]
            if not opt_params:
                print("ERROR: No trainable parameters found for the decoder. Check model.set_decoder_trainable()."); return
            optimizer = optim.AdamW(opt_params, lr=args.decoder_lr, weight_decay=args.weight_decay)

            decoder_stage_config = {
                'is_vicreg': False, 'pbar_desc': "Decoder Train",
                'encoder_trainable': False, 'projector_trainable': False, 'decoder_trainable': True
            }
            print(f"Running {args.decoder_epochs} Decoder training epochs (LR: {args.decoder_lr})...")
            for epoch in range(args.decoder_epochs):
                train_one_epoch_unified(model, loader, optimizer, DEVICE, epoch, decoder_stage_config)
            
            Path(args.model_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.model_save_path) # Save updated model (with trained decoder)
            print(f"Decoder training finished. Model updated and saved to {args.model_save_path}")
        except Exception as e:
            print(f"Error during Decoder training: {e}")
            traceback.print_exc()
    
    # --- Solving Phase (Evaluation or Test) ---
    solve_file_path_str: str | None = None
    solve_mode_str: str | None = None # To pass to ARCDataset

    if args.solve_eval:
        if Path(args.eval_challenges).exists():
            solve_file_path_str = str(args.eval_challenges)
            solve_mode_str = 'eval'
        else:
            print(f"Warning: --solve_eval specified, but evaluation file not found: {args.eval_challenges}")
    elif args.solve_test:
        if Path(args.test_challenges).exists():
            solve_file_path_str = str(args.test_challenges)
            solve_mode_str = 'test'
        else:
            print(f"Warning: --solve_test specified, but test file not found: {args.test_challenges}")
    elif not (args.pretrain or args.train_decoder): # Default to solving test set if no training actions were specified
        if Path(args.test_challenges).exists():
            solve_file_path_str = str(args.test_challenges)
            solve_mode_str = 'test'
        else:
            print(f"Default solve action: Test file {args.test_challenges} not found. Skipping submission.")

    if solve_file_path_str and solve_mode_str:
        print(f"\n--- Starting Solving Phase for '{solve_mode_str}' set from {solve_file_path_str} ---")
        if not Path(args.model_save_path).exists():
            print(f"ERROR: Model file {args.model_save_path} not found. Cannot solve."); return
        
        print(f"Loading model from {args.model_save_path} for solving...")
        try:
            model.load_state_dict(torch.load(args.model_save_path, map_location=DEVICE))
            print("Model loaded successfully for solving.")
            
            # Pass the determined solve_mode_str to ARCDataset
            dataset = ARCDataset(Path(solve_file_path_str), solutions_file=None, mode=solve_mode_str)
            if not dataset or len(dataset) == 0:
                print(f"ERROR: {solve_mode_str.capitalize()} dataset is empty. Cannot solve."); return
            
            submission_dict = {} # Use a more descriptive name
            # Using DataLoader for solving can be convenient, especially if num_workers > 0
            solve_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers) 
            
            for item_data in tqdm(solve_loader, desc=f"Solving {solve_mode_str.capitalize()}"):
                task_id = item_data['task_id'][0] # DataLoader wraps strings in lists for B=1
                
                # 'original_input_grid' from ARCDataset is already a list of lists.
                # DataLoader with B=1 might wrap it again.
                original_input_grid = item_data['original_input_grid']
                if isinstance(original_input_grid, list) and len(original_input_grid) == 1 and \
                   isinstance(original_input_grid[0], list) and isinstance(original_input_grid[0][0], list):
                     original_input_grid = original_input_grid[0] # Unwrap if default_collate added an extra list layer
                
                predicted_grid = solve_task(model, original_input_grid, DEVICE)
                
                if task_id not in submission_dict:
                    submission_dict[task_id] = []
                
                submission_dict[task_id].append({
                    "output_id": item_data['test_example_index'].item(), # .item() to get Python number
                    "prediction_counts": 1, # Assuming one prediction attempt per test case   
                    "predictions": [{"prediction_id": 0, "output": predicted_grid}]
                })
            
            Path(args.submission_file).parent.mkdir(parents=True, exist_ok=True)
            with open(args.submission_file, 'w') as f:
                json.dump(submission_dict, f, indent=2)
            print(f"SUCCESS: Submission file saved to {args.submission_file}")
        except Exception as e:
            print(f"Error during solving: {e}")
            traceback.print_exc()
    elif not (args.pretrain or args.train_decoder or args.solve_eval or args.solve_test):
        # This condition means no action was specified or possible.
        print("\n--- No action (pretrain, train_decoder, solve_eval, solve_test) specified, or required files missing. ---")
    
    print("\n--- Script Finished ---")

def is_notebook() -> bool:
    """Checks if the script is running in a notebook environment (Jupyter, Colab)."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell': # Jupyter notebook or qtconsole
            return True   
        elif shell == 'TerminalInteractiveShell': # Terminal IPython
            return False  
        elif 'google.colab' in sys.modules: # Google Colab
            return True
        return False # Other unknown IPython shells
    except (NameError, ImportError):
        return False # Not in IPython environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ARC Prize Solver: ViT with VICReg Pre-training and Decoder Fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )

    # File Path Arguments
    path_group = parser.add_argument_group('File Paths')
    path_group.add_argument('--training_challenges', type=str, default=str(TRAINING_CHALLENGES_FILE_DEFAULT), help="Path to training challenges JSON file.")
    path_group.add_argument('--training_solutions', type=str, default=str(TRAINING_SOLUTIONS_FILE_DEFAULT), help="Path to training solutions JSON file.")
    path_group.add_argument('--eval_challenges', type=str, default=str(EVALUATION_CHALLENGES_FILE_DEFAULT), help="Path to evaluation challenges JSON file.")
    path_group.add_argument('--test_challenges', type=str, default=str(TEST_CHALLENGES_FILE_DEFAULT), help="Path to test challenges JSON file.")
    path_group.add_argument('--submission_file', type=str, default=str(SUBMISSION_FILE_DEFAULT), help="Path to save the submission JSON file.")
    path_group.add_argument('--model_save_path', type=str, default=str(MODEL_SAVE_PATH_DEFAULT), help="Path to save/load the trained model.")

    # Action Arguments
    action_group = parser.add_argument_group('Actions')
    action_group.add_argument('--pretrain', action='store_true', help="Run VICReg pre-training for the encoder.")
    action_group.add_argument('--train_decoder', action='store_true', help="Run supervised training for the decoder.")
    action_group.add_argument('--solve_eval', action='store_true', help="Generate submission for the evaluation set.")
    action_group.add_argument('--solve_test', action='store_true', help="Generate submission for the test set.")

    # Training Hyperparameter Arguments
    train_hp_group = parser.add_argument_group('Training Hyperparameters')
    train_hp_group.add_argument('--epochs', type=int, default=EPOCHS_DEFAULT, help="Epochs for VICReg pre-training.")
    train_hp_group.add_argument('--lr', type=float, default=LR_DEFAULT, help="Learning rate for VICReg pre-training.")
    train_hp_group.add_argument('--decoder_epochs', type=int, default=DECODER_EPOCHS_DEFAULT, help="Epochs for decoder training.")
    train_hp_group.add_argument('--decoder_lr', type=float, default=DECODER_LR_DEFAULT, help="Learning rate for decoder training.")
    train_hp_group.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT, help="Batch size for training and data loading.")
    train_hp_group.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY_DEFAULT, help="Weight decay for optimizers.")
    train_hp_group.add_argument('--num_workers', type=int, default=0, help="Number of worker processes for DataLoader.")

    # Model Hyperparameter Arguments
    model_hp_group = parser.add_argument_group('Model Hyperparameters')
    model_hp_group.add_argument('--embed_dim', type=int, default=EMBED_DIM_DEFAULT, help="Embedding dimension for ViT.")
    model_hp_group.add_argument('--encoder_depth', type=int, default=ENCODER_DEPTH_DEFAULT, help="Number of Transformer blocks in the encoder.")
    model_hp_group.add_argument('--encoder_heads', type=int, default=ENCODER_HEADS_DEFAULT, help="Number of attention heads in the encoder.")
    model_hp_group.add_argument('--decoder_dim', type=int, default=DECODER_DIM_DEFAULT, help="Embedding dimension for the decoder.")
    model_hp_group.add_argument('--mask_ratio', type=float, default=MASK_RATIO_DEFAULT, help="Patch mask ratio for VICReg pre-training views.")
    model_hp_group.add_argument('--vicreg_proj_hidden_dim', type=int, default=VICREG_PROJECTOR_HIDDEN_DIM_DEFAULT, help="Hidden dimension of VICReg projector.")
    model_hp_group.add_argument('--vicreg_proj_output_dim', type=int, default=VICREG_PROJECTOR_OUTPUT_DIM_DEFAULT, help="Output dimension of VICReg projector.")

    # VICReg Coefficient Arguments
    vicreg_coeff_group = parser.add_argument_group('VICReg Coefficients')
    vicreg_coeff_group.add_argument('--vicreg_sim_coeff', type=float, default=VICREG_SIM_COEFF_DEFAULT, help="VICReg similarity loss coefficient.")
    vicreg_coeff_group.add_argument('--vicreg_std_coeff', type=float, default=VICREG_STD_COEFF_DEFAULT, help="VICReg standard deviation loss coefficient.")
    vicreg_coeff_group.add_argument('--vicreg_cov_coeff', type=float, default=VICREG_COV_COEFF_DEFAULT, help="VICReg covariance loss coefficient.")

    # Debugging Arguments
    debug_group = parser.add_argument_group('Debugging')
    debug_group.add_argument('--debug_print_batch_zero', action='store_true', default=DEBUG_PRINT_BATCH_ZERO_DEFAULT, help="Enable detailed debug prints for batch 0 of epoch 0 during pre-training.")
    debug_group.add_argument('--no_debug_print_batch_zero', action='store_false', dest='debug_print_batch_zero', help="Disable detailed debug prints for batch 0 of epoch 0.")

    # Parse arguments
    if is_notebook():
        print("Running in Notebook environment. Using default args or those modifiable in a notebook cell.")
        # For notebooks, parse known args and ignore others like -f from Jupyter.
        # Provide an empty list to parse_known_args if sys.argv is not what we want (e.g. in Colab/Jupyter).
        args, unknown_args = parser.parse_known_args([]) 
        # Example: Override specific args for notebook testing if needed
        # args.epochs = 1 
        # args.pretrain = True 
        # args.debug_print_batch_zero = True # Often want this enabled for notebook debugging
        if unknown_args:
            # Filter out common notebook kernel arguments before printing warnings
            known_notebook_kernel_args = ['-f', '/root/.local', '/kernel-', '.json'] 
            filtered_unknown = [arg for arg in unknown_args if not any(known_part in arg for known_part in known_notebook_kernel_args)]
            if filtered_unknown:
                print(f"Ignoring unknown notebook args: {filtered_unknown}")
    else:
        args = parser.parse_args()
    
    main(args)

