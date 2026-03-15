"""
Day 155 - Introduction to Computer Vision
==========================================
Core image preprocessing pipeline used in production CV systems.
"""

import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


# ── 1. Image Loading and Inspection ───────────────────────────────────────────

def load_and_inspect(image_path: str) -> dict:
    """
    Load an image and return its raw properties.
    Mimics the acquisition stage of a production CV pipeline.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"OpenCV could not decode: {image_path}")

    return {
        "shape_hwc": img_bgr.shape,           # (H, W, C)
        "height": img_bgr.shape[0],
        "width": img_bgr.shape[1],
        "channels": img_bgr.shape[2] if img_bgr.ndim == 3 else 1,
        "dtype": str(img_bgr.dtype),
        "pixel_min": int(img_bgr.min()),
        "pixel_max": int(img_bgr.max()),
        "pixel_mean": float(img_bgr.mean()),
        "size_bytes": img_bgr.nbytes,
    }


# ── 2. Color Space Conversion ──────────────────────────────────────────────────

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """
    OpenCV loads images as BGR. All pretrained PyTorch models expect RGB.
    Forgetting this swap is the #1 silent bug in new CV pipelines.
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """Convert to single-channel grayscale."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ── 3. ImageNet Preprocessing Pipeline ────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_inference_transform(size: int = 224) -> transforms.Compose:
    """
    Standard preprocessing pipeline for ImageNet-pretrained models.
    This is the exact transform used before feeding images to ResNet,
    EfficientNet, ViT, and most other pretrained architectures.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),          # uint8 HxWxC  ->  float32 CxHxW  (/255)
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image_path: str, size: int = 224) -> torch.Tensor:
    """
    Full preprocessing pipeline: load -> RGB -> PIL -> tensor.
    Returns a float32 tensor of shape [3, size, size].
    """
    img_bgr = cv2.imread(image_path)
    img_rgb = bgr_to_rgb(img_bgr)
    pil_img = Image.fromarray(img_rgb)
    transform = build_inference_transform(size)
    return transform(pil_img)


def preprocess_batch(image_paths: list, size: int = 224) -> torch.Tensor:
    """
    Preprocess multiple images into a batched tensor [N, 3, H, W].
    Production systems use DataLoader workers for this; this is the
    single-threaded equivalent for learning purposes.
    """
    tensors = [preprocess_image(p, size) for p in image_paths]
    return torch.stack(tensors)


# ── 4. Inverse Transform (for visualization) ──────────────────────────────────

def tensor_to_displayable(tensor: torch.Tensor) -> np.ndarray:
    """
    Undo ImageNet normalization and convert back to uint8 HxWxC numpy array.
    Only for visualization — never pass this through a model.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor * std + mean).clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# ── 5. Basic Image Statistics ──────────────────────────────────────────────────

def channel_statistics(tensor: torch.Tensor) -> dict:
    """
    Per-channel mean and std of a preprocessed tensor.
    Useful for verifying normalization was applied correctly.
    Expected range after ImageNet normalization: approximately [-2.1, 2.6].
    """
    stats = {}
    channel_names = ["R", "G", "B"]
    for i, name in enumerate(channel_names):
        ch = tensor[i]
        stats[name] = {
            "mean": float(ch.mean()),
            "std":  float(ch.std()),
            "min":  float(ch.min()),
            "max":  float(ch.max()),
        }
    return stats


# ── 6. Visualization ───────────────────────────────────────────────────────────

def visualize_pipeline(image_path: str, output_path: str = "pipeline_demo.png"):
    """
    Four-panel visualization: raw BGR | RGB | grayscale | preprocessed tensor.
    """
    img_bgr  = cv2.imread(image_path)
    img_rgb  = bgr_to_rgb(img_bgr)
    img_gray = to_grayscale(img_bgr)
    tensor   = preprocess_image(image_path)
    img_disp = tensor_to_displayable(tensor)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), "Raw (BGR→display RGB)"),
        (img_rgb, "Corrected RGB"),
        (img_gray, "Grayscale"),
        (img_disp, "After ImageNet Normalization\n(undo for display)"),
    ]
    for ax, (data, title) in zip(axes, panels):
        if data.ndim == 2:
            ax.imshow(data, cmap="gray")
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Pipeline visualization saved: {output_path}")
    return output_path


# ── Main Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "sample.png"
    if not os.path.exists(sample):
        print(f"Sample image '{sample}' not found. Run setup.sh first.")
        exit(1)

    print("\n=== Stage 1: Acquisition & Inspection ===")
    info = load_and_inspect(sample)
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n=== Stage 2: Preprocessing ===")
    tensor = preprocess_image(sample)
    print(f"  Output tensor shape : {tensor.shape}")
    print(f"  Output tensor dtype : {tensor.dtype}")

    print("\n=== Stage 3: Channel Statistics ===")
    stats = channel_statistics(tensor)
    for ch, s in stats.items():
        print(f"  {ch} → mean={s['mean']:.4f}  std={s['std']:.4f}  "
              f"range=[{s['min']:.4f}, {s['max']:.4f}]")

    print("\n=== Stage 4: Batch Demo ===")
    batch = preprocess_batch([sample, sample])
    print(f"  Batch shape: {batch.shape}")

    print("\n=== Stage 5: Visualization ===")
    visualize_pipeline(sample)
    print("\nAll stages complete.")
