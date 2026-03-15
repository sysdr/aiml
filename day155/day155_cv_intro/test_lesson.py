"""
Day 155 Test Suite - Introduction to Computer Vision
=====================================================
25 tests covering all pipeline stages.
Run: pytest test_lesson.py -v
"""

import os
import numpy as np
import pytest
import cv2
import torch
from PIL import Image
from lesson_code import (
    load_and_inspect,
    bgr_to_rgb,
    to_grayscale,
    build_inference_transform,
    preprocess_image,
    preprocess_batch,
    tensor_to_displayable,
    channel_statistics,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

SAMPLE = "sample.png"


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_image(tmp_path_factory):
    """Create a synthetic 100x100 RGB test image."""
    path = tmp_path_factory.mktemp("data") / "test_img.png"
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:50, :50] = [255, 0, 0]    # red quadrant (BGR: blue channel = 255)
    arr[:50, 50:] = [0, 255, 0]    # green quadrant
    arr[50:, :50] = [0, 0, 255]    # blue quadrant
    arr[50:, 50:] = [128, 128, 128]  # gray quadrant
    cv2.imwrite(str(path), arr)
    return str(path)


# ── Class 1: Acquisition ───────────────────────────────────────────────────────

class TestAcquisition:
    def test_load_returns_dict(self, sample_image):
        result = load_and_inspect(sample_image)
        assert isinstance(result, dict)

    def test_shape_keys_present(self, sample_image):
        result = load_and_inspect(sample_image)
        assert "height" in result and "width" in result and "channels" in result

    def test_correct_dimensions(self, sample_image):
        result = load_and_inspect(sample_image)
        assert result["height"] == 100
        assert result["width"] == 100
        assert result["channels"] == 3

    def test_pixel_range_uint8(self, sample_image):
        result = load_and_inspect(sample_image)
        assert result["pixel_min"] >= 0
        assert result["pixel_max"] <= 255

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_and_inspect("nonexistent_file.png")

    def test_dtype_reported(self, sample_image):
        result = load_and_inspect(sample_image)
        assert result["dtype"] == "uint8"


# ── Class 2: Color Space Conversion ───────────────────────────────────────────

class TestColorConversion:
    def test_bgr_to_rgb_shape_preserved(self, sample_image):
        img = cv2.imread(sample_image)
        out = bgr_to_rgb(img)
        assert out.shape == img.shape

    def test_bgr_to_rgb_channels_swapped(self, sample_image):
        img = cv2.imread(sample_image)
        out = bgr_to_rgb(img)
        # Channel 0 and 2 should be swapped
        assert np.array_equal(out[:, :, 0], img[:, :, 2])
        assert np.array_equal(out[:, :, 2], img[:, :, 0])

    def test_grayscale_output_2d(self, sample_image):
        img = cv2.imread(sample_image)
        gray = to_grayscale(img)
        assert gray.ndim == 2

    def test_grayscale_dtype_preserved(self, sample_image):
        img = cv2.imread(sample_image)
        gray = to_grayscale(img)
        assert gray.dtype == np.uint8

    def test_rgb_values_in_expected_range(self, sample_image):
        img = cv2.imread(sample_image)
        out = bgr_to_rgb(img)
        assert out.min() >= 0 and out.max() <= 255


# ── Class 3: Preprocessing Pipeline ───────────────────────────────────────────

class TestPreprocessing:
    def test_output_is_tensor(self, sample_image):
        tensor = preprocess_image(sample_image)
        assert isinstance(tensor, torch.Tensor)

    def test_output_shape_default(self, sample_image):
        tensor = preprocess_image(sample_image)
        assert tensor.shape == torch.Size([3, 224, 224])

    def test_output_dtype_float32(self, sample_image):
        tensor = preprocess_image(sample_image)
        assert tensor.dtype == torch.float32

    def test_custom_size(self, sample_image):
        tensor = preprocess_image(sample_image, size=128)
        assert tensor.shape == torch.Size([3, 128, 128])

    def test_values_approximately_normalized(self, sample_image):
        tensor = preprocess_image(sample_image)
        # After ImageNet normalization, values should not be in [0, 1] raw range
        assert tensor.min() < 0 or tensor.max() > 1  # normalization shifted the values

    def test_channel_first_ordering(self, sample_image):
        tensor = preprocess_image(sample_image)
        # CxHxW — channel dim should be 3
        assert tensor.shape[0] == 3


# ── Class 4: Batch Processing ──────────────────────────────────────────────────

class TestBatchProcessing:
    def test_batch_shape(self, sample_image):
        batch = preprocess_batch([sample_image, sample_image])
        assert batch.shape == torch.Size([2, 3, 224, 224])

    def test_single_item_batch(self, sample_image):
        batch = preprocess_batch([sample_image])
        assert batch.shape[0] == 1

    def test_batch_dtype(self, sample_image):
        batch = preprocess_batch([sample_image, sample_image])
        assert batch.dtype == torch.float32

    def test_identical_images_identical_tensors(self, sample_image):
        batch = preprocess_batch([sample_image, sample_image])
        assert torch.allclose(batch[0], batch[1])


# ── Class 5: Statistics and Inverse Transform ─────────────────────────────────

class TestStatisticsAndInverse:
    def test_channel_stats_keys(self, sample_image):
        tensor = preprocess_image(sample_image)
        stats = channel_statistics(tensor)
        assert set(stats.keys()) == {"R", "G", "B"}

    def test_channel_stats_has_mean(self, sample_image):
        tensor = preprocess_image(sample_image)
        stats = channel_statistics(tensor)
        assert "mean" in stats["R"]

    def test_inverse_transform_shape(self, sample_image):
        tensor = preprocess_image(sample_image)
        display = tensor_to_displayable(tensor)
        assert display.shape == (224, 224, 3)

    def test_inverse_transform_dtype(self, sample_image):
        tensor = preprocess_image(sample_image)
        display = tensor_to_displayable(tensor)
        assert display.dtype == np.uint8

    def test_inverse_transform_value_range(self, sample_image):
        tensor = preprocess_image(sample_image)
        display = tensor_to_displayable(tensor)
        assert display.min() >= 0 and display.max() <= 255
