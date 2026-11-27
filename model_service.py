"""Inference service for the trained Siamese Network V4 model."""

from __future__ import annotations

import gc
import io
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Configure CPU-only, low-overhead TensorFlow execution before importing TF/Keras.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

TARGET_WIDTH = 32
TARGET_HEIGHT = 32
TARGET_CHANNELS = 3
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF"}


class InvalidImageError(ValueError):
    """Raised when an uploaded file cannot be decoded as a supported image."""


@dataclass(frozen=True)
class ComparisonResult:
    """Structured result returned by the Siamese V4 comparison."""

    distance: float
    threshold: float
    is_similar: bool
    margin: float
    cosine_proximity: float
    inference_ms: float
    image1_size: tuple[int, int]
    image2_size: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance": round(self.distance, 6),
            "threshold": round(self.threshold, 6),
            "is_similar": self.is_similar,
            "verdict": "Similar" if self.is_similar else "Different",
            "margin": round(self.margin, 6),
            "cosine_proximity": round(self.cosine_proximity, 6),
            "inference_ms": round(self.inference_ms, 2),
            "model": {
                "name": "Siamese Network V4",
                "input_shape": [TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS],
                "image1_original_size": list(self.image1_size),
                "image2_original_size": list(self.image2_size),
            },
        }


def _load_siamese_model(model_path: Path) -> keras.Model:
    """Load siamese_v4.keras without modifying the saved model artifact.

    The saved `lambda_3` layer omits `output_shape` and stores Python 3.13
    bytecode for Euclidean distance `tf.sqrt(tf.reduce_sum(tf.square(t[0] - t[1]), axis=1, keepdims=True) + 1e-7)`.
    Providing an equivalent `keras.ops` Lambda with `output_shape` during
    deserialization allows Keras 3 to reconstruct the exact graph and load
    all trained weights from `model.weights.h5`.
    """
    orig_from_config = keras.layers.Lambda.from_config

    @classmethod  # type: ignore[misc]
    def _compatible_lambda_from_config(
        cls: type[keras.layers.Lambda],
        config: dict[str, Any],
        custom_objects: dict[str, Any] | None = None,
        safe_mode: bool | None = None,
    ) -> keras.layers.Lambda:
        if config.get("name") == "lambda_3":
            return keras.layers.Lambda(
                lambda t: keras.ops.sqrt(
                    keras.ops.sum(keras.ops.square(t[0] - t[1]), axis=1, keepdims=True) + 1e-7
                ),
                output_shape=lambda shapes: (shapes[0][0], 1),
                name="lambda_3",
                trainable=config.get("trainable", True),
            )
        return orig_from_config.__func__(  # type: ignore[attr-defined]
            cls, config, custom_objects=custom_objects, safe_mode=safe_mode
        )

    keras.layers.Lambda.from_config = _compatible_lambda_from_config  # type: ignore[method-assign]
    try:
        return keras.models.load_model(str(model_path), compile=False, safe_mode=False)
    finally:
        keras.layers.Lambda.from_config = orig_from_config  # type: ignore[method-assign]


def _load_threshold(threshold_path: Path) -> float:
    """Load the scalar similarity threshold from a .npy file."""
    raw = np.load(str(threshold_path), allow_pickle=False)
    value = float(np.asarray(raw).item())
    if not np.isfinite(value):
        raise ValueError(f"Invalid non-finite threshold value loaded from {threshold_path}")
    return value


def preprocess_image_bytes(image_bytes: bytes, label: str = "Image") -> tuple[np.ndarray, tuple[int, int]]:
    """Decode raw image bytes and preprocess to (1, 32, 32, 3) float32 in [0, 1]."""
    if not image_bytes:
        raise InvalidImageError(f"{label} is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img_format = (img.format or "").upper()
            if img_format not in ALLOWED_IMAGE_FORMATS:
                raise InvalidImageError(
                    f"{label} has unsupported format '{img_format or 'unknown'}'. "
                    f"Supported formats: {', '.join(sorted(ALLOWED_IMAGE_FORMATS))}."
                )

            img.load()
            img = ImageOps.exif_transpose(img)
            original_size = img.size

            if img.mode != "RGB":
                img = img.convert("RGB")

            resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"{label} is not a valid or readable image file.") from exc

    if arr.shape != (TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS):
        raise InvalidImageError(
            f"{label} produced unexpected tensor shape {arr.shape}; expected "
            f"({TARGET_HEIGHT}, {TARGET_WIDTH}, {TARGET_CHANNELS})."
        )

    return np.expand_dims(arr, axis=0), original_size


class SiameseSimilarityService:
    """Thread-safe wrapper around the loaded Siamese V4 model and threshold."""

    def __init__(self, model_path: Path, threshold_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not threshold_path.is_file():
            raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

        self.model_path = model_path
        self.threshold_path = threshold_path

        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            # TensorFlow runtime already initialized in this process.
            pass

        self.model = _load_siamese_model(model_path)
        self.threshold = _load_threshold(threshold_path)
        self._lock = threading.Lock()

        # Warm up TensorFlow graph execution once on startup, then reclaim temp buffers.
        dummy = np.zeros((1, TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS), dtype=np.float32)
        with self._lock:
            _ = self.model([dummy, dummy], training=False)
        gc.collect()

    def is_ready(self) -> bool:
        """Return True when the model and finite threshold are loaded."""
        return self.model is not None and np.isfinite(self.threshold)

    def compare_bytes(self, image1_bytes: bytes, image2_bytes: bytes) -> ComparisonResult:
        """Preprocess both images and run Siamese model inference."""
        tensor1, size1 = preprocess_image_bytes(image1_bytes, label="Image 1")
        tensor2, size2 = preprocess_image_bytes(image2_bytes, label="Image 2")

        start = time.perf_counter()
        with self._lock:
            # Model expects [input_layer_10, input_layer_11] in exact order.
            output = self.model([tensor1, tensor2], training=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        distance = float(np.asarray(output).reshape(-1)[0])
        is_similar = bool(distance <= self.threshold)
        margin = float(self.threshold - distance)
        # Embeddings are L2 unit-normalized, so d^2 = ||u - v||^2 = 2 - 2*cos(theta) (+ 1e-7).
        cosine_proximity = float(np.clip(1.0 - (distance * distance) / 2.0, -1.0, 1.0))

        return ComparisonResult(
            distance=distance,
            threshold=self.threshold,
            is_similar=is_similar,
            margin=margin,
            cosine_proximity=cosine_proximity,
            inference_ms=elapsed_ms,
            image1_size=size1,
            image2_size=size2,
        )
