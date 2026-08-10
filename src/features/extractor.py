"""
src/features/extractor.py
--------------------------
Unified AdFeatureExtractor & AdFeature data container.
Delegates model inference to src.utils.model_utils.
"""

from __future__ import annotations

import os
import pickle

import cv2
import numpy as np
import imagehash
import torch
from PIL import Image

from src.utils.model_utils import (
    load_visual_model,
    get_visual_embedding,
    load_text_model,
    extract_text_representation,
)


# ---------------------------------------------------------------------------
# Unified AdFeature
# ---------------------------------------------------------------------------
class AdFeature:
    """
    Immutable container for per-ad extracted features.
    """

    __slots__ = (
        "index", "raw_image", "text", "dimensions",
        "phash", "hist", "visual_emb", "text_emb", "tokens",
    )

    def __repr__(self) -> str:
        return f"AdFeature(index={self.index}, dims={self.dimensions}, text_len={len(self.text)})"

    @property
    def clip_emb(self) -> np.ndarray:
        return self.visual_emb

    @property
    def resnet_emb(self) -> np.ndarray:
        return self.visual_emb


# ---------------------------------------------------------------------------
# Unified AdFeatureExtractor
# ---------------------------------------------------------------------------
class AdFeatureExtractor:
    """
    Unified Feature Extractor supporting any visual model (CLIP, ResNet-18)
    and text representation (SentenceTransformer, Lexical Jaccard).

    Parameters
    ----------
    visual_model : str
        Visual model identifier: "openai/clip-vit-base-patch32" or "resnet18".
    text_model : str
        Text model identifier: "all-MiniLM-L6-v2" or "jaccard".
    device : str
        Compute device: "auto", "cpu", "mps", or "cuda".
    cache_dir : str
        Directory for pickle feature caches.
    """

    def __init__(
        self,
        visual_model: str = "openai/clip-vit-base-patch32",
        text_model: str = "all-MiniLM-L6-v2",
        device: str = "auto",
        cache_dir: str = "cache",
    ) -> None:
        if device == "auto":
            device = (
                "mps" if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        self.device = device
        self.visual_model_name = visual_model
        self.text_model_name = text_model
        self.cache_dir = cache_dir

        self._visual_model = None
        self._visual_proc = None
        self._text_model = None

    def _init_models(self):
        if self._visual_model is None:
            self._visual_model, self._visual_proc = load_visual_model(
                self.visual_model_name, self.device
            )
        if self._text_model is None and self.text_model_name.lower() != "jaccard":
            self._text_model = load_text_model(self.text_model_name, self.device)

    def extract_one(self, ad_item: dict, index: int = -1) -> AdFeature:
        """Extract all features for a single ad dictionary."""
        self._init_models()

        feat = AdFeature.__new__(AdFeature)
        feat.index = index

        # Image
        feat.raw_image = ad_item["image"].convert("RGBA").convert("RGB")

        # Text
        feat.text = (ad_item.get("text") or "").strip().lower()

        # Dimensions
        dims_raw = ad_item.get("dimensions")
        feat.dimensions = None
        if dims_raw and "x" in str(dims_raw).lower():
            try:
                w, h = str(dims_raw).lower().split("x")
                feat.dimensions = (int(w.strip()), int(h.strip()))
            except ValueError:
                pass

        # Perceptual hash & Color Histogram
        feat.phash = imagehash.phash(feat.raw_image)
        cv_img = np.array(feat.raw_image)[:, :, ::-1]
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        feat.hist = cv2.normalize(hist, hist).flatten()

        # Visual embedding
        feat.visual_emb = get_visual_embedding(
            self.visual_model_name,
            self._visual_model,
            self._visual_proc,
            feat.raw_image,
            self.device,
        )

        # Text representation
        feat.text_emb, feat.tokens = extract_text_representation(
            self.text_model_name,
            self._text_model,
            feat.text,
        )

        return feat

    def extract(
        self,
        dataset,
        sample_size: int | None = None,
    ) -> list[AdFeature]:
        """
        Extract features for an entire dataset split with pickle caching.
        """
        from tqdm import tqdm

        n_str = str(sample_size) if sample_size else "all"
        vis_tag = "clip" if "clip" in self.visual_model_name.lower() else "resnet"
        cache_path = os.path.join(self.cache_dir, f"features_{vis_tag}_{n_str}.pkl") if self.cache_dir else None

        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path} ...")
            with open(cache_path, "rb") as fh:
                features = pickle.load(fh)
            print(f"  Loaded {len(features)} cached features.")
            return features

        if sample_size and sample_size < len(dataset):
            dataset = dataset.select(range(sample_size))

        print(f"Extracting features for {len(dataset)} ads ({self.visual_model_name} + {self.text_model_name})...")
        features: list[AdFeature] = []
        errors = 0

        for idx in tqdm(range(len(dataset)), desc="Feature extraction", unit="ad"):
            try:
                features.append(self.extract_one(dataset[idx], index=idx))
            except Exception as exc:
                print(f"\n  [WARN] Skipped ad #{idx}: {exc}")
                errors += 1

        print(f"  Done. {len(features)} extracted, {errors} skipped.")

        if cache_path:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(features, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Features cached → {cache_path}")

        return features
