"""
src/utils/signal_utils.py
--------------------------
Pairwise similarity and distance signal computation utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import cv2
import numpy as np

if TYPE_CHECKING:
    from src.features.extractor import AdFeature


def phash_distance(f1: AdFeature, f2: AdFeature) -> int:
    """Hamming distance between perceptual hashes (0 = identical)."""
    return f1.phash - f2.phash


def color_similarity(f1: AdFeature, f2: AdFeature) -> float:
    """HSV histogram correlation (1.0 = identical palette)."""
    return float(cv2.compareHist(f1.hist, f2.hist, cv2.HISTCMP_CORREL))


def visual_similarity(f1: AdFeature, f2: AdFeature) -> float:
    """Cosine similarity between L2-normalised visual feature vectors."""
    return float(np.dot(f1.visual_emb, f2.visual_emb))


def text_similarity(f1: AdFeature, f2: AdFeature, text_model_type: str = "sentence_transformer") -> float:
    """
    Compute text similarity using SentenceTransformer cosine sim or Lexical Jaccard.
    """
    if text_model_type.lower() == "jaccard":
        if not f1.tokens or not f2.tokens:
            return 0.0
        intersection = len(f1.tokens & f2.tokens)
        union = len(f1.tokens | f2.tokens)
        return intersection / union if union > 0 else 0.0
    else:
        if not f1.text or not f2.text:
            return 0.0
        if f1.text_emb is None or f2.text_emb is None:
            return 0.0
        return float(np.dot(f1.text_emb, f2.text_emb))


def dims_match(f1: AdFeature, f2: AdFeature) -> bool:
    """True if both ads share IAB dimensions or raw pixel dimensions."""
    if f1.dimensions is not None and f2.dimensions is not None:
        return f1.dimensions == f2.dimensions
    return f1.raw_image.size == f2.raw_image.size


def compute_all_signals(
    f1: AdFeature,
    f2: AdFeature,
    text_model_type: str = "sentence_transformer",
    visual_model_type: str = "clip",
) -> dict:
    """
    Compute all pairwise signals between two ad features.
    """
    v_sim = visual_similarity(f1, f2)
    t_sim = text_similarity(f1, f2, text_model_type=text_model_type)

    signals = {
        "phash_dist": phash_distance(f1, f2),
        "color_sim":  color_similarity(f1, f2),
        "text_sim":   t_sim,
        "visual_sim": v_sim,
        "dims_match": dims_match(f1, f2),
    }

    if "resnet" in visual_model_type.lower():
        signals["resnet_sim"] = v_sim
    elif "dino" in visual_model_type.lower():
        signals["dino_sim"] = v_sim
    else:
        signals["clip_sim"] = v_sim

    return signals

