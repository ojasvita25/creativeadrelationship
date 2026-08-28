"""
src/classifier/classifier.py
-----------------------------
Unified AdRelationshipClassifier — rule-based relationship classifier working with any
visual model (CLIP, ResNet) and text model (SentenceTransformer, Lexical Jaccard).
"""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from src.utils.signal_utils import compute_all_signals

if TYPE_CHECKING:
    from src.features.extractor import AdFeature

# ---------------------------------------------------------------------------
# Default threshold values
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: dict = {
    # Candidate selection
    "candidate_threshold": 0.65,

    # Identical
    "identical_phash_max": 2,
    "identical_text_min":  0.90,
    "identical_visual_min": 0.93,

    # Containment
    "containment_visual_min": 0.70,
    "containment_text_min":   0.80,

    # Color-variant
    "color_palette_max": 0.60,

    # Text-variant
    "text_variant_text_min": 0.30,
    "text_variant_text_max": 0.89,

    # Layout-variant
    "layout_variant_text_min": 0.40,
    "layout_variant_text_max": 0.89,
    "layout_variant_visual_min": 0.65,
    "layout_variant_visual_max": 0.90,
}

LABELS: list[str] = [
    "Identical",
    "Containment",
    "Color-variant",
    "Text-variant",
    "Layout-variant",
    "Unrelated",
]

Match = tuple[int, int, str, dict]


class AdRelationshipClassifier:
    """
    Unified Classifier executing decision-tree relationship taxonomy rules over
    computed visual, text, pHash, color, and dimension signals.

    Parameters
    ----------
    thresholds : dict | None
        Configuration thresholds from YAML file.
    text_model_type : str
        "sentence_transformer" or "jaccard".
    """

    def __init__(
        self,
        thresholds: dict | None = None,
        text_model_type: str = "sentence_transformer",
        visual_model_type: str = "clip",
    ) -> None:
        self.text_model_type = text_model_type
        self.visual_model_type = visual_model_type
        # Build thresholds dict with fallback mapping for clip/resnet keys
        t_raw = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

        # Normalize threshold key names
        vis_min = t_raw.get("dinov2_threshold") or t_raw.get("dino_threshold") or t_raw.get("clip_threshold") or t_raw.get("resnet_threshold") or t_raw.get("candidate_threshold", 0.65)
        ident_vis = t_raw.get("identical_dinov2_min") or t_raw.get("identical_dino_min") or t_raw.get("identical_clip_min") or t_raw.get("identical_resnet_min") or t_raw.get("identical_visual_min", 0.85)
        cont_vis  = t_raw.get("containment_dinov2_min") or t_raw.get("containment_dino_min") or t_raw.get("containment_clip_min") or t_raw.get("containment_resnet_min") or t_raw.get("containment_visual_min", 0.70)
        lay_vis_min = t_raw.get("layout_variant_dinov2_min") or t_raw.get("layout_variant_dino_min") or t_raw.get("layout_variant_clip_min") or t_raw.get("layout_variant_resnet_min") or t_raw.get("layout_variant_visual_min", 0.65)
        lay_vis_max = t_raw.get("layout_variant_dinov2_max") or t_raw.get("layout_variant_dino_max") or t_raw.get("layout_variant_clip_max") or t_raw.get("layout_variant_resnet_max") or t_raw.get("layout_variant_visual_max", 0.90)

        self.thresholds: dict = {
            **t_raw,
            "candidate_threshold":       vis_min,
            "identical_visual_min":      ident_vis,
            "containment_visual_min":    cont_vis,
            "layout_variant_visual_min": lay_vis_min,
            "layout_variant_visual_max": lay_vis_max,
        }

    def classify(self, signals: dict) -> str:
        """
        Apply rules to signal dictionary.
        """
        t = self.thresholds
        phash_dist = signals["phash_dist"]
        color_sim  = signals["color_sim"]
        text_sim   = signals["text_sim"]
        visual_sim = signals["visual_sim"]
        same_dims  = signals.get("dims_match", True)

        # Rule 1: IDENTICAL
        if (same_dims
                and phash_dist <= t["identical_phash_max"]
                and visual_sim >  t["identical_visual_min"]
                and text_sim   >  t["identical_text_min"]):
            return "Identical"

        # Rule 2: CONTAINMENT
        if (not same_dims
                and color_sim  >= t["color_palette_max"]
                and text_sim   >  t["containment_text_min"]
                and visual_sim >  t["containment_visual_min"]):
            return "Containment"

        # Rule 3: COLOR-VARIANT
        if (same_dims and color_sim < t["color_palette_max"]
                and text_sim   > t["identical_text_min"]
                and visual_sim > t["identical_visual_min"]):
            return "Color-variant"

        # Rule 4: TEXT-VARIANT
        if (same_dims
                and color_sim  >= t["color_palette_max"]
                and visual_sim >  t["identical_visual_min"]
                and t["text_variant_text_min"] < text_sim <= t["text_variant_text_max"]):
            return "Text-variant"

        # Rule 5: LAYOUT-VARIANT
        if (same_dims
                and color_sim  >= t["color_palette_max"]
                and t["layout_variant_text_min"] < text_sim <= t["layout_variant_text_max"]
                and t["layout_variant_visual_min"] < visual_sim < t["layout_variant_visual_max"]):
            return "Layout-variant"

        # Rule 6: UNRELATED
        return "Unrelated"

    def find_candidates(
        self,
        features: list[AdFeature],
        candidate_threshold: float | None = None,
        top_k: int | None = None,
        use_faiss: bool = True,
    ) -> list[tuple[int, int]]:
        """
        Retrieve candidate pairs using FAISS IndexFlatIP (or PyTorch fallback).
        Supports FAISS Range Search (thresh) and Top-K ANN search (top_k).
        """
        thresh = candidate_threshold or self.thresholds.get("candidate_threshold", 0.65)
        top_k_param = top_k or self.thresholds.get("top_k")
        n = len(features)

        raw = np.stack([f.visual_emb.flatten().astype(np.float32) for f in features])

        try:
            import faiss
            has_faiss = True
        except ImportError:
            has_faiss = False

        if use_faiss and has_faiss:
            dim = raw.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner Product == Cosine Sim for L2-normalized vectors
            index.add(raw)

            if top_k_param:
                print(f"Executing FAISS Top-{top_k_param} nearest neighbor search ({n} ads)...")
                distances, indices = index.search(raw, top_k_param + 1)
                pairs_set: set[tuple[int, int]] = set()
                for i in range(n):
                    for rank in range(1, indices.shape[1]):
                        j = int(indices[i, rank])
                        dist = float(distances[i, rank])
                        if j != -1 and j != i and dist >= thresh:
                            pairs_set.add((min(i, j), max(i, j)))
                pairs = sorted(list(pairs_set))
                print(f"  FAISS Top-{top_k_param} returned {len(pairs):,} unique candidate pairs (sim ≥ {thresh:.2f}).")
            else:
                print(f"Executing FAISS Range Search (sim ≥ {thresh:.2f}) over {n} ads...")
                lims, distances, indices = index.range_search(raw, thresh)
                pairs_set: set[tuple[int, int]] = set()
                for i in range(n):
                    start, end = lims[i], lims[i + 1]
                    for idx_pos in range(start, end):
                        j = int(indices[idx_pos])
                        if j > i:
                            pairs_set.add((i, j))
                pairs = sorted(list(pairs_set))
                print(f"  FAISS Range Search returned {len(pairs):,} candidate pairs.")
        else:
            print(f"Building PyTorch visual similarity matrix ({n}×{n})...")
            embs = torch.tensor(raw, dtype=torch.float32)
            sim  = torch.mm(embs, embs.t())
            rows, cols = torch.where(torch.triu(sim, diagonal=1) > thresh)
            pairs = [(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist())]
            print(f"  PyTorch matrix search returned {len(pairs):,} candidate pairs.")

        return pairs

    def classify_pairs(
        self,
        candidates: list[tuple[int, int]],
        features: list[AdFeature],
        cache_path: str | None = None,
    ) -> list[Match]:
        """
        Classify all candidate pairs, using cache if available.
        """
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached matches from {cache_path} ...")
            with open(cache_path, "rb") as fh:
                matches = pickle.load(fh)
            print(f"  Loaded {len(matches)} cached matches.")
            return matches

        print(f"Classifying {len(candidates)} candidate pairs...")
        matches: list[Match] = []

        for i, j in tqdm(candidates, desc="Classifying pairs", unit="pair"):
            f1, f2  = features[i], features[j]
            signals = compute_all_signals(f1, f2, text_model_type=self.text_model_type, visual_model_type=self.visual_model_type)
            label   = self.classify(signals)
            if label != "Unrelated":
                matches.append((f1.index, f2.index, label, signals))

        print(f"  Found {len(matches)} related pairs.")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(matches, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Matches cached → {cache_path}")

        return matches
