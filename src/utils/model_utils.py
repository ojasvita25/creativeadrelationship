"""
src/utils/model_utils.py
-------------------------
Utility functions for visual and text model loading, inference, and feature extraction.
Supports CLIP, ResNet-18, SentenceTransformer, and Lexical Tokenization.
"""

from __future__ import annotations

import re
from typing import Any, Tuple

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sentence_transformers import SentenceTransformer


def load_visual_model(
    model_name: str,
    device: str,
) -> Tuple[Any, Any]:
    """
    Load visual model and processor/transform.

    Parameters
    ----------
    model_name : str
        Model identifier: e.g. "facebook/dinov2-base", "openai/clip-vit-base-patch32", or "resnet18".
    device : str
        Compute device ("mps", "cuda", "cpu").

    Returns
    -------
    Tuple[model, processor_or_transform]
    """
    if "clip" in model_name.lower():
        from transformers import CLIPModel, CLIPProcessor
        print(f"Loading CLIP ({model_name}) on device: {device}...")
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        return model, processor

    elif "resnet" in model_name.lower():
        print(f"Loading ResNet-18 baseline on device: {device}...")
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        # Remove classification head to get pooled features (512-d)
        modules = list(resnet.children())[:-1]
        model = torch.nn.Sequential(*modules).to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return model, transform

    elif "dinov2" in model_name.lower():
        from transformers import AutoImageProcessor, AutoModel
        hf_name = "facebook/dinov2-base" if ("vitb" in model_name.lower() or "torch" in model_name.lower() or "base" in model_name.lower()) else model_name
        print(f"Loading DINOv2 ({hf_name}) on device: {device}...")
        processor = AutoImageProcessor.from_pretrained(hf_name)
        model = AutoModel.from_pretrained(hf_name).to(device)
        model.eval()
        return model, processor
    else:
        raise ValueError(f"Unsupported visual model name: {model_name!r}")


def get_visual_embedding(
    model_name: str,
    model: Any,
    processor_or_transform: Any,
    image: Image.Image,
    device: str,
) -> np.ndarray:
    """
    Extract L2-normalized visual embedding for an image.
    """
    if "clip" in model_name.lower():
        inputs = processor_or_transform(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            if hasattr(model, "get_base_model"):
                base = model.get_base_model()
                vision_out = base.vision_model(pixel_values=inputs["pixel_values"])
                emb = base.visual_projection(vision_out.pooler_output)
            else:
                vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
                emb = model.visual_projection(vision_out.pooler_output)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.detach().squeeze().cpu().numpy().flatten()

    elif "resnet" in model_name.lower():
        img_tensor = processor_or_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).squeeze()
            norm = feat.norm(p=2, dim=-1, keepdim=True)
            if norm > 0:
                feat = feat / norm
        return feat.cpu().numpy().flatten()

    elif "dinov2" in model_name.lower():
        inputs = processor_or_transform(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0]
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.detach().squeeze().cpu().numpy().flatten()
    else:
        raise ValueError(f"Unsupported visual model name: {model_name!r}")


def load_text_model(model_name: str, device: str) -> SentenceTransformer | None:
    """
    Load text embedding model or return None for lexical Jaccard.
    """
    if model_name.lower() == "jaccard":
        return None
    print(f"Loading SentenceTransformer ({model_name}) on device: {device}...")
    return SentenceTransformer(model_name, device=device)


def extract_text_representation(
    model_name: str,
    text_model: SentenceTransformer | None,
    text: str,
) -> Tuple[np.ndarray | None, set[str]]:
    """
    Extract text embedding vector and/or lexical token set.

    Returns
    -------
    Tuple[text_emb, tokens_set]
    """
    clean_text = (text or "").strip().lower()
    words = set(re.findall(r"\b\w+\b", clean_text)) if clean_text else set()

    if model_name.lower() == "jaccard" or text_model is None:
        return None, words

    if not clean_text:
        return np.zeros(384, dtype=np.float32), words

    emb = text_model.encode(clean_text, convert_to_numpy=True, show_progress_bar=False)
    norm = np.linalg.norm(emb)
    text_emb = (emb / norm).flatten() if norm > 0 else emb.flatten()
    return text_emb, words
