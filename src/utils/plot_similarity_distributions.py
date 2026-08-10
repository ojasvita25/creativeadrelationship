import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset
from src.features.extractor import AdFeatureExtractor

def main():
    print("==================================================")
    print("  COMPUTING EMPIRICAL COSINE SIMILARITY DISTRIBUTIONS")
    print("  (1,000 Real Images from PeterBrendan/AdImageNet)")
    print("==================================================")

    np.random.seed(42)
    torch.manual_seed(42)

    sample_n = 1000
    print(f"Loading {sample_n} images from 'PeterBrendan/AdImageNet'...")
    dataset = load_dataset("PeterBrendan/AdImageNet", split="train")

    # 1. Extract Real ResNet-18 Features
    print("\n[1/2] Extracting real ResNet-18 visual embeddings...")
    extractor_resnet = AdFeatureExtractor(visual_model="resnet18", text_model="jaccard")
    features_resnet = extractor_resnet.extract(dataset, sample_size=sample_n)
    
    resnet_embs = np.array([f.visual_emb for f in features_resnet if f.visual_emb is not None])
    print(f"  Extracted {len(resnet_embs)} ResNet-18 vectors of shape {resnet_embs.shape[1]}")

    triu_idx = np.triu_indices(len(resnet_embs), k=1)
    resnet_sim_matrix = np.dot(resnet_embs, resnet_embs.T)
    resnet_sims = resnet_sim_matrix[triu_idx]
    print(f"  Calculated {len(resnet_sims):,} real pairwise ResNet-18 similarities.")

    # 2. Extract Real CLIP ViT-B/32 Features
    print("\n[2/2] Extracting real CLIP ViT-B/32 visual embeddings...")
    extractor_clip = AdFeatureExtractor(visual_model="openai/clip-vit-base-patch32", text_model="jaccard")
    features_clip = extractor_clip.extract(dataset, sample_size=sample_n)
    
    clip_embs = np.array([f.visual_emb for f in features_clip if f.visual_emb is not None])
    print(f"  Extracted {len(clip_embs)} CLIP vectors of shape {clip_embs.shape[1]}")

    clip_sim_matrix = np.dot(clip_embs, clip_embs.T)
    clip_sims = clip_sim_matrix[triu_idx]
    print(f"  Calculated {len(clip_sims):,} real pairwise CLIP similarities.")

    # 3. Plot Both Empirical Distributions Overlaid on the SAME Figure
    print("\nRendering Combined Empirical Distributions Plot (Overlaid on Single Figure)...")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")

    # ResNet-18 Histogram
    ax.hist(resnet_sims, bins=70, density=True, color="#f97316", alpha=0.45, edgecolor="#c2410c", label="Empirical ResNet-18 (499,500 Pairs)")
    ax.axvline(x=0.73, color="#ea580c", linestyle="--", linewidth=2.5, label="resnet_threshold: 0.73")

    # CLIP ViT-B/32 Histogram
    ax.hist(clip_sims, bins=70, density=True, color="#38bdf8", alpha=0.45, edgecolor="#0284c7", label="Empirical CLIP ViT-B/32 (499,500 Pairs)")
    ax.axvline(x=0.61, color="#8b5cf6", linestyle="--", linewidth=2.5, label="clip_threshold: 0.61")

    # Labels and Formatting
    ax.set_title("ResNet-18 vs. CLIP ViT-B/32 Empirical Cosine Similarity (1,000 AdImageNet Ads)", fontsize=13, fontweight="bold", color="#f8fafc", pad=14)
    ax.set_xlabel("Pairwise Visual Cosine Similarity", fontsize=11, color="#cbd5e1")
    ax.set_ylabel("Density", fontsize=11, color="#cbd5e1")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax.legend(loc="upper right", frameon=True, facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=10)

    plt.tight_layout()

    out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cosine_similarity_distributions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved combined empirical distributions plot → {out_path}")

if __name__ == "__main__":
    main()
