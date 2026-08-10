import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

LABEL_NAMES = [
    "Unrelated",
    "Identical",
    "Containment",
    "Color-variant",
    "Text-variant",
    "Layout-variant",
]

LABEL_COLORS = {
    "Unrelated": "#64748b",      # slate gray
    "Identical": "#8b5cf6",      # purple
    "Containment": "#10b981",    # green
    "Color-variant": "#f97316",  # orange
    "Text-variant": "#06b6d4",   # cyan
    "Layout-variant": "#ec4899", # pink
}


def load_model(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            return data["model"] if isinstance(data, dict) and "model" in data else data
    return None


def main():
    print("==================================================")
    print("  GENERATING 2D GBDT DECISION BOUNDARY MAPS (2 MODELS)")
    print("==================================================")

    model1_path = "models/gbdt_clip_sentencetransformer_classifier.pkl"
    if not os.path.exists(model1_path):
        model1_path = "models/gbdt_classifier.pkl"

    model2_path = "models/gbdt_clip_jaccard_classifier.pkl"
    if not os.path.exists(model2_path):
        model2_path = "models/gbdt_jaccard_classifier.pkl"

    model1 = load_model(model1_path)
    model2 = load_model(model2_path)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0f172a")

    v_grid = np.linspace(0.55, 1.0, 250)
    t_grid = np.linspace(0.0, 1.0, 250)
    V, T = np.meshgrid(v_grid, t_grid)

    cmap = ListedColormap([LABEL_COLORS[name] for name in LABEL_NAMES])

    # 4D feature grid: [visual_sim, text_sim, color_sim, phash_dist]
    # Slice A: Near-Duplicate Slice (phash_dist = 2, color_sim = 0.85)
    grid_X_near = np.column_stack([
        V.ravel(),
        T.ravel(),
        np.full(V.size, 0.85),
        np.full(V.size, 2.0),
    ])

    # Slice B: Variant / Non-Duplicate Slice (phash_dist = 12, color_sim = 0.85)
    grid_X_var = np.column_stack([
        V.ravel(),
        T.ravel(),
        np.full(V.size, 0.85),
        np.full(V.size, 12.0),
    ])

    # Row 0: SentenceTransformer Model
    if model1 is not None:
        # Near duplicate slice
        preds1_near = model1.predict(grid_X_near).reshape(V.shape)
        axes[0, 0].set_facecolor("#1e293b")
        axes[0, 0].contourf(V, T, preds1_near, levels=np.arange(7) - 0.5, cmap=cmap, alpha=0.7)
        axes[0, 0].set_title("SentenceTransformer GBDT — Near-Duplicate Slice (pHash ≤ 2)", fontsize=11, fontweight="bold", color="#f8fafc")
        axes[0, 0].set_xlabel("CLIP Visual Similarity", fontsize=10, color="#cbd5e1")
        axes[0, 0].set_ylabel("SentenceTransformer Text Sim", fontsize=10, color="#cbd5e1")
        axes[0, 0].grid(True, linestyle=":", alpha=0.3, color="#64748b")

        # Variant slice
        preds1_var = model1.predict(grid_X_var).reshape(V.shape)
        axes[0, 1].set_facecolor("#1e293b")
        axes[0, 1].contourf(V, T, preds1_var, levels=np.arange(7) - 0.5, cmap=cmap, alpha=0.7)
        axes[0, 1].set_title("SentenceTransformer GBDT — Variant Slice (pHash > 2)", fontsize=11, fontweight="bold", color="#f8fafc")
        axes[0, 1].set_xlabel("CLIP Visual Similarity", fontsize=10, color="#cbd5e1")
        axes[0, 1].set_ylabel("SentenceTransformer Text Sim", fontsize=10, color="#cbd5e1")
        axes[0, 1].grid(True, linestyle=":", alpha=0.3, color="#64748b")

    # Row 1: Lexical Jaccard Model
    if model2 is not None:
        # Near duplicate slice
        preds2_near = model2.predict(grid_X_near).reshape(V.shape)
        axes[1, 0].set_facecolor("#1e293b")
        axes[1, 0].contourf(V, T, preds2_near, levels=np.arange(7) - 0.5, cmap=cmap, alpha=0.7)
        axes[1, 0].set_title("Lexical Jaccard GBDT — Near-Duplicate Slice (pHash ≤ 2)", fontsize=11, fontweight="bold", color="#f8fafc")
        axes[1, 0].set_xlabel("CLIP Visual Similarity", fontsize=10, color="#cbd5e1")
        axes[1, 0].set_ylabel("Lexical Jaccard Text Sim", fontsize=10, color="#cbd5e1")
        axes[1, 0].grid(True, linestyle=":", alpha=0.3, color="#64748b")

        # Variant slice
        preds2_var = model2.predict(grid_X_var).reshape(V.shape)
        axes[1, 1].set_facecolor("#1e293b")
        axes[1, 1].contourf(V, T, preds2_var, levels=np.arange(7) - 0.5, cmap=cmap, alpha=0.7)
        axes[1, 1].set_title("Lexical Jaccard GBDT — Variant Slice (pHash > 2)", fontsize=11, fontweight="bold", color="#f8fafc")
        axes[1, 1].set_xlabel("CLIP Visual Similarity", fontsize=10, color="#cbd5e1")
        axes[1, 1].set_ylabel("Lexical Jaccard Text Sim", fontsize=10, color="#cbd5e1")
        axes[1, 1].grid(True, linestyle=":", alpha=0.3, color="#64748b")

    legend_elements = [Patch(facecolor=LABEL_COLORS[name], label=name) for name in LABEL_NAMES]
    axes[0, 1].legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    plt.suptitle("Multi-Slice 2D Decision Boundary Maps (SentenceTransformer vs Jaccard GBDT)", fontsize=14, fontweight="bold", color="#f8fafc", y=0.98)
    plt.tight_layout()

    out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gbdt_2d_decision_map.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved multi-slice 2D decision map → {out_path}")


if __name__ == "__main__":
    main()
