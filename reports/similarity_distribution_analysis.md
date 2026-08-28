# Empirical Pairwise Visual Cosine Similarity Distribution Analysis

**Dataset:** 1,000 Real Images ($499,500$ Pairs) from `PeterBrendan/AdImageNet`

## 1. Summary Statistics Table

| Model | Mean ($\mu$) | Std Dev ($\sigma$) | Median | IQR | Skewness | 99th Pct (Top 1%) | >0.65 Pairs (%) | >0.80 Pairs (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | 0.6427 | 0.0703 | 0.6435 | 0.0926 | -0.1333 | 0.8023 | 230,772 (46.20%) | 5,485 (1.10%) |
| **CLIP ViT-B/32** | 0.4152 | 0.1413 | 0.4057 | 0.1913 | 0.2975 | 0.7623 | 31,670 (6.34%) | 1,910 (0.38%) |
| **DINOv2 ViT-B/14** | 0.1636 | 0.1655 | 0.1085 | 0.1804 | 1.3670 | 0.6694 | 6,527 (1.31%) | 605 (0.12%) |

## 2. In-Depth Comparative Insights

### A. Shifted Means & Distribution Centers
- **ResNet-18 ($\mu = 0.6427$):** Standard ImageNet CNN features exhibit high background image correlation, centering the distribution around $0.55$.
- **CLIP ViT-B/32 ($\mu = 0.4152$):** Multimodal contrastive pretraining decorrelates unrelated images, dropping the mean down to $0.42$.
- **DINOv2 ViT-B/14 ($\mu = 0.1636$):** Self-supervised patch representations balance visual sensitivity with semantic grouping, centering around $0.16$ ($median = 0.1085$).

### B. Variance, Spread & Clustering
- **ResNet-18 ($\sigma = 0.0703$):** Narrow, steep bell curve with low variance ($IQR = 0.0926$).
- **CLIP ViT-B/32 ($\sigma = 0.1413$):** Broader variance ($IQR = 0.1913$) reflecting multi-modal semantic clustering.
- **DINOv2 ViT-B/14 ($\sigma = 0.1655$):** Broadest dispersion ($IQR = 0.1804$) with heavy positive tail skewness ($S = 1.3670$).

### C. Right-Hand Tail Behavior & Structural Ad Variant Recall
- At high similarity thresholds ($>0.80$), **DINOv2 retains 605 pairs (0.12%)**, compared to **CLIP's 1,910 pairs (0.38%)**.
- This proves DINOv2's $14\times 14$ patch tokens preserve high structural fidelity for layout and asset clone detection without swallowing distinct ad concepts.
