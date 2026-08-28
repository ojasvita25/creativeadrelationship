"""
train_gbdt.py
-------------
Training script for Gradient Boosted Decision Tree (GBDT) Ad Relationship Classifier.
Train on training ad split and validate on an independent holdout validation ad set.
"""

from __future__ import annotations

import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from imblearn.under_sampling import RandomUnderSampler

from src.config import load_config
from src.utils.signal_utils import compute_all_signals, dims_match
from src.classifier.classifier import AdRelationshipClassifier


LABEL_NAMES = [
    "Unrelated",
    "Identical",
    "Containment",
    "Color-variant",
    "Text-variant",
    "Layout-variant",
]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}


def extract_feature_vector(f1, f2, sig: dict) -> list[float]:
    """
    Extract a 4-dimensional pure multi-modal feature vector X for a candidate ad pair.
    """
    return [
        float(sig.get("visual_sim", 0.0)),
        float(sig.get("text_sim", 0.0)),
        float(sig.get("color_sim", 0.0)),
        float(sig.get("phash_dist", 64.0)),
    ]


def build_pair_dataset(candidates, features, rule_classifier, text_model_type, visual_model_type, desc="Building dataset"):
    X_list, y_list = [], []
    for i, j in tqdm(candidates, desc=desc):
        f1, f2 = features[i], features[j]
        sig = compute_all_signals(f1, f2, text_model_type=text_model_type, visual_model_type=visual_model_type)

        label = rule_classifier.classify(sig)
        feat_vec = extract_feature_vector(f1, f2, sig)
        label_id = LABEL_TO_ID[label]

        X_list.append(feat_vec)
        y_list.append(label_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train GBDT Classifier on 2,000 Ads and Validate on 1,000 Holdout Ads")
    parser.add_argument("--config", type=str, default="configs/clip_sentencetransformer_gbdt.yaml", help="Path to config YAML file")
    parser.add_argument("--output", type=str, default=None, help="Output path for trained model pkl")
    parser.add_argument("--train-size", type=int, default=2000, help="Number of ad samples for training (default: 2000)")
    parser.add_argument("--val-size", type=int, default=1000, help="Number of separate holdout ad samples for validation (default: 1000)")
    args = parser.parse_args()

    train_size = args.train_size
    val_size = args.val_size
    total_size = train_size + val_size

    print("==================================================")
    print(f"  TRAINING GBDT CLASSIFIER WITH HOLDOUT VALIDATION ({args.config})")
    print(f"  Train Set: {train_size:,} ads | Validation Holdout Set: {val_size:,} ads")
    print("==================================================")
    cfg = load_config(args.config)

    text_model_type = cfg.get("models", {}).get("text_model", "sentence_transformer")
    visual_model_type = cfg.get("models", {}).get("visual_model", "clip")

    if args.output:
        model_path = args.output
    else:
        model_path = cfg.get("models", {}).get("gbdt_model_path") or "models/gbdt_classifier.pkl"

    vis_tag = "dinov2" if "dinov2" in visual_model_type.lower() else ("clip" if "clip" in visual_model_type.lower() else "resnet")
    cache_path = f"cache/features_{vis_tag}_{total_size}.pkl"

    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        with open(cache_path, "rb") as f:
            all_features = pickle.load(f)
    else:
        print(f"Extracting {total_size:,} ad features using {visual_model_type} visual & {text_model_type} text models...")
        from src.pipeline.pipeline import AdRelationshipPipeline
        cfg_train = dict(cfg)
        cfg_train.setdefault("pipeline", {})["use_cache"] = False
        pipe = AdRelationshipPipeline.from_config(cfg_train)
        ds = pipe._load_dataset()
        all_features = pipe.extractor.extract(ds, sample_size=total_size)
        os.makedirs("cache", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(all_features, f)

    train_features = all_features[:train_size]
    val_features = all_features[train_size:total_size]

    # Re-index validation features from 0 to val_size - 1 for candidates matching
    for idx, f in enumerate(val_features):
        f.index = idx

    rule_classifier = AdRelationshipClassifier(
        thresholds=cfg.get("thresholds"),
        text_model_type=text_model_type,
        visual_model_type=visual_model_type,
    )

    t_cfg = cfg.get("thresholds", {})
    cand_thresh = t_cfg.get("dinov2_threshold") or t_cfg.get("dino_threshold") or t_cfg.get("clip_threshold") or t_cfg.get("resnet_threshold", 0.65)

    print(f"\n[1/3] Extracting Candidate Pairs for Training ({train_size} ads)...")
    train_candidates = rule_classifier.find_candidates(train_features, candidate_threshold=cand_thresh)
    print(f"  Retrieved {len(train_candidates):,} training candidate pairs.")

    print(f"\n[2/3] Extracting Candidate Pairs for Validation Holdout ({val_size} ads)...")
    val_candidates = rule_classifier.find_candidates(val_features, candidate_threshold=cand_thresh)
    print(f"  Retrieved {len(val_candidates):,} validation candidate pairs.")

    feature_names = ["visual_sim", "text_sim", "color_sim", "phash_dist"]

    X_train, y_train = build_pair_dataset(train_candidates, train_features, rule_classifier, text_model_type, visual_model_type, desc="Building Train Dataset")
    X_val, y_val = build_pair_dataset(val_candidates, val_features, rule_classifier, text_model_type, visual_model_type, desc="Building Validation Dataset")

    print(f"\n==================================================")
    print("  FEATURE MATRIX STATS")
    print("==================================================")
    print(f"Train Dataset: X shape {X_train.shape}, y shape {y_train.shape}")
    print(f"Validation Dataset: X shape {X_val.shape}, y shape {y_val.shape}")

    # Resampling Strategy: Downsample Unrelated (0) in training set
    rus = RandomUnderSampler(sampling_strategy={0: 5000}, random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )

    print(f"\n[3/3] Training GBDT Model on {len(y_train_res):,} Training Pairs...")
    clf.fit(X_train_res, y_train_res)

    print("\n--------------------------------------------------")
    print("  EVALUATING MODEL ON 1,000 HOLDOUT VALIDATION ADS")
    print("--------------------------------------------------")
    val_probs = clf.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    val_confidences = np.max(val_probs, axis=1)

    CONF_THRESHOLD = 0.70
    for idx in range(len(val_preds)):
        pred_id = val_preds[idx]
        
        # Enforce confidence threshold
        if pred_id > 0 and val_confidences[idx] < CONF_THRESHOLD:
            val_preds[idx] = 0

    present_classes = sorted(np.unique(np.concatenate([y_val, val_preds])))
    present_names = [LABEL_NAMES[i] for i in present_classes]

    print("\nHOLDOUT VALIDATION CLASSIFICATION REPORT (1,000 Independent Ads):")
    print(classification_report(y_val, val_preds, target_names=present_names, digits=4))

    print("\nHOLDOUT CONFUSION MATRIX:")
    cm = confusion_matrix(y_val, val_preds)
    header = " ".join([f"{name[:8]:>8}" for name in present_names])
    print(f"{'':10} {header}")
    for idx, row in enumerate(cm):
        row_str = " ".join([f"{val:8d}" for val in row])
        print(f"{present_names[idx]:<10} {row_str}")

    # Feature Importance Analysis (Evaluated on balanced resampled training pairs)
    print("\n--------------------------------------------------")
    print("  FEATURE IMPORTANCE ANALYSIS (PERMUTATION)")
    print("--------------------------------------------------")
    perm_imp = permutation_importance(clf, X_train_res, y_train_res, scoring='f1_macro', n_repeats=10, random_state=42)
    importances = perm_imp.importances_mean
    std = perm_imp.importances_std
    indices = np.argsort(importances)[::-1]

    for f in range(X_val.shape[1]):
        idx = indices[f]
        print(f"  {feature_names[idx]:<20}: {importances[idx]:.4f} ± {std[idx]:.4f}")

    # Save Model
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": clf,
                "feature_names": feature_names,
                "label_names": LABEL_NAMES,
            },
            f,
        )
    print(f"\n✅ Model successfully saved → {model_path}")


if __name__ == "__main__":
    main()