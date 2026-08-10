import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from src.utils.signal_utils import compute_all_signals
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


def main():
    parser = argparse.ArgumentParser(description="Train GBDT Classifier from Config and Pseudo-Labels")
    parser.add_argument("--config", type=str, default="configs/clip_rulebased.yaml", help="Path to config YAML file")
    parser.add_argument("--output", type=str, default=None, help="Output path for trained model pkl (default derived from config or models/gbdt_classifier.pkl)")
    args = parser.parse_args()

    print("==================================================")
    print(f"  TRAINING GBDT CLASSIFIER ({args.config})")
    print("==================================================")
    from src.config import load_config
    cfg = load_config(args.config)

    text_model_type = cfg.get("models", {}).get("text_model", "sentence_transformer")
    visual_model_type = cfg.get("models", {}).get("visual_model", "clip")

    if args.output:
        model_path = args.output
    elif text_model_type.lower() == "jaccard":
        model_path = "models/gbdt_jaccard_classifier.pkl"
    else:
        model_path = "models/gbdt_classifier.pkl"

    cache_path = "cache/features_clip_2000.pkl"
    if not os.path.exists(cache_path):
        print("Extracting 2,000 ad features using CLIP visual & text models...")
        from src.pipeline.pipeline import AdRelationshipPipeline
        pipe = AdRelationshipPipeline.from_config(cfg)
        ds = pipe._load_dataset()
        features = pipe.extractor.extract(ds, sample_size=2000)
        os.makedirs("cache", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(features, f)
    else:
        print(f"Loading 2,000 ad features from {cache_path} ...")
        with open(cache_path, "rb") as f:
            features = pickle.load(f)

    rule_classifier = AdRelationshipClassifier(
        thresholds=cfg.get("thresholds"),
        text_model_type=text_model_type,
        visual_model_type=visual_model_type,
    )

    candidates = rule_classifier.find_candidates(features, candidate_threshold=0.61)
    print(f"Retrieved {len(candidates):,} candidate pairs from FAISS.")

    # 1. Build Dataset (X, y)
    X_list = []
    y_list = []

    print(f"Extracting multi-modal signals (text_model={text_model_type}) for training dataset...")
    for i, j in tqdm(candidates, desc="Building dataset"):
        f1, f2 = features[i], features[j]
        sig = compute_all_signals(f1, f2, text_model_type=text_model_type, visual_model_type=visual_model_type)

        label = rule_classifier.classify(sig)
        feat_vec = extract_feature_vector(f1, f2, sig)
        label_id = LABEL_TO_ID[label]

        X_list.append(feat_vec)
        y_list.append(label_id)

    X_raw = np.array(X_list, dtype=np.float32)
    y_raw = np.array(y_list, dtype=np.int64)

    feature_names = [
        "visual_sim",
        "text_sim",
        "color_sim",
        "phash_dist",
    ]

    print(f"\n==================================================")
    print("  VERIFYING TRAINING DATASET FEATURE MATRIX")
    print("==================================================")
    print(f"X shape: {X_raw.shape}, y shape: {y_raw.shape}")
    print(f"Features: {feature_names}")

    counts = np.bincount(y_raw, minlength=len(LABEL_NAMES))
    print("\nOriginal Class Distribution:")
    for label_name, count in zip(LABEL_NAMES, counts):
        pct = 100 * count / len(y_raw) if len(y_raw) else 0
        print(f"  {label_name:<16}: {count:6d} pairs ({pct:5.2f}%)")

    # 2. Resampling Strategy
    rus = RandomUnderSampler(sampling_strategy={0: 1000}, random_state=42)

    # 3. Build Leakage-Free Pipeline
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        class_weight="balanced",
        random_state=42,
    )

    # The pipeline ensures resampling only applies to the training data inside each cross-validation fold
    pipeline = ImbPipeline([
        ("rus", rus),
        ("classifier", clf),
    ])

    # 4. 5-Fold Stratified Cross-Validation
    print("\n--------------------------------------------------")
    print("  EVALUATING 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("--------------------------------------------------")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Reverted to standard cross-validation predictions without thresholding
    y_pred_oof = cross_val_predict(pipeline, X_raw, y_raw, cv=skf)

    present_classes = sorted(np.unique(np.concatenate([y_raw, y_pred_oof])))
    present_names = [LABEL_NAMES[i] for i in present_classes]

    print("\nClassification Report (Out-of-Fold Predictions):")
    print(classification_report(y_raw, y_pred_oof, target_names=present_names, digits=4))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_raw, y_pred_oof)
    header = " ".join([f"{name[:8]:>8}" for name in present_names])
    print(f"{'':10} {header}")
    for idx, row in enumerate(cm):
        row_str = " ".join([f"{val:8d}" for val in row])
        print(f"{present_names[idx]:<10} {row_str}")

    # 5. Feature Importance Analysis
    print("\n--------------------------------------------------")
    print("  FEATURE IMPORTANCE ANALYSIS (PERMUTATION)")
    print("--------------------------------------------------")
    # Fit the pipeline on the full dataset for final feature importance and saving
    X_res, y_res = rus.fit_resample(X_raw, y_raw)
    clf.fit(X_res, y_res)

    perm_imp = permutation_importance(clf, X_res, y_res, n_repeats=10, random_state=42)
    importances = perm_imp.importances_mean
    std = perm_imp.importances_std
    indices = np.argsort(importances)[::-1]

    for f in range(X_res.shape[1]):
        idx = indices[f]
        print(f"  {feature_names[idx]:<20}: {importances[idx]:.4f} ± {std[idx]:.4f}")

    # 6. Save Model
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