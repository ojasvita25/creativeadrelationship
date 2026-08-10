import os
import pickle
import numpy as np

from src.utils.signal_utils import color_similarity, dims_match, text_similarity

LABEL_NAMES = [
    "Unrelated",
    "Identical",
    "Containment",
    "Color-variant",
    "Text-variant",
    "Layout-variant",
]


class GBDTAdRelationshipClassifier:
    """
    Lightweight GBDT Classifier for Ad Creative Relationships.
    Uses a pre-trained HistGradientBoostingClassifier model trained on 4 feature signals.
    """

    def __init__(self, model_path: str = "models/gbdt_classifier.pkl") -> None:
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.label_names = LABEL_NAMES

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"] if isinstance(data, dict) and "model" in data else data
                if isinstance(data, dict):
                    self.feature_names = data.get("feature_names")
                    self.label_names = data.get("label_names", LABEL_NAMES)
        else:
            raise FileNotFoundError(f"GBDT model not found at {model_path}. Run train_gbdt.py first!")

    def classify_signals(self, f1, f2, sig: dict) -> str:
        """
        Classify a single pair given their features and signals dict using GBDT.
        """
        X_vec = np.array(
            [[
                float(sig.get("visual_sim", 0.0)),
                float(sig.get("text_sim", 0.0)),
                float(sig.get("color_sim", 0.0)),
                float(sig.get("phash_dist", 64.0)),
            ]],
            dtype=np.float32,
        )
        pred_id = int(self.model.predict(X_vec)[0])
        return self.label_names[pred_id]

    def classify_pairs(
        self,
        candidates: list[tuple[int, int]],
        features: list,
        text_model_type: str = "sentence_transformer",
        visual_model_type: str = "clip",
        verbose: bool = False,
    ) -> list[tuple[int, int, str, dict]]:
        """
        Classify all candidate pairs using vectorized batch GBDT prediction.
        """
        from tqdm import tqdm

        print(f"Extracting signals for {len(candidates):,} candidate pairs for GBDT...")
        X_list = []
        sig_list = []
        pairs_list = []

        print(f"Vectorizing signals for {len(candidates):,} candidate pairs...")
        for i, j in tqdm(candidates, desc="Vectorizing signals", unit="pair"):
            f1, f2 = features[i], features[j]
            v_sim = float(np.dot(f1.visual_emb, f2.visual_emb))
            t_sim = text_similarity(f1, f2, text_model_type=text_model_type)
            c_sim = color_similarity(f1, f2)
            phash_d = float(f1.phash - f2.phash)

            X_list.append([v_sim, t_sim, c_sim, phash_d])
            sig_list.append({
                "visual_sim": v_sim,
                "clip_sim": v_sim,
                "text_sim": t_sim,
                "color_sim": c_sim,
                "phash_dist": phash_d,
                "dims_match": dims_match(f1, f2),
            })
            pairs_list.append((i, j))

        X = np.array(X_list, dtype=np.float32)
        print("  Running GBDT batch prediction...")
        
        probs = self.model.predict_proba(X)
        GLOBAL_THRESHOLD = 0.90
        matches = []
        
        debug_counts = {name: 0 for name in self.label_names}
        DEBUG_LIMIT = 5

        for idx, row_probs in enumerate(probs):
            pred_id = np.argmax(row_probs)
            confidence = row_probs[pred_id]
            label = self.label_names[pred_id]
            
            if verbose and debug_counts[label] < DEBUG_LIMIT:
                v_sim = sig_list[idx]["visual_sim"]
                t_sim = sig_list[idx]["text_sim"]
                c_sim = sig_list[idx]["color_sim"]
                phash = sig_list[idx]["phash_dist"]
                print(f"  [{label[:9]:<9}] Conf: {confidence:.3f} | Vis: {v_sim:.3f} | Txt: {t_sim:.3f} | Col: {c_sim:.3f} | pHash: {phash:<4.1f}")
                debug_counts[label] += 1

            if label != "Unrelated" and confidence < GLOBAL_THRESHOLD:
                label = "Unrelated"

            if label != "Unrelated":
                i, j = pairs_list[idx]
                f1, f2 = features[i], features[j]
                is_same = dims_match(f1, f2)

                # Strict Domain Rules:
                # Identical, Color-variant, Text-variant, Layout-variant MUST have matching dimensions.
                # Containment MUST have different dimensions.
                if label in ["Identical", "Color-variant", "Text-variant", "Layout-variant"] and not is_same:
                    continue
                if label == "Containment" and is_same:
                    continue
                
                sig_list[idx]["confidence"] = float(confidence)
                matches.append((features[i].index, features[j].index, label, sig_list[idx]))

        print(f"  GBDT Classifier found {len(matches):,} related pairs.")
        return matches