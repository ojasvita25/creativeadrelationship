"""
src/pipeline/pipeline.py
------------------------
Unified AdRelationshipPipeline — single reproducible pipeline orchestrator for
visual & text relationship discovery on ad creative datasets.
"""

from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from src.features.extractor import AdFeature, AdFeatureExtractor
from src.classifier.classifier import AdRelationshipClassifier, Match, LABELS
from src.reporter.generator import AdReportGenerator


@dataclass(frozen=True)
class PipelineResult:
    """Holds outputs of an AdRelationshipPipeline run."""
    features: list[AdFeature]
    matches:  list[Match]
    stats:    dict
    elapsed:  float


class AdRelationshipPipeline:
    """
    Unified pipeline orchestrator.

    Parameters
    ----------
    sample_size : int | None
    visual_model : str
    text_model : str
    candidate_threshold : float
    device : str
    cache_dir : str
    hf_token : str | None
    thresholds : dict | None
    max_pairs_per_type : int
    output_path : str
    pipeline_name : str | None
    """

    def __init__(
        self,
        dataset_name: str = "PeterBrendan/AdImageNet",
        dataset_split: str = "train",
        sample_size: int | None = 1000,
        visual_model: str = "openai/clip-vit-base-patch32",
        text_model: str = "all-MiniLM-L6-v2",
        candidate_threshold: float = 0.65,
        device: str = "auto",
        cache_dir: str = "cache",
        hf_token: str | None = None,
        thresholds: dict | None = None,
        max_pairs_per_type: int = 3,
        output_path: str = "reports/report.html",
        pipeline_name: str | None = None,
    ) -> None:
        self.dataset_name         = dataset_name
        self.dataset_split        = dataset_split
        self.sample_size          = sample_size
        self.visual_model         = visual_model
        self.text_model           = text_model
        self.candidate_threshold   = candidate_threshold
        self.cache_dir            = cache_dir
        self.hf_token             = hf_token or os.environ.get("HF_TOKEN")
        self.output_path          = output_path
        self.pipeline_name        = pipeline_name or "Ad Relationship Pipeline"

        # Initialize extractor, classifier, and reporter
        self.extractor  = AdFeatureExtractor(
            visual_model=visual_model,
            text_model=text_model,
            device=device,
            cache_dir=cache_dir,
        )
        self.classifier = AdRelationshipClassifier(
            thresholds=thresholds,
            text_model_type=text_model,
            visual_model_type=visual_model,
        )
        self.reporter   = AdReportGenerator(
            max_pairs_per_type=max_pairs_per_type,
            output_path=output_path,
        )

        self._result: PipelineResult | None = None

    def _load_dataset(self):
        from datasets import load_dataset
        n = self.sample_size
        print(f"Loading dataset '{self.dataset_name}' (split='{self.dataset_split}', n={n or 'all'})...")
        kwargs: dict = {"split": self.dataset_split}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        ds = load_dataset(self.dataset_name, **kwargs)
        print(f"  Dataset loaded: {len(ds)} total examples.")
        return ds

    @staticmethod
    def _compute_stats(matches: list[Match], total_n: int) -> dict:
        related_indices: set[int] = set()
        grouped: dict[str, list] = defaultdict(list)

        for idx_i, idx_j, label, signals in matches:
            if label != "Unrelated":
                related_indices.add(idx_i)
                related_indices.add(idx_j)
            grouped[label].append((idx_i, idx_j, label, signals))

        num_related    = len(related_indices)
        num_standalone = total_n - num_related
        per_type_pairs = Counter(label for _, _, label, _ in matches if label != "Unrelated")

        per_type_ads: dict[str, int] = {}
        for label, group in grouped.items():
            ads: set[int] = set()
            for idx_i, idx_j, _, _ in group:
                ads.add(idx_i)
                ads.add(idx_j)
            per_type_ads[label] = len(ads)

        return {
            "total_n":         total_n,
            "num_related":     num_related,
            "num_standalone":  num_standalone,
            "pct_related":     round(100 * num_related / total_n, 1) if total_n else 0,
            "per_type_pairs":  per_type_pairs,
            "per_type_ads":    per_type_ads,
            "grouped_matches": dict(grouped),
        }

    def _match_cache_path(self) -> str:
        n_str = str(self.sample_size) if self.sample_size else "all"
        vis_tag = "clip" if "clip" in self.visual_model.lower() else "resnet"
        txt_tag = "jaccard" if "jaccard" in self.text_model.lower() else "st"
        return os.path.join(
            self.cache_dir,
            f"matches_{vis_tag}_{txt_tag}_{n_str}_t{self.candidate_threshold}.pkl",
        )

    def run(self) -> PipelineResult:
        """
        Run the end-to-end relationship discovery pipeline.
        """
        t0 = time.time()

        dataset = self._load_dataset()
        features = self.extractor.extract(dataset, sample_size=self.sample_size)
        candidates = self.classifier.find_candidates(
            features, candidate_threshold=self.candidate_threshold
        )
        if getattr(self, "use_gbdt", False):
            from src.classifier.gbdt_classifier import GBDTAdRelationshipClassifier
            model_path = getattr(self, "gbdt_model_path", "models/gbdt_classifier.pkl")
            gbdt_clf = GBDTAdRelationshipClassifier(model_path=model_path)
            matches = gbdt_clf.classify_pairs(
                candidates, features, text_model_type=self.text_model, visual_model_type=self.visual_model
            )
        else:
            matches = self.classifier.classify_pairs(
                candidates, features, cache_path=self._match_cache_path()
            )
        stats = self._compute_stats(matches, total_n=len(features))
        elapsed = time.time() - t0

        print(f"\nPipeline ({self.pipeline_name}) complete in {elapsed:.1f}s.")
        print(f"  Related ads : {stats['num_related']} / {stats['total_n']} "
              f"({stats['pct_related']}%)")
        for label in LABELS[:-1]:
            count = stats["per_type_pairs"].get(label, 0)
            print(f"  {label:<18}: {count} pairs")

        self._result = PipelineResult(
            features=features,
            matches=matches,
            stats=stats,
            elapsed=elapsed,
        )
        return self._result

    def report(
        self,
        output_path: str | None = None,
        timings: dict | None = None,
    ) -> str:
        """
        Generate the HTML report for the completed run.
        """
        if self._result is None:
            raise RuntimeError("Call run() before report().")

        return self.reporter.generate(
            features=self._result.features,
            matches=self._result.matches,
            stats=self._result.stats,
            timings=timings,
            output_path=output_path,
        )

    @classmethod
    def from_config(cls, config: dict, hf_token: str | None = None) -> "AdRelationshipPipeline":
        """
        Instantiate pipeline directly from a parsed configuration dictionary.
        """
        p_cfg = config.get("pipeline", {})
        m_cfg = config.get("models", {})
        t_cfg = config.get("thresholds", {})

        pipe_type = p_cfg.get("type", "improved").lower()
        vis_model = m_cfg.get("visual_model", "openai/clip-vit-base-patch32" if pipe_type == "improved" else "resnet18")
        txt_model = m_cfg.get("text_model", "all-MiniLM-L6-v2" if pipe_type == "improved" else "jaccard")

        candidate_thresh = (
            t_cfg.get("clip_threshold")
            or t_cfg.get("resnet_threshold")
            or t_cfg.get("candidate_threshold", 0.65)
        )

        use_cache = p_cfg.get("use_cache", True)
        cache_dir = p_cfg.get("cache_dir", "cache") if use_cache else ""

        d_cfg = config.get("dataset", {})
        ds_name  = d_cfg.get("name", "PeterBrendan/AdImageNet")
        ds_split = d_cfg.get("split", "train")

        instance = cls(
            dataset_name=ds_name,
            dataset_split=ds_split,
            sample_size=p_cfg.get("sample_size", 1000),
            visual_model=vis_model,
            text_model=txt_model,
            candidate_threshold=candidate_thresh,
            device=m_cfg.get("device", "auto"),
            cache_dir=cache_dir,
            hf_token=hf_token,
            thresholds=t_cfg,
            max_pairs_per_type=p_cfg.get("max_pairs_per_type", 3),
            output_path=p_cfg.get("output_path", "reports/report.html"),
            pipeline_name=p_cfg.get("name"),
        )
        instance.use_gbdt = p_cfg.get("use_gbdt", False)
        instance.gbdt_model_path = m_cfg.get("gbdt_model_path", "models/gbdt_classifier.pkl")
        return instance


def create_pipeline(config: dict, hf_token: str | None = None) -> AdRelationshipPipeline:
    """Convenience alias for AdRelationshipPipeline.from_config."""
    return AdRelationshipPipeline.from_config(config, hf_token=hf_token)
