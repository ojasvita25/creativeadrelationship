#!/usr/bin/env python3
"""
main.py
-------
Single unified CLI entry point for Ad Creative Relationship Pipelines.

Usage
-----
    # Run Improved Pipeline using default YAML config
    python main.py --config configs/improved.yaml

    # Run Baseline Pipeline using YAML config
    python main.py --config configs/baseline.yaml

    # Override sample size and output path
    python main.py --config configs/improved.yaml --sample 500 --output reports/my_report.html

    # Re-generate report from cache only
    python main.py --config configs/improved.yaml --report-only
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))

from src.config import load_config
from src.pipeline.pipeline import AdRelationshipPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ad Creative Similarity & Relationship Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", "-c",
        default="configs/improved.yaml",
        help="Path to YAML configuration file. Default: configs/improved.yaml",
    )
    p.add_argument(
        "--sample", "-n",
        default=None,
        help="Override sample size (integer or 'all').",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Override HTML report output path.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default=None,
        help="Override compute device (auto, cpu, mps, cuda).",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Override cache directory.",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Override max example pairs per relationship category.",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Re-generate report from cached features and matches only.",
    )
    p.add_argument(
        "--dataset-name",
        default=None,
        help="Override HuggingFace dataset repository name (e.g. PeterBrendan/AdImageNet).",
    )
    p.add_argument(
        "--dataset-split",
        default=None,
        help="Override HuggingFace dataset split (e.g. train).",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace access token (or set HF_TOKEN env var).",
    )
    return p.parse_args()


def main() -> None:
    # Set global random seeds
    import random
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    args = parse_args()

    t_wall_start = time.perf_counter()
    started_at   = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Load configuration
    cli_overrides = {
        "dataset_name":  args.dataset_name,
        "dataset_split": args.dataset_split,
        "sample":        args.sample,
        "output":        args.output,
        "device":        args.device,
        "cache_dir":     args.cache_dir,
        "max_pairs":     args.max_pairs,
    }
    cfg = load_config(args.config, cli_overrides)
    p_cfg = cfg.get("pipeline", {})
    t_cfg = cfg.get("thresholds", {})

    pipe_name   = p_cfg.get("name", "Ad Relationship Pipeline")
    output_path = p_cfg.get("output_path", "reports/report.html")
    sample_size = p_cfg.get("sample_size")

    # HuggingFace token resolution
    hf_token: str | None = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        print("[auth] HuggingFace token found — using it for dataset access.")

    if args.report_only:
        print("[report-only mode] Loading cached results...")

    # Build and run pipeline
    pipeline = AdRelationshipPipeline.from_config(cfg, hf_token=hf_token)

    t0 = time.perf_counter()
    result = pipeline.run()
    pipeline_secs = round(time.perf_counter() - t0, 2)

    def _fmt(secs: float) -> str:
        m, s = divmod(int(secs), 60)
        return f"{m}m {s:02d}s" if m else f"{secs:.1f}s"

    total_secs = round(time.perf_counter() - t_wall_start, 2)

    timing_record = {
        "started_at":       started_at,
        "sample_size":      sample_size,
        "pipeline_name":    pipe_name,
        "timings_seconds": {
            "pipeline_total":   pipeline_secs,
            "total_wall_clock": total_secs,
        },
        "timings_formatted": {
            "pipeline_total":   _fmt(pipeline_secs),
            "total_wall_clock": _fmt(total_secs),
        },
    }
    # Pass threshold values for HTML timing card display
    if "resnet_threshold" in t_cfg:
        timing_record["resnet_threshold"] = t_cfg["resnet_threshold"]
    if "clip_threshold" in t_cfg:
        timing_record["clip_threshold"] = t_cfg["clip_threshold"]

    t_rep = time.perf_counter()
    out_path = pipeline.report(timings=timing_record)
    rep_secs = round(time.perf_counter() - t_rep, 2)
    timing_record["timings_seconds"]["report_generation"]   = rep_secs
    timing_record["timings_formatted"]["report_generation"] = _fmt(rep_secs)

    print("\n" + "─" * 50)
    print(f"  ⏱  {pipe_name.upper()} TIMING SUMMARY")
    print("─" * 50)
    print(f"  Started at       : {started_at}")
    print(f"  Pipeline         : {_fmt(pipeline_secs)}")
    print(f"  Report generation: {_fmt(rep_secs)}")
    print(f"  Total wall-clock : {_fmt(timing_record['timings_seconds']['total_wall_clock'])}")
    print("─" * 50)

    timing_path = output_path.replace(".html", "_timing.json")
    with open(timing_path, "w") as fh:
        json.dump(timing_record, fh, indent=2)
    print(f"  Timing log saved → {timing_path}")
    print(f"\nDone! Report generated at:\n  open {output_path}")


if __name__ == "__main__":
    main()
