"""
src/reporter/generator.py
--------------------------
AdReportGenerator — wraps the HTML report generation logic.

The heavy HTML rendering lives in improved/report_generator.py (unchanged).
This class provides the OOP interface, default configuration, and
output-path management.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.features.extractor import AdFeature
    from src.classifier.classifier import Match


class AdReportGenerator:
    """
    Generates an HTML report summarising the results of an ad similarity pipeline run.

    Parameters
    ----------
    max_pairs_per_type : int
        Maximum number of example image pairs shown per relationship category.
        Default: 3.
    output_path : str
        Default output path for the generated HTML file.
    """

    def __init__(
        self,
        max_pairs_per_type: int = 3,
        output_path: str = "reports/improved_report.html",
    ) -> None:
        self.max_pairs_per_type = max_pairs_per_type
        self.output_path = output_path

    def generate(
        self,
        features: list[AdFeature],
        matches: list[Match],
        stats: dict,
        timings: dict | None = None,
        output_path: str | None = None,
    ) -> str:
        """
        Render the HTML report and write it to disk.

        Parameters
        ----------
        features    : Full list of AdFeature objects (used for image rendering).
        matches     : Classified pairs from AdRelationshipClassifier.classify_pairs().
        stats       : Stats dict from AdRelationshipPipeline._compute_stats().
        timings     : Optional timing metadata dict for the report footer.
        output_path : Override the instance default output path.

        Returns
        -------
        str — absolute path to the written HTML file.
        """
        from src.reporter.report_generator import generate_report

        out = output_path or self.output_path
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

        generate_report(
            features=features,
            matches=matches,
            stats=stats,
            output_path=out,
            max_pairs_per_type=self.max_pairs_per_type,
            timings=timings,
        )
        return os.path.abspath(out)
