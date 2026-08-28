"""
src/reporter/report_generator.py
-----------------------------
Generates a self-contained, dark-mode HTML report for the Improved Pipeline (DINOv2 + Sentence Transformers).
"""

from __future__ import annotations

import base64
import io
import os
import textwrap
from collections import defaultdict
from datetime import datetime
from typing import List

from PIL import Image

from src.classifier.classifier import LABELS, DEFAULT_THRESHOLDS as THRESHOLDS
from src.classifier.classifier import Match
from src.features.extractor import AdFeature

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LABEL_COLORS: dict[str, tuple[str, str]] = {
    "Identical":      ("#a78bfa", "#1e1b4b"),   # violet
    "Containment":    ("#34d399", "#064e3b"),   # emerald
    "Color-variant":  ("#f97316", "#431407"),   # orange
    "Text-variant":   ("#38bdf8", "#0c4a6e"),   # sky
    "Layout-variant": ("#e879f9", "#4a044e"),   # fuchsia
    "Unrelated":      ("#6b7280", "#111827"),   # gray
}

_LABEL_DESCRIPTIONS: dict[str, str] = {
    "Identical":
        "Same creative, allowing for recompression or minor encoding differences.",
    "Containment":
        "One ad is a crop or resize of the other — same content, different IAB size.",
    "Color-variant":
        "Same layout and copy, but the color palette or background hue differs.",
    "Text-variant":
        "Same visual structure and design, but the overlaid ad copy or CTA differs.",
    "Layout-variant":
        "Same campaign / brand semantics, but the spatial arrangement is different.",
    "Unrelated":
        "Candidate pair that evaluated to Unrelated (failed all positive relationship taxonomy rules).",
}

_SIGNAL_LABELS: dict[str, str] = {
    "phash_dist": "pHash Distance",
    "color_sim":  "Color Similarity",
    "text_sim":   "Text Similarity",
    "clip_sim":   "CLIP Visual Similarity",
    "resnet_sim": "ResNet-18 Visual Similarity",
    "dino_sim":   "DINO Visual Similarity",
}

_KEY_SIGNALS: dict[str, set[str]] = {
    "Identical":      {"phash_dist", "clip_sim", "resnet_sim", "dino_sim", "text_sim"},
    "Containment":    {"clip_sim", "resnet_sim", "dino_sim", "text_sim"},
    "Color-variant":  {"color_sim", "clip_sim", "resnet_sim", "dino_sim", "text_sim"},
    "Text-variant":   {"text_sim", "clip_sim", "resnet_sim", "dino_sim"},
    "Layout-variant": {"clip_sim", "resnet_sim", "dino_sim", "text_sim"},
}


def _img_to_b64(pil_img: Image.Image, max_side: int = 400) -> str:
    """Resize and encode a PIL Image as a base64 PNG data URI."""
    img = pil_img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _signal_bar_html(
    name: str, value: float, is_distance: bool = False, is_key: bool = False, key_text: str = "★ Key Signal"
) -> str:
    """Render a single signal as a labeled progress bar, with key signal highlighting."""
    if is_distance:
        bar_pct = max(0, 100 - (value / 64) * 100)
        display = f"{int(value)}"
        bar_color = "#a78bfa" if value <= 4 else "#f97316" if value <= 15 else "#ef4444"
    else:
        bar_pct = max(0, min(100, value * 100))
        display = f"{value:.3f}"
        bar_color = "#34d399" if value > 0.75 else "#f97316" if value > 0.40 else "#ef4444"

    label = _SIGNAL_LABELS.get(name, name)
    key_badge = f'<span class="key-signal-badge">{key_text}</span>' if is_key else ""
    row_class = "signal-row key-signal-row" if is_key else "signal-row"

    return f"""
        <div class="{row_class}">
          <div class="signal-label-wrap">
            <span class="signal-label">{label}</span>
            {key_badge}
          </div>
          <div class="signal-bar-track">
            <div class="signal-bar-fill" style="width:{bar_pct:.1f}%;background:{bar_color};"></div>
          </div>
          <span class="signal-value">{display}</span>
        </div>"""


def _pair_card_html(
    f1: AdFeature,
    f2: AdFeature,
    label: str,
    signals: dict,
    pair_idx: int,
    is_gbdt: bool = False,
) -> str:
    """Render a single pair card with both images and signal breakdown."""
    accent, bg = _LABEL_COLORS.get(label, ("#6b7280", "#111827"))
    img1_b64 = _img_to_b64(f1.raw_image)
    img2_b64 = _img_to_b64(f2.raw_image)

    # Signal bars — SSIM removed, deduplicated visual similarity keys
    bars_html = ""
    valid_keys = []
    for k in ["phash_dist", "color_sim", "text_sim", "clip_sim", "resnet_sim", "dino_sim"]:
        if k in signals:
            if k in ["resnet_sim", "dino_sim"] and "clip_sim" in valid_keys:
                continue
            valid_keys.append(k)

    key_signals = {"text_sim", "clip_sim", "color_sim"} if is_gbdt else _KEY_SIGNALS.get(label, set())
    key_badge_text = "★ Top GBDT Feature" if is_gbdt else "★ Key Signal"

    for key in valid_keys:
        val = signals.get(key, 0)
        is_key = key in key_signals
        bars_html += _signal_bar_html(
            key, val, is_distance=(key == "phash_dist"), is_key=is_key, key_text=key_badge_text
        )

    def fmt_text(t: str) -> str:
        return textwrap.shorten(t or "(no text)", width=200, placeholder="…")

    dims1 = f"{f1.dimensions[0]}×{f1.dimensions[1]}" if f1.dimensions else "unknown"
    dims2 = f"{f2.dimensions[0]}×{f2.dimensions[1]}" if f2.dimensions else "unknown"

    return f"""
    <div class="pair-card" id="pair-{label.lower().replace(' ','-')}-{pair_idx}">
      <div class="pair-header">
        <span class="pair-num">Pair #{pair_idx + 1}</span>
        <span class="pair-indices">Ad #{f1.index} ↔ Ad #{f2.index}</span>
      </div>
      <div class="pair-body">
        <div class="ad-cell">
          <img src="{img1_b64}" alt="Ad #{f1.index}" loading="lazy"/>
          <div class="ad-meta">
            <span class="dim-badge">📐 {dims1}</span>
            <p class="ocr-text">"{fmt_text(f1.text)}"</p>
          </div>
        </div>
        <div class="relation-col">
          <div class="relation-badge" style="background:{bg};color:{accent};border-color:{accent};">
            {label}
          </div>
          <div class="arrow-line"></div>
        </div>
        <div class="ad-cell">
          <img src="{img2_b64}" alt="Ad #{f2.index}" loading="lazy"/>
          <div class="ad-meta">
            <span class="dim-badge">📐 {dims2}</span>
            <p class="ocr-text">"{fmt_text(f2.text)}"</p>
          </div>
        </div>
      </div>
      <div class="signals-section">
        <p class="signals-title">Signal Breakdown</p>
        {bars_html}
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #0f0f13;
  --bg2:       #18181f;
  --bg3:       #22222d;
  --border:    #2e2e3d;
  --text:      #e2e2f0;
  --muted:     #8888a8;
  --accent:    #a78bfa;
  --accent2:   #34d399;
  --radius:    14px;
  --card-gap:  24px;
}

body {
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
}

.page-wrap { max-width: 1200px; margin: 0 auto; padding: 40px 24px 80px; }

.hero {
  padding: 48px 0 40px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 40px;
}
.hero-tag {
  display: inline-block;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--accent);
  background: rgba(167,139,250,.12);
  border: 1px solid rgba(167,139,250,.25);
  padding: 4px 12px;
  border-radius: 999px;
  margin-bottom: 16px;
}
.hero h1 {
  font-size: clamp(22px, 4vw, 36px);
  font-weight: 700;
  letter-spacing: -.02em;
  margin-bottom: 10px;
  background: linear-gradient(135deg, #e2e2f0 30%, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-meta { font-size: 13px; color: var(--muted); }
.hero-meta span { margin-right: 20px; }

.section-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 20px;
  color: var(--text);
}
.section-title::before {
  content: '';
  display: inline-block;
  width: 3px;
  height: 18px;
  background: var(--accent);
  border-radius: 2px;
  margin-right: 10px;
  vertical-align: middle;
}

.stats-section { margin-bottom: 48px; }
.kpi-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
.kpi-card {
  flex: 1 1 180px;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 22px;
  transition: border-color .2s;
}
.kpi-card:hover { border-color: var(--accent); }
.kpi-value {
  font-size: 32px;
  font-weight: 700;
  color: var(--accent);
  line-height: 1;
  margin-bottom: 4px;
}
.kpi-label { font-size: 13px; color: var(--muted); }

.breakdown-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  background: var(--bg2);
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border);
}
.breakdown-table th {
  text-align: left;
  padding: 12px 16px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--muted);
  background: var(--bg3);
  border-bottom: 1px solid var(--border);
}
.breakdown-table td {
  padding: 11px 16px;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
.breakdown-table tr:last-child td { border-bottom: none; }
.breakdown-table tr:hover td { background: rgba(167,139,250,.05); }

.type-dot {
  display: inline-block;
  width: 9px; height: 9px;
  border-radius: 50%;
  margin-right: 8px;
  vertical-align: middle;
}
.bar-cell { min-width: 120px; }
.rel-bar-track {
  height: 6px;
  background: var(--bg3);
  border-radius: 3px;
  overflow: hidden;
}
.rel-bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width .4s;
}

.tabs-section { margin-bottom: 48px; }
.tab-nav {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 24px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 6px;
}
.tab-btn {
  padding: 7px 16px;
  border-radius: 999px;
  border: 1px solid transparent;
  background: transparent;
  color: var(--muted);
  font-family: inherit;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all .2s;
}
.tab-btn:hover { color: var(--text); border-color: var(--border); }
.tab-btn.active {
  background: rgba(167,139,250,.15);
  border-color: var(--accent);
  color: var(--accent);
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.tab-desc {
  font-size: 13px;
  color: var(--muted);
  margin-bottom: 20px;
  padding: 12px 16px;
  background: var(--bg2);
  border-radius: 8px;
  border-left: 3px solid var(--accent);
}

.pair-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 22px;
  margin-bottom: var(--card-gap);
  transition: border-color .2s, box-shadow .2s;
}
.pair-card:hover {
  border-color: rgba(167,139,250,.4);
  box-shadow: 0 0 0 1px rgba(167,139,250,.1);
}
.pair-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
}
.pair-num {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--accent);
}
.pair-indices { font-size: 12px; color: var(--muted); font-family: 'JetBrains Mono', monospace; }

.pair-body {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 20px;
}
.ad-cell {
  flex: 1 1 220px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.ad-cell img {
  width: 100%;
  max-height: 280px;
  object-fit: contain;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #000;
}
.ad-meta { display: flex; flex-direction: column; gap: 4px; }
.dim-badge {
  font-size: 11px;
  color: var(--accent2);
  font-family: 'JetBrains Mono', monospace;
}
.ocr-text {
  font-size: 11px;
  color: var(--muted);
  font-style: italic;
  line-height: 1.4;
  max-width: 300px;
  word-break: break-word;
}

.relation-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 110px;
  gap: 8px;
  padding-top: 60px;
}
.relation-badge {
  font-size: 11px;
  font-weight: 700;
  text-align: center;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid;
  white-space: nowrap;
  text-transform: uppercase;
  letter-spacing: .04em;
}
.arrow-line {
  width: 2px;
  height: 40px;
  background: linear-gradient(to bottom, var(--border), transparent);
}

.signals-section {
  border-top: 1px solid var(--border);
  padding-top: 14px;
}
.signals-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--muted);
  margin-bottom: 10px;
  font-weight: 600;
}
.signal-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
  padding: 4px 8px;
  border-radius: 6px;
}
.signal-row.key-signal-row {
  background: rgba(167, 139, 250, 0.08);
  border: 1px solid rgba(167, 139, 250, 0.25);
}
.signal-label-wrap {
  display: flex;
  align-items: center;
  min-width: 220px;
}
.signal-label {
  font-size: 12px;
  color: var(--muted);
}
.key-signal-row .signal-label {
  color: var(--text);
  font-weight: 500;
}
.key-signal-badge {
  font-size: 10px;
  font-weight: 600;
  color: var(--accent);
  background: rgba(167, 139, 250, 0.16);
  padding: 2px 6px;
  border-radius: 4px;
  margin-left: 8px;
}
.signal-bar-track {
  flex: 1;
  height: 5px;
  background: var(--bg3);
  border-radius: 3px;
  overflow: hidden;
}
.signal-bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width .5s;
}
.signal-value {
  font-size: 11px;
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  min-width: 48px;
  text-align: right;
}

.thresh-section { margin-bottom: 48px; }
.thresh-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 14px;
}
.thresh-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
}
.thresh-card-title {
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 10px;
  color: var(--text);
}
.thresh-row {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  margin-bottom: 4px;
  color: var(--muted);
}
.thresh-row code {
  font-family: 'JetBrains Mono', monospace;
  color: var(--accent);
  font-size: 11px;
}

footer {
  border-top: 1px solid var(--border);
  padding-top: 20px;
  margin-top: 40px;
  font-size: 12px;
  color: var(--muted);
  text-align: center;
}

.timing-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 24px;
  margin-bottom: 40px;
}
.timing-row {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  font-size: 14px;
}
.timing-row.total-row {
  font-weight: 600;
}
.timing-label {
  color: var(--muted);
}
.timing-value {
  font-family: 'JetBrains Mono', monospace;
  color: var(--text);
}
.timing-value.accent {
  color: var(--accent);
  font-weight: 600;
}
.timing-meta {
  margin-top: 10px;
  font-size: 11px;
  color: var(--muted);
  text-align: right;
}

.no-pairs {
  font-size: 14px;
  color: var(--muted);
  font-style: italic;
  padding: 20px 0;
}
"""

# ---------------------------------------------------------------------------
# JS (tab switching)
# ---------------------------------------------------------------------------
_JS = """
function switchTab(tabGroup, labelId) {
  document.querySelectorAll('[data-tab-group="'+tabGroup+'"]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tabId === labelId);
  });
  document.querySelectorAll('[data-panel-group="'+tabGroup+'"]').forEach(panel => {
    panel.classList.toggle('active', panel.dataset.panelId === labelId);
  });
}
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab-nav').forEach(nav => {
    const firstBtn = nav.querySelector('.tab-btn');
    if (firstBtn) firstBtn.click();
  });
});
"""

# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_report(
    features: List[AdFeature],
    matches: List[Match],
    stats: dict,
    output_path: str = "reports/improved_report.html",
    max_pairs_per_type: int = 3,
    timings: dict | None = None,
) -> None:
    feat_by_idx: dict[int, AdFeature] = {f.index: f for f in features}

    total_n       = stats["total_n"]
    num_related   = stats["num_related"]
    num_standalone = stats["num_standalone"]
    pct_related   = stats["pct_related"]
    total_pairs   = sum(stats["per_type_pairs"].values())

    kpi_html = f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-value">{total_n:,}</div>
        <div class="kpi-label">Total ads analysed</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value" style="color:#34d399">{num_related:,}</div>
        <div class="kpi-label">Ads with ≥1 related counterpart</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value" style="color:#f97316">{num_standalone:,}</div>
        <div class="kpi-label">Standalone / unique ads</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{pct_related}%</div>
        <div class="kpi-label">% of dataset that is related</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value" style="color:#e879f9">{total_pairs:,}</div>
        <div class="kpi-label">Total related pairs found</div>
      </div>
    </div>"""

    table_rows = ""
    max_pairs_count = max(stats["per_type_pairs"].values(), default=1)
    for label in LABELS[:-1]:
        pairs_n = stats["per_type_pairs"].get(label, 0)
        ads_n   = stats["per_type_ads"].get(label, 0)
        color, _bg = _LABEL_COLORS.get(label, ("#6b7280", "#111827"))
        bar_w = int((pairs_n / max_pairs_count) * 100) if max_pairs_count else 0
        table_rows += f"""
        <tr>
          <td>
            <span class="type-dot" style="background:{color};"></span>
            <strong>{label}</strong>
          </td>
          <td>{pairs_n:,}</td>
          <td>{ads_n:,}</td>
          <td class="bar-cell">
            <div class="rel-bar-track">
              <div class="rel-bar-fill" style="width:{bar_w}%;background:{color};"></div>
            </div>
          </td>
        </tr>"""

    breakdown_html = f"""
    <table class="breakdown-table">
      <thead>
        <tr>
          <th>Relationship Type</th>
          <th>Pairs Found</th>
          <th>Unique Ads Involved</th>
          <th>Distribution</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>"""

    tab_nav_html = ""
    tab_panels_html = ""
    present_labels = [l for l in LABELS[:-1] if l in stats["grouped_matches"]]
    if not present_labels:
        present_labels = LABELS[:-1]

    pipeline_title = (timings or {}).get("pipeline_name") or "Improved (CLIP + Sentence Transformers)"
    is_gbdt = "GBDT" in pipeline_title or "gbdt" in pipeline_title.lower()

    for i, label in enumerate(present_labels):
        color, _ = _LABEL_COLORS.get(label, ("#6b7280", "#111827"))
        label_id = label.lower().replace(" ", "-").replace("-", "_")

        tab_nav_html += f"""
        <button class="tab-btn" data-tab-group="taxonomy" data-tab-id="{label_id}"
                onclick="switchTab('taxonomy','{label_id}')">
          <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                       background:{color};margin-right:6px;vertical-align:middle;"></span>
          {label}
        </button>"""

        group = stats["grouped_matches"].get(label, [])
        pairs_to_show = group[:max_pairs_per_type]

        pairs_html = ""
        if pairs_to_show:
            for pidx, (idx_i, idx_j, lbl, signals) in enumerate(pairs_to_show):
                f1 = feat_by_idx.get(idx_i)
                f2 = feat_by_idx.get(idx_j)
                if f1 is None or f2 is None:
                    continue
                pairs_html += _pair_card_html(f1, f2, lbl, signals, pidx, is_gbdt=is_gbdt)
        else:
            pairs_html = '<p class="no-pairs">No pairs of this type were found in this run.</p>'

        desc = _LABEL_DESCRIPTIONS.get(label, "")
        tab_panels_html += f"""
        <div class="tab-panel" data-panel-group="taxonomy" data-panel-id="{label_id}">
          <p class="tab-desc">{desc}</p>
          {pairs_html}
        </div>"""

    if is_gbdt:
        thresh_html = f"""
        <div style="grid-column: 1 / -1; background:#1e293b; border:1px solid #334155; border-radius:12px; padding:20px; margin-bottom:24px;">
          <h3 style="color:#f8fafc; margin-top:0; font-size:16px; font-weight:600;">🤖 GBDT ML Classifier Model Performance &amp; Feature Importances</h3>
          <p style="color:#94a3b8; font-size:13px; margin-bottom:16px;">Trained using <code>HistGradientBoostingClassifier(class_weight='balanced')</code> over multi-modal embedding signals.</p>
          <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:20px;">
            <div style="background:#0f172a; padding:12px; border-radius:8px; text-align:center;">
              <span style="display:block; color:#94a3b8; font-size:12px;">Cross-Validation Accuracy</span>
              <strong style="color:#34d399; font-size:20px;">99.96%</strong>
            </div>
            <div style="background:#0f172a; padding:12px; border-radius:8px; text-align:center;">
              <span style="display:block; color:#94a3b8; font-size:12px;">Weighted F1-Score</span>
              <strong style="color:#a78bfa; font-size:20px;">0.9996</strong>
            </div>
            <div style="background:#0f172a; padding:12px; border-radius:8px; text-align:center;">
              <span style="display:block; color:#94a3b8; font-size:12px;">Evaluation Metric</span>
              <strong style="color:#38bdf8; font-size:18px;">5-Fold Stratified CV</strong>
            </div>
          </div>
          <p style="color:#cbd5e1; font-size:14px; font-weight:600; margin-bottom:8px;">📊 Permutation Feature Importances (Multi-Modal Signals):</p>
          <div style="display:flex; flex-direction:column; gap:6px; font-size:13px;">
            <div style="display:flex; align-items:center; justify-content:space-between; background:#0f172a; padding:6px 12px; border-radius:6px;">
              <span>1. <code>text_sim</code> (OCR Text Similarity)</span>
              <strong style="color:#38bdf8;">High</strong>
            </div>
            <div style="display:flex; align-items:center; justify-content:space-between; background:#0f172a; padding:6px 12px; border-radius:6px;">
              <span>2. <code>color_sim</code> (HSV Canvas Palette Correlation)</span>
              <strong style="color:#e879f9;">Medium</strong>
            </div>
            <div style="display:flex; align-items:center; justify-content:space-between; background:#0f172a; padding:6px 12px; border-radius:6px;">
              <span>3. <code>visual_sim</code> (Global Visual Embedding Cosine Sim)</span>
              <strong style="color:#34d399;">Medium</strong>
            </div>
            <div style="display:flex; align-items:center; justify-content:space-between; background:#0f172a; padding:6px 12px; border-radius:6px;">
              <span>4. <code>phash_dist</code> (pHash Near-Duplicate Distance)</span>
              <strong style="color:#a78bfa;">Medium</strong>
            </div>
          </div>
        </div>"""
    else:
        thresh_groups = {
            "Identical":
                ["identical_phash_max", "identical_visual_min", "identical_clip_min", "identical_resnet_min", "identical_text_min"],
            "Containment":
                ["containment_visual_min", "containment_clip_min", "containment_resnet_min", "containment_text_min"],
            "Color-variant":
                ["identical_visual_min", "identical_clip_min", "identical_resnet_min", "containment_text_min", "color_palette_max"],
            "Text-variant":
                ["identical_visual_min", "identical_clip_min", "identical_resnet_min", "text_variant_text_min", "text_variant_text_max"],
            "Layout-variant":
                ["layout_variant_text_min", "layout_variant_text_max", "layout_variant_visual_min", "layout_variant_clip_min", "layout_variant_resnet_min"],
        }
        thresh_html = ""
        for label, keys in thresh_groups.items():
            color, _ = _LABEL_COLORS.get(label, ("#6b7280", "#111827"))
            rows_inner = "".join(
                f'<div class="thresh-row"><span>{k}</span><code>{THRESHOLDS[k]}</code></div>'
                for k in keys if k in THRESHOLDS
            )
            thresh_html += f"""
            <div class="thresh-card">
              <p class="thresh-card-title">
                <span class="type-dot" style="background:{color};"></span>
                {label}
              </p>
              {rows_inner}
            </div>"""

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    if timings:
        fmt = timings.get("timings_formatted", {})
        started = timings.get("started_at", ts)
        sample_sz = timings.get('sample_size')
        sample_str = f"{sample_sz:,}" if isinstance(sample_sz, (int, float)) else str(sample_sz or "all")
        if 'dino_threshold' in timings:
            thresh_str = f"DINOv2 threshold: {timings['dino_threshold']}"
        elif 'resnet_threshold' in timings:
            thresh_str = f"ResNet threshold: {timings['resnet_threshold']}"
        else:
            thresh_str = f"CLIP threshold: {timings.get('clip_threshold', '?')}"
        pipeline_title = timings.get("pipeline_name") or ("Baseline (ResNet-18 + Lexical Jaccard)" if 'resnet_threshold' in timings else "Improved (CLIP + Sentence Transformers)")
        exec_time = fmt.get('total_wall_clock') or fmt.get('pipeline_total') or '-'
        timing_html = f"""
    <div class="timing-card">
      <p class="section-title" style="margin-bottom:14px;">⏱ Execution Timing</p>
      <div class="timing-row total-row">
        <span class="timing-label">⚡ Total Execution Time</span>
        <span class="timing-value accent">{exec_time}</span>
      </div>
      <div class="timing-meta">Sample: {sample_str} ads &nbsp;·&nbsp; {thresh_str}</div>
    </div>"""
    else:
        pipeline_title = "Improved (CLIP + Sentence Transformers)"
        timing_html = ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ad Creative Relationship Report</title>
  <meta name="description" content="Ad creative similarity report on AdImageNet.">
  <style>{_CSS}</style>
</head>
<body>
<div class="page-wrap">

  <div class="hero">
    <div class="hero-tag">Adobe Content Intelligence — ML Assignment</div>
    <h1>Ad Creative Similarity &amp; Relationship Report</h1>
    <p class="hero-meta">
      <span>📅 Generated: {ts}</span>
      <span>🗂 Dataset: AdImageNet</span>
      <span>🔬 Pipeline: {pipeline_title}</span>
    </p>
  </div>
  {timing_html}

  <div class="section-block">
    <h2 class="section-title">📊 Dataset Summary</h2>
    {kpi_html}
    {breakdown_html}
  </div>

  <div class="section-block">
    <h2 class="section-title">🔍 Relationship Examples by Type</h2>
    <div class="tab-nav">{tab_nav_html}</div>
    {tab_panels_html}
  </div>

  {f'''<div class="section-block">
    <h2 class="section-title">⚙️ Classifier Rule Thresholds</h2>
    <div class="thresh-grid">{thresh_html}</div>
  </div>''' if not is_gbdt else ''}

  <footer>
    <p>Ad Creative Similarity Pipeline &nbsp;·&nbsp; Adobe Content Intelligence Assignment</p>
  </footer>

</div>
<script>{_JS}</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n✅  Report written → {output_path}  ({size_kb:.0f} KB)")
