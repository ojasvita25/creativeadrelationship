# Take-Home Assignment: Ad Creative Similarity & Relationship Discovery

## Background

Our team works on understanding relationships between advertising creatives — identifying
when two assets are the same, near-duplicates, or variants of each other (recolored,
retexted, resized, restructured). This assignment is a scaled-down, self-contained version of
that problem. It's meant to test how you build and reason about a solution, not how much ML
theory you can recite.

## Objective

You're given a folder of real ad creative images. Build a system that determines which ads
are related to each other and how, then produce a report that answers:

1. **How many of the ads have at least one related counterpart in the set**, and how many
   appear to be unique/standalone?
2. **Of the related ones, what type of relationship do they share?** Classify each into one
   of the categories below.
3. **For a sample of pairs/clusters you find in each category, show your work**: the two
   images side by side, a plain-language explanation of why they're related, and the
   technical basis for the match — what your model or signal actually measured to flag them.

## Relationship Taxonomy

1. **Identical** — same creative (allowing for recompression/near-duplicate noise)
2. **Color-variant** — same layout and content, only the color/palette differs
3. **Text-variant** — same layout and visual content, only the overlaid text differs
4. **Layout-variant** — same template/structure, different photo or content
5. **Containment** — one image is a crop/resize of the other (e.g. same creative at a
   different ad size)
6. **Unrelated** — no meaningful relationship

## Dataset

A folder of real ad images — a subset of the open, MIT-licensed
[AdImageNet](https://huggingface.co/datasets/PeterBrendan/AdImageNet) dataset (real
programmatic display ads, with OCR-extracted ad text and IAB ad-size metadata).

## Required Approach

- This isn't a prompting exercise — solving it by one-shot prompting a foundation model (e.g.
  asking a vision-language model whether two images are related) doesn't meet the bar.
- Beyond that, the approach is your call — off-the-shelf models, something you train,
  classical ML, a combination of signals, whatever you think actually solves the problem
  well. Novelty can come from how you combine or apply things, not only from training
  something new.
- Include a simple baseline for comparison, and show where your approach improves on it.

### Non-goals (explicitly out of scope)

- No requirement to fine-tune a large pretrained model's full weights, if your approach uses
  one. Partial fine-tuning is a bonus stretch goal, not an expectation.
- No serving API, no UI, no demo required.
- No GPU is assumed — your approach should run in reasonable time on a laptop or free-tier
  Colab.

## Deliverables

1. **Code** (a git repo you can share with us)
2. **A report** containing:
   - Summary statistics: what fraction of the dataset has at least one related counterpart,
     broken down by relationship type
   - For each relationship type, 2-3 concrete example pairs/clusters — images shown side by
     side, a plain-language rationale, and the methodological basis for the match
   - Your baseline vs. your final approach, and how you validated that your approach
     actually works
3. **A short writeup**: your approach and why, and what you'd do next with more time

No fixed accuracy threshold to hit — we're interested in how far you push the solution and
how soundly you reason about it, not a pass/fail number.

## Timeline

About 1 week, elapsed (not full-time) — treat it like a real deadline around your own
schedule. If you have a working version earlier, feel free to send it in — if time permits,
we can share feedback so you can improve it before the final deadline.

## How this will be reviewed

We're not going to read through your code line by line — it's there to support an AI-assisted
review and, more importantly, a **follow-up conversation** where you walk us through your
approach and the choices you made. The report, writeup, and that conversation are the main
things we're evaluating, so don't over-invest in polish at the expense of having a clear
story to tell.

## Extensions (optional — attempt only if the core solution is solid)

- **Multi-modal fusion**: combine what you learn from the image with a separate signal
  derived from the ad's text, rather than relying on visual information alone.
- **Generalization check**: evaluate your approach against a small held-out set built from a
  *different* ad dataset (we can provide one) to see whether it generalizes or overfit.
