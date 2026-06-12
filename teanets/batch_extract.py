#!/usr/bin/env python3
"""
Batch SVO extraction module for the teanets library.

Processes texts in batches with batch coreference resolution (one model call
per batch instead of per text) to avoid memory saturation and maximise GPU
throughput.

Usage (standalone CLI):
    # From the project root:
    python -m teanets.batch_extract --help

    # CPU (default):
    python -m teanets.batch_extract --input data/sexualassault.csv

    # GPU for fastcoref (torch CUDA):
    python -m teanets.batch_extract --input data/sexualassault.csv --gpu

    # Custom batch size and sample:
    python -m teanets.batch_extract --gpu --batch-size 50 --sample-size 500 --seed 42

    # Resume from checkpoint:
    python -m teanets.batch_extract --gpu --resume
"""

import gc
import argparse
import time
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import warnings

from .textloader import clean_text, load_fastcoref_model, resolve_coref_prediction
from .nlp_utils import spacynlp
from .svo_extraction import extract_svos
import teanets.analytics as analytics

warnings.filterwarnings("ignore")


def _load_coref_model(use_gpu):
    """Load the fastcoref model once (per device) and return it."""
    device = "cuda" if use_gpu else "cpu"
    print(f"  Loading fastcoref model on {device}...")
    model = load_fastcoref_model(device=device)
    print(f"  fastcoref model loaded on {device}")
    return model


def batch_coref_resolve(texts, use_gpu):
    """Resolve coreferences for a list of texts in ONE model call.

    This is more efficient than calling solve_coreferences() per text
    (as the legacy extract_svos_from_text does) because the fastcoref model
    is loaded once and processes all texts in a single forward pass.
    """
    model = _load_coref_model(use_gpu)

    # Clean texts first (same as text_preparation does)
    cleaned = [clean_text(t) if isinstance(t, str) and t.strip() else "" for t in texts]

    # Filter out empty texts — fastcoref doesn't handle them
    valid_indices = [i for i, t in enumerate(cleaned) if len(t.strip()) > 0]
    valid_texts = [cleaned[i] for i in valid_indices]

    if not valid_texts:
        return cleaned

    # Single batch prediction for all texts
    preds = model.predict(texts=valid_texts)
    if not isinstance(preds, list):
        preds = [preds]

    # Apply resolutions
    resolved = list(cleaned)  # copy
    for idx, pred in zip(valid_indices, preds):
        try:
            resolved[idx] = resolve_coref_prediction(pred, cleaned[idx])
        except Exception:
            resolved[idx] = cleaned[idx]  # fallback to cleaned text

    return resolved


def setup_gpu_spacy():
    """Try to enable GPU for spaCy (requires cupy)."""
    try:
        import spacy
        if spacy.prefer_gpu():
            print("[GPU] spaCy using GPU (cupy found)")
            return True
        else:
            print("[GPU] spaCy: cupy not available, using CPU")
            print("      To enable: uv pip install cupy-cuda12x")
            return False
    except Exception:
        return False


def process_batch(texts, batch_idx, output_dir, group_name, use_coref, use_gpu):
    """Process a batch: batch-coref then SVO extraction per text."""
    # Step 1: batch coreference resolution (1 model call for all texts)
    if use_coref:
        resolved_texts = batch_coref_resolve(texts, use_gpu)
    else:
        resolved_texts = [clean_text(t) if isinstance(t, str) and t.strip() else "" for t in texts]

    # Step 2: SVO extraction per text (spaCy, no heavy model reload)
    results = []
    for text in resolved_texts:
        if not text or len(text.strip()) == 0:
            continue
        try:
            doc = spacynlp(text)
            svo_df = extract_svos(doc)
            if svo_df is not None and len(svo_df) > 0:
                results.append(svo_df)
        except Exception:
            continue

    if results:
        batch_df = pd.concat(results, ignore_index=True)
        # Use proper dtypes instead of stringifying everything: numeric
        # columns stay numeric (svo_id is nullable because semantic rows
        # carry "N/A"), text columns become strings.
        batch_df["svo_id"] = pd.to_numeric(
            batch_df["svo_id"], errors="coerce"
        ).astype("Int64")
        for col in ("Semantic-Syntactic", "passive_approx", "is_passive"):
            if col in batch_df.columns:
                batch_df[col] = (
                    pd.to_numeric(batch_df[col], errors="coerce").fillna(0).astype(int)
                )
        for col in ("Node 1", "TEA", "Node 2", "TEA2", "Hypergraph"):
            if col in batch_df.columns:
                batch_df[col] = batch_df[col].astype(str)
        outpath = output_dir / f"{group_name}_batch_{batch_idx:04d}.parquet"
        batch_df.to_parquet(outpath, index=False)
        return len(results), len(batch_df)
    return 0, 0


def merge_batches(output_dir, group_name):
    """Merge all batch parquet files into a single CSV."""
    pattern = f"{group_name}_batch_*.parquet"
    files = sorted(output_dir.glob(pattern))
    if not files:
        print(f"  No batch files found for {group_name}")
        return None

    all_dfs = [pd.read_parquet(f) for f in files]

    merged = analytics.merge_svo_dataframes(all_dfs)
    outpath = output_dir / f"{group_name}_svo.csv"
    merged.to_csv(outpath, index=False)
    print(f"  {group_name}: merged {len(files)} batches -> {len(merged)} SVO rows -> {outpath}")
    return merged


def main(args=None):
    """CLI entry point for batch SVO extraction."""
    parser = argparse.ArgumentParser(description="Extract SVOs from a CSV dataset (teanets batch extractor)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for fastcoref (torch CUDA)")
    parser.add_argument("--batch-size", type=int, default=50, help="Texts per batch (default: 50)")
    parser.add_argument("--sample-size", type=int, default=500, help="Texts per group (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoints")
    parser.add_argument("--no-coref", action="store_true", help="Skip coreference resolution (faster)")
    parser.add_argument("--output-dir", type=str, default="data/svo_output", help="Output directory")
    parser.add_argument("--input", type=str, default="data/sexualassault.csv", help="Input CSV file")
    parser.add_argument("--text-col", type=str, default="text", help="Column name for text (default: text)")
    parser.add_argument("--group-col", type=str, default="comment", help="Column name for group split (default: comment)")
    args = parser.parse_args(args)

    use_gpu = args.gpu
    use_coref = not args.no_coref

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Load data ----
    print(f"Loading {args.input}...")
    df_raw = pd.read_csv(args.input)
    print(f"  Total rows: {len(df_raw)}")

    # Split into groups using --group-col when available; otherwise process
    # the whole file as a single group.
    if args.group_col in df_raw.columns:
        group_series = df_raw[args.group_col]
        if group_series.dtype != bool:
            # CSV round-trips often turn booleans into strings: try to map
            # them back instead of silently producing two empty groups.
            mapped = (
                group_series.astype(str).str.strip().str.lower().map(
                    {"true": True, "false": False, "1": True, "0": False}
                )
            )
            if mapped.isna().any():
                bad = group_series[mapped.isna()].unique()[:5]
                raise ValueError(
                    f"Column '{args.group_col}' is not boolean and contains "
                    f"unmappable values (e.g. {bad!r}). Expected True/False "
                    "(or 1/0) markers for the posts/comments split."
                )
            group_series = mapped
        posts = df_raw[~group_series]
        comments = df_raw[group_series]
        print(f"  Posts: {len(posts)}, Comments: {len(comments)}")
        raw_groups = [("posts", posts), ("comments", comments)]
    else:
        print(
            f"  Column '{args.group_col}' not found: processing all rows as a "
            "single group 'texts'."
        )
        raw_groups = [("texts", df_raw)]

    groups = []
    for group_name, group_df in raw_groups:
        sample = group_df.sample(
            n=min(args.sample_size, len(group_df)), random_state=args.seed
        )
        print(f"  Sampled {group_name}: {len(sample)}")
        groups.append((group_name, sample))

    # ---- Setup models ----
    print("\nLoading NLP models...")

    if use_gpu:
        setup_gpu_spacy()

    # Load coref model once upfront
    if use_coref:
        _load_coref_model(use_gpu)

    # Warmup spaCy
    print("Warming up spaCy...")
    doc = spacynlp("The cat sat on the mat.")
    _ = extract_svos(doc)
    print("  Warmup OK")

    # ---- Process groups ----
    for group_name, sample_df in groups:
        texts = sample_df[args.text_col].tolist()
        n_batches = (len(texts) + args.batch_size - 1) // args.batch_size

        if args.resume:
            existing = sorted(output_dir.glob(f"{group_name}_batch_*.parquet"))
            if existing:
                last_batch = int(existing[-1].stem.split("_")[-1])
                start_batch = last_batch + 1
                print(f"\n[RESUME] {group_name}: found {len(existing)} existing batches, starting from batch {start_batch}")
            else:
                start_batch = 0
        else:
            start_batch = 0

        print(f"\n{'='*60}")
        print(f"Processing {group_name.upper()}: {len(texts)} texts in {n_batches} batches")
        print(f"{'='*60}")

        total_svos = 0
        total_rows = 0
        t_start = time.time()

        for batch_idx in tqdm(range(start_batch, n_batches), desc=group_name):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            n_svos, n_rows = process_batch(
                batch_texts, batch_idx, output_dir, group_name, use_coref, use_gpu
            )
            total_svos += n_svos
            total_rows += n_rows

            gc.collect()

        elapsed = time.time() - t_start
        print(f"  Done: {total_svos} texts -> {total_rows} SVO rows in {elapsed:.1f}s")

    # ---- Merge ----
    print(f"\n{'='*60}")
    print("Merging batches...")
    print(f"{'='*60}")

    for group_name, _ in groups:
        merge_batches(output_dir, group_name)

    print("\nAll done! Load results in notebook with:")
    for group_name, _ in groups:
        print(f"  {group_name}_svo = pd.read_csv('{output_dir}/{group_name}_svo.csv')")


if __name__ == "__main__":
    main()
