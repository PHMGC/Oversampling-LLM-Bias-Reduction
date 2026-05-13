"""Evaluate models sequentially across datasets for a given balancing strategy.

Usage:
    python scripts/eval.py --strategy baseline                        # evaluate all datasets
    python scripts/eval.py --strategy baseline --dataset mcauley/luxury_beauty
    python scripts/eval.py --strategy baseline --dataset mcauley/luxury_beauty mcauley/cds_reviews
    python scripts/eval.py --strategy baseline --gpu 0
    python scripts/eval.py --strategy baseline --dry-run              # dry-run
    python scripts/eval.py --strategy baseline --batch-size 128       # tuning
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import STRATEGIES, DATASET_IDS  # lightweight, safe to import early

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",   required=True, choices=STRATEGIES)
    p.add_argument("--dataset",    nargs='*', default=None, choices=DATASET_IDS,
                   help="Evaluate specific dataset(s). Can specify multiple: --dataset mcauley/luxury_beauty mcauley/cds_reviews. Default: all datasets.")
    p.add_argument("--model-name", default=None)
    p.add_argument("--gpu",        default=None, help="Override CUDA_VISIBLE_DEVICES, e.g. '0'")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Inference batch size (try 96/128/160 for tuning)")
    p.add_argument("--num-workers",  type=int, default=4,
                   help="DataLoader worker processes for data loading (default 4)")
    p.add_argument("--dry-run",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from transformers import AutoTokenizer
    from src.config import DATASETS, MODEL_NAME, SEED, TRAIN_RATIO
    from src.data_utils import get_tokenized_cache_path, get_tokenized_dataset
    from src.paths import MODELS_DIR
    from src.eval_utils import evaluate_one_job

    model_name = args.model_name or MODEL_NAME
    strategy   = args.strategy

    logging.info("Strategy : %s", strategy)
    logging.info("Model    : %s", model_name)

    # Handle dataset filtering: --dataset with no args means "all", --dataset X Y means "X and Y"
    target_datasets = set(args.dataset) if args.dataset is not None and len(args.dataset) > 0 else set()
    if target_datasets:
        logging.info("Datasets : %s", ", ".join(sorted(target_datasets)))
    else:
        logging.info("Datasets : all")

    # 1. Prepare test datasets
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    dataset_ids = []

    for author, names in DATASETS.items():
        for name in names:
            dataset_id = f"{author}/{name}"

            # Filter by --dataset if specified
            if target_datasets and dataset_id not in target_datasets:
                continue

            try:
                # We need the 'test' split.
                get_tokenized_dataset(author, name, tokenizer,
                                      split="test", strategy="baseline",
                                      train_ratio=TRAIN_RATIO, seed=SEED)
                dataset_ids.append(dataset_id)
                logging.info("  ready: %s", dataset_id)
            except Exception as exc:
                logging.error("  SKIP %s: %s", dataset_id, exc)

    if not dataset_ids:
        logging.error("No test datasets ready. Run 00_preprocessing.ipynb first.")
        sys.exit(1)

    # 2. Execute evaluations sequentially
    models_dir = MODELS_DIR / strategy
    job_results = []

    for dataset_id in dataset_ids:
        author, name = dataset_id.split("/", 1)
        # Note: Evaluation is always done on 'baseline' strategy tokenized dataset
        # (the real test set), not necessarily the oversampled one.
        eval_cache = get_tokenized_cache_path(author, name, "test", "baseline")
        model_dir = models_dir / dataset_id

        if args.dry_run:
            logging.info("  [dry-run] %s", dataset_id)
            continue

        # Execute evaluation for this dataset
        import time
        start = time.monotonic()
        try:
            metrics = evaluate_one_job(dataset_id, str(eval_cache), str(model_dir),
                                       strategy=strategy,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers)
            elapsed = time.monotonic() - start
            logging.info("  %-45s  %.0fs  ok", dataset_id, elapsed)
            job_results.append({
                "job_id": dataset_id,
                "success": True,
                "metrics": metrics,
                "elapsed_seconds": elapsed,
                "error": None,
            })
        except Exception as exc:
            elapsed = time.monotonic() - start
            logging.error("  %-45s  %.0fs  FAILED: %s", dataset_id, elapsed, exc)
            job_results.append({
                "job_id": dataset_id,
                "success": False,
                "metrics": None,
                "elapsed_seconds": elapsed,
                "error": str(exc),
            })

    if args.dry_run:
        return

    # 3. Save results JSON — preserve existing entries, update only newly evaluated datasets
    results_path = models_dir / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results
    existing = {}
    if results_path.exists():
        existing = json.loads(results_path.read_text())

    # Update with newly evaluated results
    summary = dict(existing)  # Start with all existing results
    for r in job_results:
        if r["success"]:
            # Extract metrics, excluding dataset_id
            metrics = r["metrics"]
            name_only = r["job_id"].split("/")[1]
            summary[name_only] = {k: v for k, v in metrics.items() if k != "dataset_id"}

    results_path.write_text(json.dumps(summary, indent=2))
    logging.info("Saved evaluation metrics to %s", results_path)

    if any(not r["success"] for r in job_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
