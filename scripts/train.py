"""Train models sequentially across datasets for a given balancing strategy.

Usage:
    python scripts/train.py --strategy baseline                    # train all datasets
    python scripts/train.py --strategy baseline --dataset mcauley/luxury_beauty
    python scripts/train.py --strategy baseline --dataset mcauley/luxury_beauty mcauley/cds_reviews
    python scripts/train.py --strategy baseline --epochs 5 --gpu 0
    python scripts/train.py --strategy baseline --dry-run           # dry-run
    python scripts/train.py --strategy baseline --force             # re-train existing models
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

from src.config import STRATEGIES, DATASET_IDS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",   required=True, choices=STRATEGIES)
    p.add_argument("--dataset",    nargs='*', default=None, choices=DATASET_IDS,
                   help="Train specific dataset(s). Can specify multiple: --dataset mcauley/luxury_beauty mcauley/cds_reviews. Default: all datasets.")
    p.add_argument("--model-name", default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--patience",   type=int, default=None)
    p.add_argument("--gpu",   default=None, help="Override CUDA_VISIBLE_DEVICES, e.g. '0'")
    p.add_argument("--force", action="store_true",
                   help="Force re-training even if model already exists locally")
    p.add_argument("--dry-run",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from transformers import AutoTokenizer
    from src.config import DATASETS, MODEL_NAME, N_EPOCHS, PATIENCE, SEED, TRAIN_RATIO
    from src.data_utils import get_tokenized_cache_path, get_tokenized_dataset
    from src.paths import MODELS_DIR
    from src.train_utils import train_one_job

    model_name = args.model_name or MODEL_NAME
    epochs     = args.epochs or N_EPOCHS
    patience   = args.patience or PATIENCE
    strategy   = args.strategy

    logging.info("Strategy : %s", strategy)
    logging.info("Model    : %s", model_name)
    logging.info("Epochs   : %d (patience=%d)", epochs, patience)

    # Handle dataset filtering: --dataset with no args means "all", --dataset X Y means "X and Y"
    target_datasets = set(args.dataset) if args.dataset is not None and len(args.dataset) > 0 else set()
    if target_datasets:
        logging.info("Datasets : %s", ", ".join(sorted(target_datasets)))
    else:
        logging.info("Datasets : all")

    if args.force:
        logging.info("Force re-training enabled (--force)")

    # 1. Warm tokenized cache for all datasets
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    dataset_ids = []
    for author, names in DATASETS.items():
        for name in names:
            dataset_id = f"{author}/{name}"

            # Filter by --dataset if specified
            if target_datasets and dataset_id not in target_datasets:
                continue

            try:
                get_tokenized_dataset(author, name, tokenizer,
                                      split="train", strategy=strategy,
                                      train_ratio=TRAIN_RATIO, seed=SEED)
                dataset_ids.append(dataset_id)
                logging.info("  ready: %s", dataset_id)
            except Exception as exc:
                logging.error("  SKIP %s: %s", dataset_id, exc)

    if not dataset_ids:
        logging.error("No datasets ready. Run `00_preprocessing.ipynb` first.")
        sys.exit(1)

    # 2. Build and execute training jobs sequentially
    models_dir = MODELS_DIR / strategy
    job_results = []

    for dataset_id in dataset_ids:
        author, name    = dataset_id.split("/", 1)
        train_cache     = get_tokenized_cache_path(author, name, "train", strategy)
        model_dir       = models_dir / dataset_id

        if args.dry_run:
            logging.info("  [dry-run] %s", dataset_id)
            continue

        # Execute training for this dataset
        import time
        start = time.monotonic()
        try:
            rv = train_one_job(dataset_id, str(train_cache), model_name, str(model_dir),
                               epochs, patience, force_retrain=args.force)
            elapsed = time.monotonic() - start
            skipped = rv.get("skipped", False)
            status = "skipped" if skipped else "ok"
            logging.info("  %-45s  %.0fs  %s", dataset_id, elapsed, status)
            job_results.append({
                "job_id": dataset_id,
                "success": True,
                "skipped": skipped,
                "elapsed_seconds": elapsed,
                "error": None,
            })
        except Exception as exc:
            elapsed = time.monotonic() - start
            logging.error("  %-45s  %.0fs  FAILED: %s", dataset_id, elapsed, exc)
            job_results.append({
                "job_id": dataset_id,
                "success": False,
                "skipped": False,
                "elapsed_seconds": elapsed,
                "error": str(exc),
            })

    if args.dry_run:
        return

    # 3. Save results JSON — preserve existing entries, update only newly run jobs
    results_path = models_dir / "train_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if results_path.exists():
        data = json.loads(results_path.read_text())
        existing = {e["job_id"]: e for e in data if e is not None}

    # Build map of newly run job results
    new_results = {}
    for r in job_results:
        if r["skipped"]:
            # Keep the existing entry if job was skipped
            if r["job_id"] in existing:
                new_results[r["job_id"]] = existing[r["job_id"]]
        else:
            # Update with newly run result
            new_results[r["job_id"]] = {
                "job_id": r["job_id"], "strategy": strategy, "model_name": model_name,
                "epochs": epochs, "patience": patience, "gpu_index": 0,
                "success": r["success"], "elapsed_seconds": round(r["elapsed_seconds"], 1),
                "elapsed_source": "measured",
                "error": r["error"],
            }

    # Preserve all existing entries that weren't in this run
    for job_id, entry in existing.items():
        if job_id not in new_results:
            new_results[job_id] = entry

    # Write all results (old + new) sorted by job_id
    summary = [new_results[jid] for jid in sorted(new_results.keys())]
    results_path.write_text(json.dumps(summary, indent=2))
    logging.info("Saved %s", results_path)

    if any(not r["success"] for r in job_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
