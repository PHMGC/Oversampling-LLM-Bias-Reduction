#!/usr/bin/env python3
"""Train model across all datasets for a given balancing strategy.

Usage:
    python scripts/train.py                        # baseline
    python scripts/train.py --strategy baseline
    python scripts/train.py --epochs 5 --gpu 0,1
    python scripts/train.py --dry-run              # print job plan only
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

from src.config import STRATEGIES  # lightweight, safe to import early

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",   required=True, choices=STRATEGIES)
    p.add_argument("--model-name", default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--gpu",          default=None, help="Override CUDA_VISIBLE_DEVICES, e.g. '0,1'")
    p.add_argument("--max-parallel", type=int, default=1,
                   help="Max concurrent jobs per GPU (default 1 = fastest for compute-bound training)")
    p.add_argument("--dry-run",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.max_parallel < 1:
        logging.error("--max-parallel must be >= 1")
        sys.exit(1)

    from transformers import AutoTokenizer
    from src.config import DATASETS, MODEL_NAME, N_EPOCHS, SEED, TRAIN_RATIO
    from src.data_utils import get_tokenized_cache_path, get_tokenized_dataset
    from src.parallel_utils import JobSpec, estimate_job_memory_gb, run_parallel_jobs
    from src.paths import MODELS_DIR
    from src.train_utils import train_one_job

    model_name = args.model_name or MODEL_NAME
    epochs     = args.epochs or N_EPOCHS
    strategy   = args.strategy

    logging.info("Strategy : %s", strategy)
    logging.info("Model    : %s", model_name)
    logging.info("Epochs   : %d", epochs)

    # 1. Warm tokenized cache for all datasets
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    dataset_ids = []
    for author, names in DATASETS.items():
        for name in names:
            dataset_id = f"{author}/{name}"
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

    # 2. Build job specs
    weight_gb  = estimate_job_memory_gb(model_name)
    models_dir = MODELS_DIR / strategy
    jobs = []
    for dataset_id in dataset_ids:
        author, name    = dataset_id.split("/", 1)
        train_cache     = get_tokenized_cache_path(author, name, "train", strategy)
        model_dir       = models_dir / dataset_id
        jobs.append(JobSpec(
            job_id=dataset_id,
            args=(dataset_id, str(train_cache), model_name, str(model_dir), epochs),
            weight_gb=weight_gb,
        ))

    if args.dry_run:
        for j in jobs:
            logging.info("  %s  (%.2f GB)", j.job_id, j.weight_gb)
        return

    # 3. Dispatch
    job_results = run_parallel_jobs(jobs, train_fn=train_one_job,
                                    max_jobs_per_gpu=args.max_parallel)

    # 4. Save results JSON
    results_path = models_dir / "train_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in job_results:
        status = "ok" if r.success else f"FAILED: {r.error}"
        logging.info("  %-45s  gpu=%d  %.0fs  %s",
                     r.job_id, r.gpu_index, r.elapsed_seconds, status)
        summary.append({
            "job_id": r.job_id, "strategy": strategy, "model_name": model_name,
            "epochs": epochs, "gpu_index": r.gpu_index,
            "success": r.success, "elapsed_seconds": round(r.elapsed_seconds, 1),
            "error": r.error,
        })
    results_path.write_text(json.dumps(summary, indent=2))
    logging.info("Saved %s", results_path)

    if any(not r.success for r in job_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
