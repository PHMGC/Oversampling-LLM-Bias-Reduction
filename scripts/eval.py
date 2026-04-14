"""Evaluate model across all datasets for a given balancing strategy.

Usage:
    python scripts/eval.py                        # baseline
    python scripts/eval.py --strategy baseline
    python scripts/eval.py --gpu 0,1
    python scripts/eval.py --dry-run              # print job plan only
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
    p.add_argument("--gpu",          default=None, help="Override CUDA_VISIBLE_DEVICES, e.g. '0,1'")
    p.add_argument("--max-parallel", type=int, default=1,
                   help="Max concurrent jobs per GPU (default 1)")
    p.add_argument("--batch-size",   type=int, default=64,
                   help="Inference batch size (try 96/128/160 for tuning)")
    p.add_argument("--num-workers",  type=int, default=4,
                   help="DataLoader worker processes for data loading (default 4)")
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
    from src.config import DATASETS, MODEL_NAME, SEED, TRAIN_RATIO
    from src.data_utils import get_tokenized_cache_path, get_tokenized_dataset
    from src.parallel_utils import JobSpec, estimate_job_memory_gb, run_parallel_jobs
    from src.paths import MODELS_DIR
    from src.eval_utils import evaluate_one_job

    model_name = args.model_name or MODEL_NAME
    strategy   = args.strategy

    logging.info("Strategy : %s", strategy)
    logging.info("Model    : %s", model_name)

    # 1. Provide test datatsets
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    dataset_ids = []
    
    for author, names in DATASETS.items():
        for name in names:
            dataset_id = f"{author}/{name}"
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

    # 2. Build job specs
    weight_gb  = estimate_job_memory_gb(model_name)
    models_dir = MODELS_DIR / strategy
    jobs = []
    
    for dataset_id in dataset_ids:
        author, name    = dataset_id.split("/", 1)
        # Note: Evaluation is always done on 'baseline' strategy tokenized dataset 
        # (the real test set), not necessarily the oversampled one. 
        eval_cache      = get_tokenized_cache_path(author, name, "test", "baseline")
        model_dir       = models_dir / dataset_id
        jobs.append(JobSpec(
            job_id=dataset_id,
            args=(dataset_id, str(eval_cache), str(model_dir)),
            kwargs={"strategy": strategy,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers},
            weight_gb=weight_gb,
        ))

    if args.dry_run:
        for j in jobs:
            logging.info("  %s  (%.2f GB)", j.job_id, j.weight_gb)
        return

    # 3. Dispatch
    job_results = run_parallel_jobs(jobs, train_fn=evaluate_one_job,
                                    max_jobs_per_gpu=args.max_parallel)

    # 4. Save results JSON
    results_path = models_dir / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for r in job_results:
        status = "ok" if r.success else f"FAILED: {r.error}"
        logging.info("  %-45s  gpu=%d  %.0fs  %s",
                     r.job_id, r.gpu_index, r.elapsed_seconds, status)
        if r.success:
            # We used dict internally and returned it
            metrics = r.return_value
            name_only = r.job_id.split("/")[1]
            summary[name_only] = {k: v for k, v in metrics.items() if k != "dataset_id"}

    results_path.write_text(json.dumps(summary, indent=2))
    logging.info("Saved evaluation metrics to %s", results_path)

    if any(not r.success for r in job_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
