"""Evaluate synthetic paraphrase quality using BERTScore and BLEU.

Runs over all datasets and passes, reporting metrics only on paraphrases
that pass the BERTScore threshold (i.e. those actually used in training).
Results saved to DATA_DIR/paraphrase_quality.json.

    python scripts/eval_paraphrase_quality.py

BERTScore ≥ 0.85 → paraphrase preserves meaning.
BLEU in 20–50   → lexical diversity without semantic drift.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import evaluate
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DATASETS, SEED, TRAIN_RATIO
from src.data_utils import _load_raw_splits
from src.paths import DATA_DIR

# Must match paraphrasing.BERTSCORE_THRESHOLD
_BERTSCORE_THRESHOLD = 0.85


def main() -> None:
    bertscore_metric = evaluate.load("bertscore")
    bleu_metric = evaluate.load("bleu")

    quality_rows: list[dict] = []

    for author, dataset_names in DATASETS.items():
        for dataset_name in dataset_names:
            cache_dir = DATA_DIR / "paraphrase_cache" / author / dataset_name
            pass_files = sorted(cache_dir.glob("pass_*.jsonl"))
            if not pass_files:
                print(f"No cache found for {author}/{dataset_name} — skipping")
                continue

            raw_train, _ = _load_raw_splits(author, dataset_name, TRAIN_RATIO, SEED)
            labels = [int(x) for x in raw_train["label"]]
            minority_label = min(Counter(labels), key=Counter(labels).get)
            minority_texts = [
                raw_train[i]["text"]
                for i, l in enumerate(labels) if l == minority_label
            ]

            pass_scores: list[dict] = []
            for pass_file in tqdm(pass_files, desc=f"{dataset_name}", unit="pass"):
                pairs: list[dict] = []
                with open(pass_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            pairs.append(json.loads(line))

                if len(pairs) < 2:
                    continue

                n = min(len(pairs), len(minority_texts))
                originals = minority_texts[:n]
                paraphrases = [p["text"] for p in pairs[:n]]

                bs = bertscore_metric.compute(
                    predictions=paraphrases,
                    references=originals,
                    lang="en",
                    model_type="distilbert-base-uncased",
                )

                # Filter to only samples that pass the threshold — mirrors training pipeline
                accepted_idx = [i for i, s in enumerate(bs["f1"]) if s >= _BERTSCORE_THRESHOLD]
                if len(accepted_idx) < 2:
                    continue

                accepted_paraphrases = [paraphrases[i] for i in accepted_idx]
                accepted_originals   = [originals[i]   for i in accepted_idx]
                accepted_bs_f1       = [bs["f1"][i]    for i in accepted_idx]

                bl = bleu_metric.compute(
                    predictions=accepted_paraphrases,
                    references=[[ref] for ref in accepted_originals],
                )
                pass_scores.append({
                    "n_total":        n,
                    "n_accepted":     len(accepted_idx),
                    "acceptance_rate": len(accepted_idx) / n,
                    "BERTScore-F1":   float(np.mean(accepted_bs_f1)),
                    "BLEU":           bl["bleu"] * 100,
                })

            if not pass_scores:
                continue

            bs_vals   = [s["BERTScore-F1"]   for s in pass_scores]
            bleu_vals = [s["BLEU"]            for s in pass_scores]
            acc_vals  = [s["acceptance_rate"] for s in pass_scores]

            row = {
                "dataset":            dataset_name,
                "n_passes":           len(pass_scores),
                "acceptance_rate":    round(float(np.mean(acc_vals)), 4),
                "BERTScore-F1 mean":  round(float(np.mean(bs_vals)), 4),
                "BERTScore-F1 std":   round(float(np.std(bs_vals)), 4),
                "BLEU mean":          round(float(np.mean(bleu_vals)), 2),
                "BLEU std":           round(float(np.std(bleu_vals)), 2),
            }
            quality_rows.append(row)
            tqdm.write(
                f"  => accepted={row['acceptance_rate']:.1%} | "
                f"BERTScore-F1={row['BERTScore-F1 mean']} ±{row['BERTScore-F1 std']}, "
                f"BLEU={row['BLEU mean']} ±{row['BLEU std']}"
            )

    out_path = DATA_DIR / "paraphrase_quality.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(quality_rows, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\nSynthetic Sample Quality Metrics (accepted paraphrases only, mean ± std across passes):")
    header = (
        f"{'dataset':<30} {'n_passes':>8} {'accept%':>8} "
        f"{'BS-F1 mean':>12} {'BS-F1 std':>10} {'BLEU mean':>10} {'BLEU std':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in quality_rows:
        print(
            f"{row['dataset']:<30} {row['n_passes']:>8} {row['acceptance_rate']:>8.1%} "
            f"{row['BERTScore-F1 mean']:>12} {row['BERTScore-F1 std']:>10} "
            f"{row['BLEU mean']:>10} {row['BLEU std']:>9}"
        )


if __name__ == "__main__":
    main()
