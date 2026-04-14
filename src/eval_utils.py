"""Evaluation helpers for Macro-F1 and TPR Gap metrics."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def compute_macro_f1_tpr_gap(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Compute Macro-F1 and absolute TPR gap between class 0 and class 1."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro"))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tpr_0_den = cm[0, 0] + cm[0, 1]
    tpr_1_den = cm[1, 1] + cm[1, 0]
    tpr_0 = float(cm[0, 0] / tpr_0_den) if tpr_0_den > 0 else 0.0
    tpr_1 = float(cm[1, 1] / tpr_1_den) if tpr_1_den > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "tpr_class_0": tpr_0,
        "tpr_class_1": tpr_1,
        "tpr_gap": abs(tpr_0 - tpr_1),
    }


def evaluate_model(model, eval_dataset, batch_size: int = 64, device: str = "cuda",
                   num_workers: int = 4):
    """Run inference and compute macro-f1 and tpr gap."""
    device_obj = torch.device(device)
    is_cuda = device_obj.type == "cuda"

    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=is_cuda,
    )
    model.to(device_obj)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device_obj, non_blocking=is_cuda) for k, v in batch.items()}
            with torch.autocast(device_type=device_obj.type, dtype=torch.float16,
                                enabled=is_cuda):
                outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    return compute_macro_f1_tpr_gap(all_labels, all_preds)


def evaluate_one_job(
    dataset_id: str,
    eval_cache_path: str,
    model_dir: str,
    repo_id: str = "PHMGC/roberta-bias-reduction",
    strategy: str = "baseline",
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    """Module-level subprocess entry point to evaluate model metrics."""
    from pathlib import Path
    from datasets import load_from_disk
    from transformers import AutoModelForSequenceClassification
    
    # Check if local weights exist
    local_path = Path(model_dir)
    has_weights = (local_path / "model.safetensors").exists() or (local_path / "pytorch_model.bin").exists()
    
    if has_weights:
        model = AutoModelForSequenceClassification.from_pretrained(str(local_path))
    else:
        # Fallback to hub
        subfolder = f"{strategy}/{dataset_id}"
        try:
            model = AutoModelForSequenceClassification.from_pretrained(repo_id, subfolder=subfolder)
        except Exception as e:
            raise RuntimeError(
                f"Model weights not found locally at '{local_path}' and could not be downloaded "
                f"from Hugging Face Hub under '{repo_id}' (subfolder='{subfolder}'). "
            )
    
    eval_tok = load_from_disk(eval_cache_path)
    metrics = evaluate_model(model, eval_tok, batch_size=batch_size, num_workers=num_workers)
    metrics["dataset_id"] = dataset_id
    
    return metrics

