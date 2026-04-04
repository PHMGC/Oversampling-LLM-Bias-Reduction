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


def evaluate_model(model, eval_dataset, batch_size: int = 64, device: str = "cuda"):
    """Run inference and compute macro-f1 and tpr gap."""
    dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    device_obj = torch.device(device)

    model.to(device_obj)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device_obj) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    return compute_macro_f1_tpr_gap(all_labels, all_preds)
