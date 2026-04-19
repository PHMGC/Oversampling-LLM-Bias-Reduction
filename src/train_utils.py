"""Training helpers shared by baseline and oversampling notebooks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.paths import MODELS_DIR


def train_loop(
    model,
    train_dataloader: Iterable,
    optimizer,
    device: torch.device,
    epochs: int = 20,
    patience: int = 5,
    val_dataloader: Optional[Iterable] = None,
    scheduler=None,
):
    """Supervised training loop with early stopping.

    Monitors validation loss when val_dataloader is provided; falls back to
    training loss otherwise.  Restores the best weights before returning.

    Returns:
        avg_loss: overall average training loss of the best epoch
    """
    model.to(device)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ── Training ─────────────────────────────────────────────────────────
        model.train()
        train_loss, train_steps = 0.0, 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [train]", dynamic_ncols=True):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            loss = outputs.loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            train_loss += float(loss.item())
            train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)

        # ── Validation ───────────────────────────────────────────────────────
        if val_dataloader is not None:
            model.eval()
            val_loss, val_steps = 0.0, 0
            with torch.inference_mode():
                for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [val]", leave=False, dynamic_ncols=True):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=(device.type == "cuda")):
                        outputs = model(**batch)
                    val_loss += float(outputs.loss.item())
                    val_steps += 1
            monitor_loss = val_loss / max(val_steps, 1)
            print(f"Epoch {epoch + 1}  train_loss={avg_train_loss:.4f}  val_loss={monitor_loss:.4f}")
        else:
            monitor_loss = avg_train_loss
            print(f"Epoch {epoch + 1}  train_loss={avg_train_loss:.4f}")

        # ── Early stopping ───────────────────────────────────────────────────
        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return best_val_loss


def train_model(
    model,
    train_dataset,
    tokenizer,
    output_dir,
    batch_size: int = 32,
    epochs: int = 20,
    patience: int = 5,
    learning_rate: float = 5e-5,
    val_fraction: float = 0.1,
    device: str = "cuda",
    num_workers: int = 4,
    repo_id: str = "PHMGC/roberta-bias-reduction"
):
    """High-level wrapper: splits off a val set, trains with early stopping, saves."""
    output_path = Path(output_dir)

    has_weights = (output_path / "model.safetensors").exists() or (output_path / "pytorch_model.bin").exists()
    if has_weights:
        print(f"Modelo já treinado localmente em: {output_path.name}")
        return model, True

    # Split train → train + val
    val_size = max(1, int(len(train_dataset) * val_fraction))
    train_size = len(train_dataset) - val_size
    train_split, val_split = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    is_cuda = device == "cuda"
    train_dl = DataLoader(train_split, shuffle=True, batch_size=batch_size,
                          num_workers=num_workers, pin_memory=is_cuda)
    val_dl   = DataLoader(val_split,   shuffle=False, batch_size=batch_size,
                          num_workers=num_workers, pin_memory=is_cuda)

    optimizer  = AdamW(model.parameters(), lr=learning_rate)
    device_obj = torch.device(device)

    total_steps  = epochs * len(train_dl)
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    train_loop(model, train_dl, optimizer, device_obj,
               epochs=epochs, patience=patience, val_dataloader=val_dl,
               scheduler=scheduler)

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

    return model, False


def train_one_job(
    dataset_id: str,
    train_cache_path: str,
    model_name: str,
    model_dir: str,
    epochs: int,
    patience: int,
) -> str:
    """Module-level subprocess entry point for run_parallel_jobs().

    Must be defined here (not in a notebook) so pickle can serialize it
    by dotted reference 'src.train_utils.train_one_job' for the spawn context.
    All heavy imports are deferred so they resolve after CUDA_VISIBLE_DEVICES
    is set by _worker_entry in parallel_utils.
    """
    from datasets import load_from_disk
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.data_utils import set_torch_format

    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    train_tok  = set_torch_format(load_from_disk(train_cache_path))
    model      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    _, skipped = train_model(
        model=model,
        train_dataset=train_tok,
        tokenizer=tokenizer,
        output_dir=model_dir,
        epochs=epochs,
        patience=patience,
    )
    return {"dataset_id": dataset_id, "skipped": skipped}
