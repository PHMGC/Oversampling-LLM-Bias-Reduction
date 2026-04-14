"""Training helpers shared by baseline and oversampling notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.paths import MODELS_DIR


def train_loop(
    model,
    dataloader: Iterable,
    optimizer,
    device: torch.device,
    epochs: int = 1,
):
    """Run a simple supervised training loop and return average loss."""
    model.to(device)
    model.train()

    total_loss = 0.0
    total_steps = 0
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for epoch in range(epochs):
        prev_loss, prev_steps = total_loss, total_steps
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in epoch_iterator:
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

            total_loss += float(loss.item())
            total_steps += 1

        epoch_steps = total_steps - prev_steps
        avg_epoch_loss = (total_loss - prev_loss) / max(epoch_steps, 1)
        print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

    avg_loss = total_loss / max(total_steps, 1)
    print(f"Overall Average Loss: {avg_loss:.4f}")
    return avg_loss


def train_model(
    model,
    train_dataset,
    tokenizer,
    output_dir,
    batch_size: int = 64,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    device: str = "cuda",
    num_workers: int = 4,
    repo_id: str = "PHMGC/roberta-bias-reduction"
):
    """High-level wrapper to train and save the model."""
    output_path = Path(output_dir)

    # 1. Check if model is already trained and available locally
    has_weights = (output_path / "model.safetensors").exists() or (output_path / "pytorch_model.bin").exists()
    if has_weights:
        print(f"Modelo já treinado localmente em: {output_path.name}")
        return model

    # Iniciação do dataloader e treinamento
    dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    device_obj = torch.device(device)

    train_loop(model, dataloader, optimizer, device_obj, epochs)

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

    return model


def train_one_job(
    dataset_id: str,
    train_cache_path: str,
    model_name: str,
    model_dir: str,
    epochs: int,
) -> str:
    """Module-level subprocess entry point for run_parallel_jobs().

    Must be defined here (not in a notebook) so pickle can serialize it
    by dotted reference 'src.train_utils.train_one_job' for the spawn context.
    All heavy imports are deferred so they resolve after CUDA_VISIBLE_DEVICES
    is set by _worker_entry in parallel_utils.
    """
    from datasets import load_from_disk
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    train_tok  = load_from_disk(train_cache_path)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_model(
        model=model,
        train_dataset=train_tok,
        tokenizer=tokenizer,
        output_dir=model_dir,
        epochs=epochs,
    )
    return dataset_id
