"""Data loading and tokenization helpers for experiments."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Sequence


_HF_DATASET_REPO = "PHMGC/roberta-bias-reduction-datasets"
_CLASS_DISTRIBUTIONS_FILE = "all_class_distributions.json"


def save_class_distribution(author_name: str, dataset_name: str, ds) -> Dict[int, int]:
    """Compute and persist class distribution into the shared class_distributions.json."""
    import json
    from src.paths import DATA_DIR
    from datasets import DatasetDict

    if isinstance(ds, DatasetDict):
        all_labels = [label for split in ds.values() for label in split["label"]]
    else:
        all_labels = list(ds["label"])

    all_labels = [l for l in all_labels if l >= 0]
    counts = compute_class_distribution(all_labels)
    total = sum(counts.values())
    minority = min(counts.values())
    majority = max(counts.values())
    entry = {str(k): v for k, v in counts.items()}
    entry["total"] = total
    entry["imbalance_ratio"] = majority / minority

    out = DATA_DIR / _CLASS_DISTRIBUTIONS_FILE
    data = json.loads(out.read_text()) if out.exists() else {}
    data[f"{author_name}/{dataset_name}"] = entry
    out.write_text(json.dumps(data, indent=2))
    return counts


def load_class_distribution(author_name: str, dataset_name: str) -> dict:
    """Load class distribution from local file or HF cache.

    Resolution order:
    1. data/all_class_distributions.json (project-local)
    2. HF cache only (no network)
    3. Download from HF Hub (stored in HF cache)
    """
    import json
    from src.paths import DATA_DIR

    key = f"{author_name}/{dataset_name}"
    out = DATA_DIR / _CLASS_DISTRIBUTIONS_FILE

    if out.exists():
        data = json.loads(out.read_text())
        if key in data:
            return data[key]

    # 2) Try HF cache only (no download)
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            repo_id=_HF_DATASET_REPO,
            repo_type="dataset",
            filename=_CLASS_DISTRIBUTIONS_FILE,
            local_files_only=True,
        )
        data = json.loads(Path(cached).read_text())
        if key in data:
            return data[key]
    except Exception:
        pass

    # 3) Fallback: download to HF cache (do not copy into DATA_DIR)
    try:
        local = hf_hub_download(
            repo_id=_HF_DATASET_REPO,
            repo_type="dataset",
            filename=_CLASS_DISTRIBUTIONS_FILE,
        )
        data = json.loads(Path(local).read_text())
        if key in data:
            return data[key]
    except Exception as e:
        raise FileNotFoundError(
            f"Chave '{key}' não encontrada em {_CLASS_DISTRIBUTIONS_FILE}. "
            "Execute 00_preprocessing.ipynb ou verifique o HF Hub."
        ) from e

    raise FileNotFoundError(
        f"Chave '{key}' não encontrada em {_CLASS_DISTRIBUTIONS_FILE}."
    )


def compute_class_distribution(labels: Sequence) -> Dict[int, int]:
    """Return class counts for a label sequence."""
    return dict(Counter(int(l) for l in labels))


def get_tokenized_dataset(
    author_name: str,
    dataset_name: str,
    tokenizer,
    split: str,
    strategy: str = "baseline",
    hf_repo: str = _HF_DATASET_REPO,
    max_length: int = 256,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """Return a tokenized Dataset ready for training or evaluation.

    Resolution order:
    1. Local tokenized cache  (DATA_DIR/tokenized/{author}__{name}_{split}_{strategy})
    2. HF Hub tokenized dataset
    3. Local raw dataset → tokenize → save to cache
    4. RuntimeError asking to run 00_preprocessing.ipynb

    Args:
        author_name:  e.g. "ribeiro", "mcauley", "stanfordnlp"
        dataset_name: e.g. "sentistrength_myspace"
        tokenizer:    HuggingFace tokenizer instance
        split:        "train" or "test"
        strategy:     experiment label used in cache folder name, e.g. "baseline"
        hf_repo:      HuggingFace Hub dataset repo ID
        max_length:   max token length for padding/truncation
        train_ratio:  train fraction when the raw dataset has no native train/test split
        seed:         random seed for splitting
    """
    from datasets import load_from_disk

    cache_path = _tokenized_cache_path(author_name, dataset_name, split, strategy)

    # 1. Local cache
    if cache_path.exists():
        print(f"Cache local: {cache_path.name}")
        return load_from_disk(str(cache_path))

    # 2. HF Hub (stored under tokenized/{name} to mirror local structure)
    try:
        from huggingface_hub import snapshot_download
        hub_folder = f"tokenized/{cache_path.name}"
        local_hub = snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            allow_patterns=f"{hub_folder}/*",
        )
        hub_path = Path(local_hub) / hub_folder
        if hub_path.is_dir():
            hub_ds = load_from_disk(str(hub_path))
            print(f"Tokenizado resolvido do Hub localmente ({hf_repo}/{hub_folder})")
            return hub_ds
        print(f"[HF Hub] Pasta não encontrada no repo: {hub_folder}")
    except Exception as e:
        print(f"[HF Hub] Falha ao baixar tokenizado: {e}")

    # 3. Tokenize from local raw — only allowed for baseline; other strategies must
    # prepare their datasets in the corresponding notebook before running train.py.
    if strategy != "baseline":
        raise FileNotFoundError(
            f"No cached dataset found for strategy '{strategy}' "
            f"({author_name}/{dataset_name}, split={split}). "
            f"Run the corresponding strategy notebook first."
        )

    train_split, test_split = _load_raw_splits(author_name, dataset_name, train_ratio, seed)
    raw = train_split if split == "train" else test_split
    return _format_and_save(raw, tokenizer, cache_path, max_length)


# ── Private helpers ───────────────────────────────────────────────────────────

def get_tokenized_cache_path(author_name: str, dataset_name: str, split: str, strategy: str) -> Path:
    """Return the actual directory path where the tokenized dataset is stored.
    Checks local DATA_DIR first, then the HuggingFace Hub cache. If neither exists,
    returns the fallback local path for saving.
    """
    from src.paths import DATA_DIR
    local_path = DATA_DIR / "tokenized" / strategy / author_name / dataset_name / split

    if local_path.is_dir():
        return local_path

    try:
        from huggingface_hub import snapshot_download
        hub_folder = f"tokenized/{strategy}/{author_name}/{dataset_name}/{split}"
        local_hub = snapshot_download(
            repo_id="PHMGC/roberta-bias-reduction-datasets",
            repo_type="dataset",
            allow_patterns=f"{hub_folder}/*",
        )
        hub_path = Path(local_hub) / hub_folder
        if hub_path.is_dir():
            return hub_path
    except Exception:
        pass
        
    return local_path


# Keep the private alias for internal use -> Resolvendo para o local ou HF CACHE auto
_tokenized_cache_path = get_tokenized_cache_path


def _load_raw_splits(author_name: str, dataset_name: str, train_ratio: float, seed: int):
    """Load raw dataset from local disk (or HF Hub fallback) and return (train, test) splits."""
    from datasets import load_from_disk, Dataset
    from src.paths import DATA_DIR

    raw_path = DATA_DIR / "raw" / author_name / dataset_name
    is_valid = lambda p: (p / "dataset_info.json").exists() or (p / "dataset_dict.json").exists()

    if not raw_path.exists() or not is_valid(raw_path):
        # Try downloading raw dataset from HF Hub
        try:
            from huggingface_hub import snapshot_download
            hub_folder = f"raw/{author_name}/{dataset_name}"
            local_hub = snapshot_download(
                repo_id=_HF_DATASET_REPO,
                repo_type="dataset",
                allow_patterns=f"{hub_folder}/*",
            )
            hub_path = Path(local_hub) / hub_folder
            if hub_path.exists() and is_valid(hub_path):
                raw_path = hub_path
                print(f"Raw dataset resolvido do Hub ({_HF_DATASET_REPO}/{hub_folder})")
            else:
                raise FileNotFoundError(f"Pasta não encontrada no Hub: {hub_folder}")
        except Exception as e:
            raise RuntimeError(
                f"Dataset '{author_name}/{dataset_name}' não encontrado localmente nem no HF Hub.\n"
                "Execute `00_preprocessing.ipynb` primeiro para baixar os dados crus."
            ) from e

    raw = load_from_disk(str(raw_path))

    if isinstance(raw, Dataset):
        src = raw
    elif "train" in raw and "test" in raw:
        return raw["train"], raw["test"]
    elif "train" in raw:
        src = raw["train"]
    else:
        raise ValueError(f"Splits válidos não encontrados em '{author_name}/{dataset_name}'.")

    s = src.train_test_split(test_size=1 - train_ratio, seed=seed)
    return s["train"], s["test"]


def _format_and_save(dataset, tokenizer, cache_path: Path, max_length: int):
    """Tokenize, set PyTorch format, save to cache_path, and return."""
    tokenized = dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
        desc=f"Tokenizando {cache_path.name}",
    )
    if "label" in tokenized.column_names:
        tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(cache_path))
    print(f"Tokenizado e salvo: {cache_path.name}")
    return tokenized


def set_torch_format(dataset):
    """Ensure a tokenized dataset returns PyTorch tensors for model inputs."""
    columns = ["input_ids", "attention_mask", "labels"]
    existing_columns = [c for c in columns if c in dataset.column_names]
    if existing_columns:
        dataset.set_format("torch", columns=existing_columns)
    return dataset
