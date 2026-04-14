"""Raw dataset downloading helpers."""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path


_RIBEIRO_GITHUB_RAW = (
    "https://raw.githubusercontent.com/guilherme8426/"
    "ACL2025_Undersampling/refs/heads/main/resources/datasets"
)

_MCAULEY_URLS = {
    "luxury_beauty": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Luxury_Beauty.json.gz",
    "cds_reviews":   "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/CDs_and_Vinyl.json.gz",
    "digital_music": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Digital_Music.json.gz",
}

_MCAULEY_MAX_SAMPLES = {
    "luxury_beauty": None,
    "cds_reviews":   1_400_000,
    "digital_music": None,
}

_DOWNLOAD_TIMEOUT = (15, 120)  # (connect timeout, read timeout)


def download_raw_dataset(author_name: str, dataset_name: str):
    """Download and save a raw dataset to DATA_DIR/raw/{author}/{name}.

    If already cached locally, loads and returns it without downloading.
    Returns a DatasetDict (stanfordnlp) or Dataset (ribeiro, mcauley).
    """
    from datasets import load_from_disk
    from src.paths import DATA_DIR

    out_path = DATA_DIR / "raw" / author_name / dataset_name

    if out_path.exists():
        print(f"Já existe localmente: {out_path}")
        return load_from_disk(str(out_path))

    if author_name == "stanfordnlp":
        return _download_stanfordnlp(dataset_name, out_path)
    elif author_name == "ribeiro":
        return _download_ribeiro(dataset_name, out_path)
    elif author_name == "mcauley":
        return _download_mcauley(dataset_name, out_path)
    else:
        raise ValueError(
            f"Autor desconhecido '{author_name}'. Suportados: stanfordnlp, ribeiro, mcauley."
        )


def _download_stanfordnlp(dataset_name: str, out_path: Path):
    from datasets import load_dataset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(f"stanfordnlp/{dataset_name}")
    ds.save_to_disk(str(out_path))
    print(f"Salvo: {out_path}")
    return ds


def _download_ribeiro(dataset_name: str, out_path: Path):
    from datasets import Dataset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = f"{_RIBEIRO_GITHUB_RAW}/{dataset_name}_2L"
    with urllib.request.urlopen(f"{base_url}/texts.txt") as r:
        texts = r.read().decode("utf-8").splitlines()
    with urllib.request.urlopen(f"{base_url}/score.txt") as r:
        scores = r.read().decode("utf-8").splitlines()
    labels = [0 if int(s) == -1 else 1 for s in scores]
    ds = Dataset.from_dict({"text": texts, "label": labels})
    ds.save_to_disk(str(out_path))
    print(f"Salvo: {out_path}")
    return ds


def _download_mcauley(dataset_name: str, out_path: Path):
    import requests
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset

    if dataset_name not in _MCAULEY_URLS:
        raise ValueError(
            f"Dataset McAuley desconhecido: '{dataset_name}'. "
            f"Disponíveis: {list(_MCAULEY_URLS)}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = _MCAULEY_URLS[dataset_name]
    max_n = _MCAULEY_MAX_SAMPLES.get(dataset_name)
    gz_path = out_path.parent / f"{dataset_name}.json.gz"
    parquet_path = out_path.parent / f"{dataset_name}.parquet"
    writer = None

    def flush(b):
        nonlocal writer
        table = pa.Table.from_pylist(b)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema)
        writer.write_table(table)

    try:
        if not gz_path.exists():
            print(f"  Baixando {dataset_name}")
            with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(gz_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(
                                f"\r  {downloaded/1e6:.0f} MB / {total/1e6:.0f} MB "
                                f"({downloaded/total*100:.1f}%)",
                                end="", flush=True,
                            )
            print()

        parquet_path.unlink(missing_ok=True)
        print(f"  Processando {dataset_name}")
        batch, total_saved = [], 0

        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rating = obj.get("overall")
                text = obj.get("reviewText", "")
                if rating is None or rating == 3 or not text:
                    continue
                batch.append({"text": text, "label": 0 if rating <= 2 else 1})

                if len(batch) >= 50_000:
                    flush(batch)
                    total_saved += len(batch)
                    batch = []
                    print(f"\r  {total_saved:,} registros processados", end="", flush=True)
                    if max_n and total_saved >= max_n:
                        break

        if batch:
            flush(batch)
            total_saved += len(batch)

        if writer is not None:
            writer.close()
            writer = None
        print(f"\n  {total_saved:,} registros no total")
        ds = Dataset.from_parquet(str(parquet_path))
        ds.save_to_disk(str(out_path))
        print(f"  Salvo: {out_path}")
        return ds

    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Timeout ao baixar '{dataset_name}'. Tente novamente mais tarde."
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erro de rede ao baixar '{dataset_name}': {e}")
    finally:
        if writer is not None:
            writer.close()
        gz_path.unlink(missing_ok=True)
        parquet_path.unlink(missing_ok=True)
