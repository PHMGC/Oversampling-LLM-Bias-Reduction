# Oversampling & LLM Bias Reduction

Experimental investigation of strategies to reduce classification bias in language models trained on imbalanced datasets.

## Motivation

Models fine-tuned on **class-imbalanced data** tend to favor the majority class. Even when macro-F1 looks acceptable, the **TPRGap** (difference between the true positive rates of each class) can be high, revealing model bias. This project evaluates four balancing strategies for binary sentiment classification.

## Metrics

| Metric | Description |
|---|---|
| **Macro-F1** | Unweighted average F1 across classes. Penalizes models that ignore the minority. |
| **TPRGap** | `abs(TPR_class_0 - TPR_class_1)`. Lower means fairer predictions. |
| **Imbalance Ratio** | `majority / minority`. Quantifies how skewed each dataset is. |

## Datasets

| Dataset | Source | IR (approx.) |
|---|---|---|
| `stanfordnlp/imdb` | HF Hub | 1.0× (balanced control) |
| `ribeiro/sentistrength_myspace` | GitHub (ACL 2025) | 5.3× |
| `mcauley/luxury_beauty` | McAuley Lab | 5.7× |
| `mcauley/cds_reviews` | McAuley Lab | 12.4× |
| `mcauley/digital_music` | McAuley Lab | 21.9× |

Base model: **`roberta-base`** (fine-tuned per experiment).

## Experiments

```
exps/
├── 00_preprocessing.ipynb       # download and prepare datasets
├── 01_baseline.ipynb            # no balancing (baseline)
├── 02_simple_oversampling.ipynb # duplicate minority samples
├── 03_llm_paraphrasing.ipynb   # synthesize minority samples via LLM
└── 04_undersampling.ipynb       # downsample the majority class
```

The project pipeline was designed to run each notebook sequentially, though it is not strictly necessary.

## Setup

> **CUDA recommended.** Install PyTorch with CUDA support first: https://pytorch.org/get-started/locally/

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

To sync artifacts to the Hugging Face Hub:

```bash
python src/upload2hf.py   # needs hf repo token
```

## Structure

```
src/
├── paths.py           # ROOT_DIR, DATA_DIR, MODELS_DIR
├── download_utils.py  # raw dataset downloading
├── data_utils.py      # tokenization and cache (local + HF Hub)
├── train_utils.py     # training loop, AMP, model saving
├── eval_utils.py      # Macro-F1 and TPRGap
├── plot_utils.py      # bar chart visualizations
└── upload2hf.py       # HF Hub sync

data/    # generated — git-ignored
models/  # generated — git-ignored
```

Tokenized datasets and trained models are cached under `data/` and `models/`, and synced to `PHMGC/roberta-bias-reduction-datasets` and `PHMGC/roberta-bias-reduction` on the HF Hub to avoid reprocessing across machines.
