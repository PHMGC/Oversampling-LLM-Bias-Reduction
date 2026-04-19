# Oversampling & LLM Bias Reduction

Experimental investigation of **oversampling** strategies to reduce classification bias in language models trained on imbalanced datasets — designed as a direct companion study to the ACL 2025 paper below.

## Reference Paper

> **Instance-Selection-Inspired Undersampling Strategies for Bias Reduction in Small and Large Language Models for Binary Text Classification**
> Guilherme Fonseca, Gabriel Prenassi, Washington Cunha, Marcos André Gonçalves, Leonardo Rocha
> *ACL 2025 — Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, pp. 9323–9340*

- PDF: [https://aclanthology.org/2025.acl-long.458.pdf](https://aclanthology.org/2025.acl-long.458.pdf)
- Codebase: [https://github.com/guilherme8426/ACL2025_Undersampling](https://github.com/guilherme8426/ACL2025_Undersampling)

This project is **intrinsically linked** to that work: it replicates the paper's experimental setup (same model, datasets, hyperparameters, and metrics) and extends it by evaluating **oversampling** and **LLM-based paraphrasing** as complementary bias-reduction strategies — areas the paper explicitly leaves as future work.

The baseline results in `01_baseline.ipynb` serve as a direct replication check against the paper's NoUnder numbers. The undersampling experiments in `04_undersampling.ipynb` reproduce the paper's methods (UBR, E2SC_US, CNN, NM1, NM2) for cross-comparison with the new oversampling strategies.

## Motivation

Models fine-tuned on **class-imbalanced data** tend to favor the majority class. Even when **Macro-F1** looks acceptable, the **TPRGap** (difference between the true positive rates of each class) can be high, revealing model bias. This project evaluates four balancing strategies for binary sentiment classification.

## Metrics

| Metric | Description |
|---|---|
| **Macro-F1** | Unweighted average F1 across classes. Penalizes models that ignore the minority. |
| **TPRGap** | `abs(TPR_class_0 - TPR_class_1)`. Lower means fairer predictions. |
| **Imbalance Ratio** | `majority / minority`. Quantifies how skewed each dataset is. |
| **Speedup** | `baseline_time / method_time`. Higher means more efficient than baseline. |

## Datasets

The same 4 datasets used in the paper (subset of the full 13), chosen to cover a range of imbalance levels:

| Dataset | Paper ID | Source | IR (approx.) |
|---|---|---|---|
| `ribeiro/sentistrength_myspace` | I | Ribeiro et al. (SentiBench) | 5.3× |
| `mcauley/luxury_beauty` | K | McAuley Lab | 10.7× |
| `mcauley/cds_reviews` | L | McAuley Lab | 13.8× |
| `mcauley/digital_music` | M | McAuley Lab | 39.7× |

## Base Model and Hyperparameters

| Parameter | Value |
|---|---|
| Model | `roberta-base` |
| Learning rate | `5e-5` |
| Max epochs | `20` (early stopping, patience `5`) |
| Max length | `256` |
| Batch size | `32` |
| LR scheduler | linear warmup + linear decay (6% warmup steps) |

## Experiments

```
exps/
├── 00_preprocessing.ipynb       # download and prepare datasets
├── 01_baseline.ipynb            # no balancing (baseline) — replication check vs. paper
├── 02_simple_oversampling.ipynb # duplicate minority samples
├── 03_llm_paraphrasing.ipynb    # synthesize minority samples via LLM
├── 04_undersampling.ipynb       # reproduce paper's undersampling methods
└── 05_conclusion.ipynb          # results summary and discussion
```

The project pipeline was designed to run each notebook sequentially, though it is not strictly necessary.

## Setup

> **CUDA recommended.** Install PyTorch with CUDA support first: https://pytorch.org/get-started/locally/

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Structure

```
scripts/
├── train.py      # train a model with specified balancing strategy
├── eval.py       # compute metrics on saved predictions
└── upload2hf.py  # sync local data/models to HF Hub

src/
├── paths.py           # ROOT_DIR, DATA_DIR, MODELS_DIR
├── download_utils.py  # raw dataset downloading
├── data_utils.py      # tokenization and cache (local + HF Hub)
├── train_utils.py     # training loop, AMP, model saving
├── eval_utils.py      # Macro-F1 and TPRGap
└── plot_utils.py      # bar chart visualizations

data/    # generated — git-ignored
models/  # generated — git-ignored
```

Tokenized datasets and trained models are cached under `data/` and `models/`, and synced to `PHMGC/roberta-bias-reduction-datasets` and `PHMGC/roberta-bias-reduction` on the HF Hub to avoid reprocessing across machines.
