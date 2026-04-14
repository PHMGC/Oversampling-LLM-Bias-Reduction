"""Experiment-wide constants shared by training scripts and analysis notebooks."""

MODEL_NAME    = "roberta-base"
TRAIN_RATIO   = 0.8
SEED          = 42
N_EPOCHS      = 3
BATCH_SIZE    = 64
LEARNING_RATE = 5e-5

# Balancing strategies available in the experiment
STRATEGIES: list[str] = [
    "baseline",
    "simple_oversampling",
    "llm_paraphrasing",
    "undersampling",
]

# {author: [dataset_name, ...]} — mirrors the notebook's DATASETS dict exactly
DATASETS = {
    "ribeiro": [
        "sentistrength_myspace",
    ],
    "mcauley": [
        "luxury_beauty",
        "cds_reviews",
        "digital_music",
    ],
    "stanfordnlp": [
        "imdb",
    ],
}
