"""Experiment-wide constants shared by training scripts and analysis notebooks."""

MODEL_NAME    = "roberta-base"
TRAIN_RATIO   = 0.8
SEED          = 42
N_EPOCHS      = 20       # max epochs (paper: up to 20 with early stopping)
PATIENCE      = 5        # early stopping patience (paper: 5 epochs)
BATCH_SIZE    = 32       # paper: grid search over {16, 32, 64}
LEARNING_RATE = 5e-5     # paper: 5e-5
MAX_LENGTH    = 256      # paper: grid search over {150, 256}

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
}
