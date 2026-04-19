"""Remove tokenized/model data from HF Hub repos.
"""
import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.paths import ROOT_DIR


from huggingface_hub import HfApi
from src.config import DATASETS, STRATEGIES

DATASETS_REPO = "PHMGC/roberta-bias-reduction-datasets"
MODELS_REPO   = "PHMGC/roberta-bias-reduction"
SPLITS = ["train", "test"]


def safe_delete_folder(api, path, repo_id, repo_type):
    try:
        api.delete_folder(path_in_repo=path, repo_id=repo_id, repo_type=repo_type)
        print(f"  deleted folder: {repo_id}/{path}")
    except Exception as e:
        print(f"  skip (not found or error): {repo_id}/{path} — {e}")


def safe_delete_file(api, path, repo_id, repo_type):
    try:
        api.delete_file(path_in_repo=path, repo_id=repo_id, repo_type=repo_type)
        print(f"  deleted file: {repo_id}/{path}")
    except Exception as e:
        print(f"  skip (not found or error): {repo_id}/{path} — {e}")


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN env var not set.")
    api = HfApi(token=token)

    print("\n=== Cleaning datasets repo ===")

    for strategy in STRATEGIES:
        for author, dataset_names in DATASETS.items():
            for ds in dataset_names:
                for split in SPLITS:
                    path = f"tokenized/{strategy}/{author}/{ds}/{split}"
                    safe_delete_folder(api, path, DATASETS_REPO, "dataset")

    safe_delete_file(api, "all_class_distributions.json", DATASETS_REPO, "dataset")

    print("\n=== Cleaning models repo ===")

    for strategy in STRATEGIES:
        for author in DATASETS:
            safe_delete_folder(api, f"{strategy}/{author}", MODELS_REPO, "model")

    print("\nDone.")


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    
    main()
