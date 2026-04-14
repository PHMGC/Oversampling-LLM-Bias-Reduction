from pathlib import Path
from huggingface_hub import HfApi, login

def upload_models_to_hub(repo_id: str, models_base_dir: str):
    """
    Uploads all trained models from a local directory to a single Hugging Face repository,
    using a different revision (branch) for each model.
    """
    api = HfApi()
    base_path = Path(models_base_dir)

    model_dirs = list({
        p.parent
        for pattern in ("model.safetensors", "pytorch_model.bin")
        for p in base_path.rglob(pattern)
    })
    print(f"Found {len(model_dirs)} models with weights to upload.")

    for model_dir in model_dirs:
        rel_path = model_dir.relative_to(base_path)

        # Skip if already on Hub
        try:
            from huggingface_hub import hf_hub_url, get_hf_file_metadata
            get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename=f"{rel_path}/config.json"))
            print(f"  {rel_path}: already on Hub, skipping.")
            continue
        except Exception:
            pass

        print(f"\nUploading: {rel_path}")
        try:
            api.upload_folder(
                folder_path=str(model_dir),
                repo_id=repo_id,
                path_in_repo=str(rel_path),
                repo_type="model",
                commit_message=f"Upload model for {rel_path}"
            )
            print(f"  Done: {rel_path}")
        except Exception as e:
            print(f"  Error uploading {rel_path}: {e}")

def upload_datasets_to_hub(repo_id: str, data_dir: str):
    """
    Uploads the entire data/ folder to a HuggingFace Dataset repository,
    preserving raw/ and tokenized/ subfolders. Unchanged files are skipped automatically.
    """
    api = HfApi()
    api.upload_folder(
        folder_path=data_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Sync datasets",
    )
    print(f"Done: {data_dir} → {repo_id}")

if __name__ == "__main__":
    from paths import ROOT_DIR

    model_repo_id = "PHMGC/roberta-bias-reduction"
    dataset_repo_id = "PHMGC/roberta-bias-reduction-datasets"
    models_dir = ROOT_DIR / "models"
    data_dir   = ROOT_DIR / "data"

    login()

    print("Uploading Models")
    upload_models_to_hub(model_repo_id, str(models_dir))

    print("\nUploading Datasets")
    upload_datasets_to_hub(dataset_repo_id, str(data_dir))
