"""Remove tokenized data and all models from HF Hub repos using root deletion.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

sys.path.append(str(Path(__file__).parent.parent))
from src.paths import ROOT_DIR

DATASETS_REPO = "PHMGC/roberta-bias-reduction-datasets"
MODELS_REPO   = "PHMGC/roberta-bias-reduction"

def safe_delete_folder(api: HfApi, path: str, repo_id: str, repo_type: str):
	try:
		print(f"Attempting to delete: {repo_id}/{path}...")
		api.delete_folder(path_in_repo=path, repo_id=repo_id, repo_type=repo_type)
		print(f"  Successfully deleted folder: {repo_id}/{path}")
	except Exception as e:
		# Silencia erro se a pasta não existir (404)
		if "404" in str(e):
			print(f"  Skip: folder '{path}' not found in {repo_id}.")
		else:
			print(f"  Error deleting {path}: {e}")

def safe_delete_file(api: HfApi, path: str, repo_id: str, repo_type: str):
	try:
		api.delete_file(path_in_repo=path, repo_id=repo_id, repo_type=repo_type)
		print(f"  Successfully deleted file: {repo_id}/{path}")
	except Exception as e:
		if "404" in str(e):
			print(f"  Skip: file '{path}' not found.")
		else:
			print(f"  Error deleting file {path}: {e}")

def main():
	token = os.getenv("HF_TOKEN")
	if not token:
		raise SystemExit("HF_TOKEN env var not set.")
	
	api = HfApi(token=token)

	# --- DATASETS CLEANUP ---
	print("\n=== Cleaning datasets repo ===")
	safe_delete_folder(api, "tokenized", DATASETS_REPO, "dataset")
	safe_delete_file(api, "all_class_distributions.json", DATASETS_REPO, "dataset")

	# --- MODELS CLEANUP ---
	print("\n=== Cleaning models repo ===")
	try:
		repo_files = api.list_repo_tree(repo_id=MODELS_REPO, repo_type="model")
		# Filtra apenas diretórios que não sejam pastas ocultas do sistema (se houver)
		top_level_folders = {f.path.split('/')[0] for f in repo_files if '/' in f.path}
		
		for folder in top_level_folders:
			safe_delete_folder(api, folder, MODELS_REPO, "model")
	except Exception as e:
		print(f"  Error listing or cleaning models repo: {e}")

	print("\nDone.")

if __name__ == "__main__":
	load_dotenv(ROOT_DIR / ".env")
	main()