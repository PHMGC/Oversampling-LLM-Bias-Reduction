import os, sys
from pathlib import Path
from huggingface_hub import HfApi
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from src.paths import ROOT_DIR

def _get_api() -> HfApi:
	token = os.environ.get("HF_TOKEN")
	if not token:
		raise SystemExit("HF_TOKEN env var not set. Run as: HF_TOKEN=hf_xxx python src/upload2hf.py")
	return HfApi(token=token)

def upload_all_to_hub(repo_id: str, local_dir: str, repo_type: str, message: str):
	"""
	Uploads an entire directory to Hugging Face in a single commit.
	The library automatically handles LFS and skips unchanged files.
	"""
	api = _get_api()
	print(f"Syncing {local_dir} to {repo_id}...")
	
	try:
		api.upload_folder(
			folder_path=local_dir,
			repo_id=repo_id,
			repo_type=repo_type,
			commit_message=message,
			# multi_commits=True # Opcional: Útil se o volume de dados for massivo (>50GB)
		)
		print(f"Done: {local_dir} → {repo_id}")
	except Exception as e:
		print(f"Error syncing {local_dir}: {e}")

if __name__ == "__main__":
	load_dotenv(ROOT_DIR / ".env")

	model_repo_id = "PHMGC/roberta-bias-reduction"
	dataset_repo_id = "PHMGC/roberta-bias-reduction-datasets"
	
	models_dir = ROOT_DIR / "models"
	data_dir   = ROOT_DIR / "data"

	# Upload de todos os modelos em um único commit
	upload_all_to_hub(
		repo_id=model_repo_id, 
		local_dir=str(models_dir), 
		repo_type="model", 
		message="Sync all models: baseline and oversampling"
	)

	# Upload de todos os datasets em um único commit
	upload_all_to_hub(
		repo_id=dataset_repo_id, 
		local_dir=str(data_dir), 
		repo_type="dataset", 
		message="Sync datasets"
	)