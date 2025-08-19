# crystal_lattice/hf_upload.py
import os
from huggingface_hub import HfApi, create_repo, upload_folder

# ======================================================
# Hugging Face Upload (Static Lattice)
# ======================================================
def upload_lattice_to_hf(
    repo_id: str,
    local_dir: str = "out",
    token: str = None,
    private: bool = True,
    branch: str = None,
):
    if not token:
        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("No Hugging Face token found.")

    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private, token=token)

    if branch:
        api = HfApi(token=token)
        try:
            api.create_branch(repo_id=repo_id, branch=branch, repo_type="dataset")
        except Exception:
            pass

    print(f"[upload] pushing {local_dir} → https://huggingface.co/datasets/{repo_id}")
    upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=".",
        allow_patterns=["*.safetensors", "*.jsonl"],
        ignore_patterns=["*.npy", "*.json"],
        revision=branch if branch else None,
        commit_message="Initial lattice upload",
        token=token,
    )
