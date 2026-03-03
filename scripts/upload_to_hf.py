import os
import argparse
from huggingface_hub import HfApi

def upload_to_hf(repo_id, folder_path, path_in_repo="data"):
    """
    Uploads a folder to a Hugging Face Dataset repository.
    """
    api = HfApi()
    
    print(f"[*] Creating/Accessing repository: {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
    except Exception as e:
        print(f"[!] Error creating repo (might already exist or permission error): {e}")

    print(f"[*] Uploading folder '{folder_path}' to '{repo_id}/{path_in_repo}'...")
    
    try:
        api.upload_folder(
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            delete_patterns="*.pt", # This ensures files deleted locally are removed from remote
        )
        print("[+] Upload successful! (Remote mirrored to local)")
        print(f"[+] View your data at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"[!] Error during upload: {e}")
        print("[!] Make sure you have set the HF_TOKEN environment variable with WRITE permissions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload preprocessed data to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID (e.g., 'username/vox2guit-data')")
    parser.add_argument("--folder", type=str, default="data/processed", help="Local folder to upload (default: data/processed)")
    parser.add_argument("--path_in_repo", type=str, default="data", help="Path inside the HF repo")
    
    args = parser.parse_args()
    
    if "HF_TOKEN" not in os.environ:
        print("[!] WARNING: HF_TOKEN not found in environment variables.")
        print("[!] Please run: export HF_TOKEN=your_token_here")
    
    upload_to_hf(args.repo_id, args.folder, args.path_in_repo)
