import os
import json
import argparse
import random

from huggingface_hub import HfApi


def do_upload_model(folder_path, repo_id):
    # model_path = "/leonardo_work/EUHPC_E03_068/safellm/results/results/seven_epochs_train_experiment-2" 
    # Instantiate the API
    api = HfApi()

    # Upload files
    import os
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file,  # File path inside the repo
                repo_id=repo_id,
                repo_type="model",
            )
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor log file for completion status.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repo ID")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the directory to upload")
    parser.add_argument("--hf_type", type=str, default="model", help="what type?")

    args = parser.parse_args()
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=args.hf_type, exist_ok=True)
    if args.hf_type == "model":
        do_upload_model(args.folder_path, args.repo_id)
    else:
        NotImplementedError