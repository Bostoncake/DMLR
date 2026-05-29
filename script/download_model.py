import os
from huggingface_hub import snapshot_download

os.environ["HF_TOKEN"]="<HF_TOKEN>"

local_path = snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir="/WillDevExt/xiongyizhe/models/Llama-3.1-8B-Instruct",
    ignore_patterns=["*.md", "LICENSE-*"],  # keep everything important; skip docs if you like
    local_dir_use_symlinks=False            # set True to save disk via symlinks
)

print("Downloaded to:", local_path)