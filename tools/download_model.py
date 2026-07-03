from huggingface_hub import hf_hub_download
import os

repo = "unsloth/gemma-4-26B-A4B-it-qat-GGUF"
filename = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
dest_dir = "/content/gemmaedge/model"

print(f"Downloading {filename} from {repo} to {dest_dir}...")
os.makedirs(dest_dir, exist_ok=True)
hf_hub_download(
    repo_id=repo,
    filename=filename,
    local_dir=dest_dir
)
print("Download completed successfully!")
