import subprocess
import sys

model_path = "/root/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-qat-GGUF/snapshots/02749a7b272109255a4c559a80894d3d9777574c/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
# Correct turn boundary with newline between <turn|> and <|turn>
prompt = "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n"

print("Launching model with exact chat template formatting...")
cmd = [
    "./build/gemmaedge", "generate",
    model_path,
    prompt,
    "128"
]
subprocess.run(cmd)
