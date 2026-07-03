import subprocess
import sys

model_path = "/content/gemmaedge/model/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
# Correct turn boundary with newline between <turn|> and <|turn>
prompt = "<|turn>user\nWhat is quantum computing?<turn|>\n<|turn>model\n"

print("Launching model with exact chat template formatting...")
cmd = [
    "./build/gemmaedge", "generate",
    model_path,
    prompt,
    "128"
]
subprocess.run(cmd)
