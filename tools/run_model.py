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
    "1024"
]

# Run process and stream stdout to Python's sys.stdout so that Jupyter/Colab
# can capture and print the output to the notebook cell in real-time.
try:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Read output line-by-line as it becomes available
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
        
    process.wait()
    sys.exit(process.returncode)
except Exception as e:
    print(f"Error running model: {e}", file=sys.stderr)
    sys.exit(1)
