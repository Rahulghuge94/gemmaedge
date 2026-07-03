import os
import sys
import subprocess
import urllib.request
import time
import threading

# 1. Install dependencies programmatically in Python
try:
    import fastapi
    import uvicorn
    import nest_asyncio
    import pydantic
    from fastapi import FastAPI, UploadFile, File, Header, HTTPException
    from pydantic import BaseModel
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "fastapi", "uvicorn", "pydantic", "nest-asyncio", "python-multipart"
    ])
    import fastapi
    import uvicorn
    import nest_asyncio
    import pydantic
    from fastapi import FastAPI, UploadFile, File, Header, HTTPException
    from pydantic import BaseModel

# Allow Uvicorn to run inside the notebook loop
nest_asyncio.apply()

app = FastAPI()

# Security token
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "gemmaedge-secret-token")
print(f"Server security token is: {AUTH_TOKEN}")

# Track persistent working directory
current_working_dir = "/content"
if not os.path.exists(current_working_dir):
    current_working_dir = os.getcwd()


class CodeExecutionRequest(BaseModel):
    command: str


@app.post("/exec")
async def execute_command(req: CodeExecutionRequest, x_auth_token: str = Header(None)):
    global current_working_dir
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        return {
            "stdout": "",
            "stderr": "Error: Unauthorized (invalid X-Auth-Token header).\n",
            "exit_code": 401,
            "cwd": current_working_dir
        }
        
    try:
        # Append a command to print the CWD with a marker
        marker = "__CWD_MARKER__"
        full_command = f"{req.command}\necho '{marker}'\npwd"

        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=current_working_dir
        )

        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode

        # Parse CWD from stdout
        if marker in stdout:
            parts = stdout.split(marker)
            user_stdout = parts[0]
            new_cwd = parts[1].strip()
            if os.path.exists(new_cwd):
                current_working_dir = new_cwd
        else:
            user_stdout = stdout

        return {
            "stdout": user_stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "cwd": current_working_dir,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Error: Command timed out.",
            "exit_code": 124,
            "cwd": current_working_dir
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Error: {str(e)}",
            "exit_code": 1,
            "cwd": current_working_dir
        }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), x_auth_token: str = Header(None)):
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    try:
        dest_dir = "/content/gemmaedge"
        os.makedirs(dest_dir, exist_ok=True)
        zip_path = os.path.join(dest_dir, "repo.zip")

        # Write upload file
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        # Unzip
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)

        os.remove(zip_path)
        return {"status": "success", "message": f"Extracted to {dest_dir}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 2. Download cloudflared if not present using pure Python
if not os.path.exists("cloudflared"):
    print("Downloading cloudflared...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    urllib.request.urlretrieve(url, "cloudflared")
    os.chmod("cloudflared", 0o755)

# Start Cloudflare Tunnel
print("Starting Cloudflare Tunnel...")
with open("tunnel.log", "w") as log_file:
    subprocess.Popen(
        ["./cloudflared", "tunnel", "--url", "http://localhost:8000"],
        stdout=log_file,
        stderr=log_file,
    )

# Extract tunnel URL from log file
time.sleep(5)  # Give the tunnel 5 seconds to establish connection
if os.path.exists("tunnel.log"):
    with open("tunnel.log", "r") as f:
        logs = f.read()
        for line in logs.split("\n"):
            if ".trycloudflare.com" in line:
                url = [w for w in line.split() if ".trycloudflare.com" in w][0]
                print(f"\n🚀 SERVER IS LIVE! Copy this URL for your local client:\n{url}\n")
                break

# 3. Start the API Server in a background thread to prevent event loop blockages
def run_api_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

server_thread = threading.Thread(target=run_api_server, daemon=True)
server_thread.start()
print("Uvicorn API server started in background thread.")