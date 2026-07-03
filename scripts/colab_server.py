import os
import sys
import subprocess
import urllib.request
import time
import uuid
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
AUTH_TOKEN = "asdfghjkl"# os.environ.get("AUTH_TOKEN", "gemmaedge-secret-token")
print(f"Server security token is: {AUTH_TOKEN}")

# Track persistent working directory
current_working_dir = "/content"
if not os.path.exists(current_working_dir):
    current_working_dir = os.getcwd()

# Track active tasks
tasks = {}
task_dir = "/content/tasks"
os.makedirs(task_dir, exist_ok=True)

class Task:
    def __init__(self, task_id, command, cwd):
        self.task_id = task_id
        self.command = command
        self.cwd = cwd
        self.stdout_path = os.path.join(task_dir, f"{task_id}.stdout")
        self.stderr_path = os.path.join(task_dir, f"{task_id}.stderr")
        self.stdout_file = open(self.stdout_path, "w+")
        self.stderr_file = open(self.stderr_path, "w+")
        
        marker = "__CWD_MARKER__"
        full_command = f"{command}\necho '{marker}'\npwd"
        
        self.proc = subprocess.Popen(
            full_command,
            shell=True,
            stdout=self.stdout_file,
            stderr=self.stderr_file,
            cwd=cwd,
            text=True
        )
        self.start_time = time.time()
        self.last_read_offset_stdout = 0
        self.last_read_offset_stderr = 0
        self.completed = False
        self.exit_code = None

    def check_status(self):
        if self.completed:
            return
            
        self.exit_code = self.proc.poll()
        if self.exit_code is not None:
            self.completed = True
            self.stdout_file.close()
            self.stderr_file.close()
        elif time.time() - self.start_time > 300: # 5 min timeout
            self.proc.terminate()
            self.exit_code = -9
            self.completed = True
            self.stdout_file.close()
            self.stderr_file.close()


class CodeExecutionRequest(BaseModel):
    command: str


@app.post("/exec")
async def execute_command(req: CodeExecutionRequest, x_auth_token: str = Header(None)):
    global current_working_dir
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        return {"error": "Unauthorized", "exit_code": 401}
        
    task_id = str(uuid.uuid4())
    tasks[task_id] = Task(task_id, req.command, current_working_dir)
    return {"task_id": task_id}


@app.get("/task/{task_id}")
async def get_task_status(task_id: str, x_auth_token: str = Header(None)):
    global current_working_dir
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        return {"error": "Unauthorized", "exit_code": 401}
        
    if task_id not in tasks:
        return {"error": "Task not found", "exit_code": 404}
        
    task = tasks[task_id]
    task.check_status()
    
    # Read new stdout content
    new_stdout = ""
    if os.path.exists(task.stdout_path):
        with open(task.stdout_path, "r") as f:
            f.seek(task.last_read_offset_stdout)
            new_stdout = f.read()
            task.last_read_offset_stdout = f.tell()
            
    # Read new stderr content
    new_stderr = ""
    if os.path.exists(task.stderr_path):
        with open(task.stderr_path, "r") as f:
            f.seek(task.last_read_offset_stderr)
            new_stderr = f.read()
            task.last_read_offset_stderr = f.tell()
            
    # If completed, check for CWD update
    cwd = task.cwd
    marker = "__CWD_MARKER__"
    if task.completed:
        # We need to read the full stdout to parse CWD marker
        if os.path.exists(task.stdout_path):
            with open(task.stdout_path, "r") as f:
                full_stdout = f.read()
            if marker in full_stdout:
                parts = full_stdout.split(marker)
                new_cwd = parts[1].strip()
                if os.path.exists(new_cwd):
                    current_working_dir = new_cwd
                    cwd = new_cwd
                # Strip marker and CWD from the new_stdout return
                if marker in new_stdout:
                    new_stdout = new_stdout.split(marker)[0]
                
    return {
        "completed": task.completed,
        "exit_code": task.exit_code,
        "stdout": new_stdout,
        "stderr": new_stderr,
        "cwd": cwd
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