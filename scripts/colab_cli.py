import sys
import os
import requests
import io
import zipfile

# read server url from env file
SERVER_URL = ""
AUTH_TOKEN = ""

if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("COLAB_NOTEBOOK_URL="):
                SERVER_URL = line.split("=")[1].strip()
            elif line.startswith("AUTH_TOKEN="):
                AUTH_TOKEN = line.split("=")[1].strip()

if not SERVER_URL:
    print("Error: COLAB_NOTEBOOK_URL is not set in your .env file.")
    print("Please create a .env file containing: COLAB_NOTEBOOK_URL=https://your-tunnel-url.trycloudflare.com")
    sys.exit(1)

def send_command(command: str, endpoint: str):
    try:
        headers = {}
        if AUTH_TOKEN:
            headers["X-Auth-Token"] = AUTH_TOKEN
        response = requests.post(endpoint, json={"command": command}, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"\n[Client Error] Server status: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"\n[Client Error] Connection failed: {e}")
        return None

def sync_repository():
    print("Syncing local repository with remote Colab...")
    zip_buffer = io.BytesIO()
    include_dirs = ["include", "src", "tests", "scripts"]
    include_files = ["CMakeLists.txt", "CMakePresets.json", "README.md"]
    
    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for f in include_files:
                if os.path.exists(f):
                    zipf.write(f, f)
            for d in include_dirs:
                if os.path.exists(d):
                    for root, dirs, files in os.walk(d):
                        for file in files:
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, ".")
                            zipf.write(full_path, rel_path)
        
        zip_buffer.seek(0)
        upload_url = f"{SERVER_URL.rstrip('/')}/upload"
        files = {"file": ("repo.zip", zip_buffer, "application/zip")}
        headers = {}
        if AUTH_TOKEN:
            headers["X-Auth-Token"] = AUTH_TOKEN
        response = requests.post(upload_url, files=files, headers=headers)
        if response.status_code == 200:
            res = response.json()
            if res.get("status") == "success":
                print("Sync successful! Repository updated on Colab.")
                return True
            else:
                print(f"Sync failed: {res.get('message')}")
        else:
            print(f"Sync failed with server status: {response.status_code}")
    except Exception as e:
        print(f"Connection failed during sync: {e}")
    return False

def start_interactive_shell():
    endpoint = f"{SERVER_URL.rstrip('/')}/exec"
    current_cwd = "/content"  # Default fallback UI display

    print("=== Connected to Google Colab Interactive Shell ===")
    print("Special Commands:")
    print("  sync   - Sync local repository files to Colab")
    print("  exit   - Close shell session")
    print("  clear  - Clear terminal screen\n")

    while True:
        try:
            # Recreate a native bash prompt format: [colab@machine:current_dir]$
            prompt = f"\033[1;32mcolab-env\033[0m:\033[1;34m{current_cwd}\033[0m$ "
            command = input(prompt).strip()

            if not command:
                continue
            if command.lower() in ["exit", "quit"]:
                print("Closing session.")
                break
            if command.lower() == "clear":
                print("\033[H\033[2J", end="")  # Clear terminal ANSI escape
                continue
            if command.lower() == "sync":
                sync_repository()
                continue

            # Send payload
            data = send_command(command, endpoint)

            if data:
                if data["stdout"]:
                    print(data["stdout"], end="")
                if data["stderr"]:
                    print(data["stderr"], file=sys.stderr, end="")
                # Update client prompt dynamically based on server directory
                current_cwd = data.get("cwd", current_cwd)

        except (KeyboardInterrupt, EOFError):
            print("\nClosing session.")
            break


if __name__ == "__main__":
    endpoint = f"{SERVER_URL.rstrip('/')}/exec"

    # If args are passed, behave like a traditional quick CLI tool
    if len(sys.argv) > 1:
        full_command = " ".join(sys.argv[1:])
        if full_command.lower() == "sync":
            sync_repository()
            sys.exit(0)
        data = send_command(full_command, endpoint)
        if data:
            print(data["stdout"], end="")
            print(data["stderr"], file=sys.stderr, end="")
            sys.exit(data["exit_code"])
        sys.exit(1)

    # If no args are passed, start the live active shell loop
    else:
        start_interactive_shell()