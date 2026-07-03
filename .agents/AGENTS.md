# GemmaEdge Workspace Rules

## Remote Linux Validation

When validating changes on the Linux target (e.g. compiling with CMake/Ninja, running tests, or executing shell commands on Linux):

1. **Check for Colab CLI availability**: Check if `.env` exists in the workspace root and contains a `COLAB_NOTEBOOK_URL` variable.
2. **Synchronize Repository (Local CLI Sync)**: If the Colab CLI is available, push any local modifications to the remote Colab workspace first by running:
   ```powershell
   python scripts/colab_cli.py sync
   ```
3. **Synchronize Repository (Git Method)**: Alternatively, to clone/pull directly from the upstream repository `https://github.com/Rahulghuge94/gemmaedge.git` on Colab:
   - Always ask the user to commit and push local modifications to the upstream repository first.
   - Run the git pull/clone command on the Colab remote VM to bring it up to date.
4. **Execute Remote Commands**: Instead of running local commands, run your Linux builds, compilations, or tests on the remote Colab instance using the wrapper CLI:
   ```powershell
   python scripts/colab_cli.py "<linux_command>"
   ```
   *Example*:
   ```powershell
   python scripts/colab_cli.py "cd /content/gemmaedge && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/gemmaedge_tests"
   ```
