# Google Colab workflow

Colab is the first Linux validation target. Use a CPU or GPU runtime; the
current storage and correctness tests require neither CUDA nor model weights.

## Build and test

```bash
!apt-get update -qq
!apt-get install -y -qq ninja-build cmake
!git clone YOUR_GEMMAEDGE_REPOSITORY_URL /content/gemmaedge
%cd /content/gemmaedge
!bash scripts/build_linux.sh
```

Inspect the text and vision GGUF headers without loading their tensor payloads:

```bash
!/content/gemmaedge/build/linux-release/gemmaedge inspect \
  /content/models/gemma-4-26B-A4B-it-Q4_K_M.gguf

!/content/gemmaedge/build/linux-release/gemmaedge inspect \
  /content/models/mmproj-gemma-4-26B-A4B-it-BF16.gguf
```

For Google Drive models:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Do not copy a 15 GB GGUF into Colab RAM. Keep it on persistent disk and pass
its path to GemmaEdge. The runtime memory-maps files and faults in requested
regions.

## Android cross-build from Colab

Download/unpack an Android NDK, set `ANDROID_NDK_HOME`, then run:

```bash
!export ANDROID_NDK_HOME=/content/android-ndk && \
  bash scripts/build_android.sh
```

The output targets 64-bit ARM and Android API 28, which is compatible with the
Pixel 8a.

