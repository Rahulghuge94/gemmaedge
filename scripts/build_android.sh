#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  echo "ANDROID_NDK_HOME must point to an Android NDK" >&2
  exit 1
fi

cmake --preset android-arm64
cmake --build --preset android-arm64 --parallel

echo "Built:"
echo "  build/android-arm64/gemmaedge"
echo "  build/android-arm64/libgemmaedge_jni.so"

