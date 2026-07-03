#!/usr/bin/env bash
set -euo pipefail

cmake --preset linux-release
cmake --build --preset linux-release --parallel
ctest --preset linux-release

