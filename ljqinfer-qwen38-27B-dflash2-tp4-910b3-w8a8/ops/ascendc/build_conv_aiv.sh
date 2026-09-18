#!/bin/bash
# Build only; never replace a library used by a running process.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CANN=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.1}
CMAKE=${CMAKE_BIN:-/tmp/cmake_portable/cmake/data/bin/cmake}
if [ ! -x "$CMAKE" ]; then CMAKE=cmake; fi
BUILD=${1:?Specify a NEW absolute build directory}
case "$BUILD" in /*) ;; *) echo "Absolute build path required" >&2; exit 2;; esac
if [ -e "$BUILD" ]; then echo "Refusing existing build path: $BUILD" >&2; exit 2; fi
"$CMAKE" -S "$ROOT/ops/ascendc/conv_aiv" -B "$BUILD" -DASCEND_CANN_PACKAGE_PATH="$CANN"
"$CMAKE" --build "$BUILD" -j2
SO="$BUILD/lib/libljq_conv_probe.so"
test -s "$SO"
sha256sum "$SO"
echo "Build only; validate before installing as ops/native/ljq_gdn_conv.so: $SO"
