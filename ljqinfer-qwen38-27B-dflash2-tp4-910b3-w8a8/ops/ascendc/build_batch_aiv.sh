#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CANN=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.1}
CMAKE=${CMAKE_BIN:-/tmp/cmake_portable/cmake/data/bin/cmake}
if [ ! -x "$CMAKE" ]; then CMAKE=cmake; fi
BUILD=${1:-/tmp/ljq_batch_aiv_build}
rm -rf "$BUILD"
"$CMAKE" -S "$ROOT/ops/ascendc/aiv" -B "$BUILD" -DASCEND_CANN_PACKAGE_PATH="$CANN"
"$CMAKE" --build "$BUILD" -j2
install -m 0755 "$BUILD/lib/libljq_gdn_recurrent_baseptr_aiv.so" "$ROOT/ops/native/ljq_gdn_recurrent_baseptr_aiv.so"
install -m 0755 "$BUILD/lib/libljq_qk_rms_norm_rope_aiv.so" "$ROOT/ops/native/ljq_qk_rms_norm_rope_aiv.so"
install -m 0755 "$BUILD/lib/libljq_gdn_l2_pair_aiv.so" "$ROOT/ops/native/ljq_gdn_l2_pair_aiv.so"
sha256sum "$ROOT"/ops/native/ljq_{gdn_recurrent_baseptr,qk_rms_norm_rope,gdn_l2_pair}_aiv.so
