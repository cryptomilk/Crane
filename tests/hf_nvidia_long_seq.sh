#!/bin/bash
# Builds crane-core with CUDA and runs the long-sequence flash-attention
# regression tests (`long_seq_*` in `fattn_long_seq_tests.rs`) on an NVIDIA
# GPU, as a Hugging Face Job. The tests fail by design while the BF16
# kernel bug is open, so there is no `set -e` and every step still runs.
#
# Steps: the full sweep for both dtypes, the BF16 test with
# `try_causal_prefill` forced onto the BF16->F16 cast path, then the same
# BF16 test under `compute-sanitizer` (racecheck, synccheck, initcheck,
# memcheck) at two lengths.
#
# Usage (from the repo root, with `hf` logged in and `hf_xet` installed):
#
#   rm -rf /tmp/crane-src && mkdir /tmp/crane-src
#   git archive HEAD | tar -x -C /tmp/crane-src
#   hf jobs run --detach --flavor l4x1 --timeout 4h -v /tmp/crane-src:/src \
#     pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash /src/tests/hf_nvidia_long_seq.sh
#   hf jobs logs <job id> > long_seq_nvidia.log
#
# The job exits when the script ends, which stops billing. Cancel it with
# `hf jobs cancel <job id>` to stop earlier.
set -ux

nvidia-smi; nvcc --version; which compute-sanitizer
#apt-get update && apt-get install -y build-essential cmake pkg-config \
#    libssl-dev git curl protobuf-compiler
#curl https://sh.rustup.rs -sSf | sh -s -- -y
## shellcheck source=/dev/null
#. ~/.cargo/env
#cp -r /src /work && cd /work || exit 1

cargo test -p crane-core --release --features "cuda,onnx" --lib --no-run 2>&1 | tail -3
BIN=$(find target/release/deps -maxdepth 1 -name 'crane_core-*' -executable -type f \
      -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)
echo "BIN=$BIN"
T="--ignored --nocapture --test-threads=1"

echo "=== 1. full sweep"
$BIN long_seq_bf16 $T
$BIN long_seq_f16 $T

echo "=== 2. bf16, try_causal_prefill forced onto the BF16->F16 cast path"
CRANE_DISABLE_BF16_FLASH_ATTN=1 $BIN long_seq_bf16 $T

# One seed per distribution is enough for the sanitizers: they report
# hazards from the access pattern, whether or not the output is wrong.
export CRANE_LONG_SEQ_SEEDS=1
for seq in 512 1024; do
  for tool in racecheck synccheck initcheck memcheck; do
    echo "=== 3. compute-sanitizer $tool, seq=$seq"
    CRANE_LONG_SEQ_SEQS=$seq timeout 20m compute-sanitizer --tool $tool --print-limit 30 \
      $BIN long_seq_bf16 $T
  done
done
