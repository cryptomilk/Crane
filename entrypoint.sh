#!/bin/sh
# SPDX-License-Identifier: MIT
# Entrypoint for the ROCm crane-serve image (see CONTAINER.md).
#
# The model path depends on $QUANT, which is only known at container
# *build* time (via the QUANT build-arg baked into this image's
# environment) -- Containerfile CMD is a static string and can't
# interpolate it, so the path is built here instead. Extra arguments
# passed to `podman run` are appended after the defaults, so they can add
# flags (e.g. --max-concurrent 8) or override one by repeating it (e.g.
# --model-path /a/different/model.gguf).
set -eu

exec crane-serve \
    --model-path "/models/llm/Qwen3.5-9B-GGUF/Qwen3.5-9B-${QUANT}.gguf" \
    --model-type qwen3.5 \
    --format gguf \
    "$@"
