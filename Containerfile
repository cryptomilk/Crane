# Runs crane-serve with Qwen 3.5 on an AMD ROCm GPU (see CONTAINER.md for
# the research/decisions behind every choice below).
#
# Build:
#   podman build -t crane-serve-rocm .
#   podman build -t crane-serve-rocm --build-arg QUANT=Q6_K .   # different quant
#
# Run (needs the host's ROCm device nodes and, on rootless Podman, group
# access to them -- adjust --group-add for your host's render group if
# `keep-groups` isn't enough):
#   podman run --rm -it \
#     --device /dev/kfd --device /dev/dri --group-add keep-groups \
#     -p 8080:8080 \
#     -v crane-rocm-kernel-cache:/cache/candle-rocm \
#     crane-serve-rocm

# ---------------------------------------------------------------------
# Builder: compiles crane-serve against the pinned candle ROCm fork.
# ---------------------------------------------------------------------
FROM rocm/dev-ubuntu-24.04:7.14.0-full AS builder

ENV ROCM_PATH=/opt/rocm
ENV DEBIAN_FRONTEND=noninteractive

# libssl-dev/pkg-config: reqwest/hf-hub's default-tls (native-tls) backend
# links system OpenSSL and needs headers to build openssl-sys from source
# -- unrelated to ROCm, needed for any crane-serve build.
# libclang-dev: rocm-rs's build.rs runs bindgen over the HIP headers and
# needs libclang.so at *build* time (distinct from the clang inside
# /opt/rocm/llvm that hipcc uses).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libssl-dev \
        pkg-config \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace
COPY . .

RUN cargo build --release -p crane-serve --features rocm

# ---------------------------------------------------------------------
# Runtime: same ROCm base as the builder, deliberately not a slimmer
# runtime-only image. candle's rocm feature JIT-compiles kernels via
# hipcc/clang-offload-bundler on first use, at *runtime*, not just at
# build time -- a runtime-libs-only image would be missing the compiler
# the first inference request needs.
# ---------------------------------------------------------------------
FROM rocm/dev-ubuntu-24.04:7.14.0-full

# Everything down to the model download is deliberately the *first* thing
# in this stage and touches nothing else: Buildah/Podman layer caching
# invalidates a layer and everything after it once any earlier
# instruction's text changes, so the multi-GB download must stay ahead of
# every other instruction in this stage, including ones that look
# unrelated (e.g. an apt package tweak below) -- not just ahead of
# `COPY --from=builder`. Learned the hard way: an earlier revision put
# the LD_LIBRARY_PATH fix above this block, and editing that one line
# alone forced a full model redownload on the next build.
#
# uv: runs this repo's own data/crane-model-download tool (PEP 723 inline
# script metadata) without separately managing a Python venv or installing
# huggingface_hub by hand.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Bake the model into the image so the container is self-contained -- no
# separate model volume required at run time. QUANT overridable at build
# time; the tool's own default is Q4_K_M, but Q5_K_M/Q6_K are recommended
# here instead given how much headroom a 9B model leaves on a 16GB card
# (see the "Which model/quant fits the RX 9070 XT" section of
# CONTAINER.md).
#
# HF_TOKEN is optional -- Qwen3.5-9B-GGUF is a public, ungated repo, but a
# token avoids anonymous-download rate limits. Not passed as --token: the
# script's own --help recommends the HF_TOKEN env var instead (avoids the
# value showing up via shell history/`ps`), and huggingface_hub already
# reads it from the environment automatically. Note it still ends up
# baked into this image's metadata (visible via `podman history`/
# `inspect`) since ENV/ARG aren't secret storage -- fine for a
# rate-limit-avoidance token on a public repo, not for anything sensitive.
ARG QUANT=Q5_K_M
ENV QUANT=${QUANT}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
COPY data/crane-model-download /opt/crane-model-download
# `hf_hub_download` (used internally by the script) downloads into
# ~/.cache/huggingface/hub *then* copies to --path, so without removing
# the cache in this same RUN/layer the image would carry two copies of
# the multi-GB model file.
RUN uv run /opt/crane-model-download --model qwen3.5-9b-gguf --path /models --quant "${QUANT}" \
    && rm -rf /root/.cache/huggingface

# Everything below is free to change without ever invalidating the
# download above, since cache invalidation only cascades forward.
ENV ROCM_PATH=/opt/rocm
ENV DEBIAN_FRONTEND=noninteractive
# rocm-rs's build.rs links crane-serve against /opt/rocm/lib's libraries
# (libamdhip64.so, librocblas.so, ...) via an explicit `-L` search path at
# *build* time, but that doesn't add an rpath to the binary or register
# the path with the dynamic linker for *run* time -- without this, the
# binary fails at startup with e.g. "libamdhip64.so.7: cannot open shared
# object file", even inside this same ROCm image.
ENV LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/target/release/crane-serve /usr/local/bin/crane-serve
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Kernels JIT-compile on first use and are cached on disk (see
# CONTAINER.md); mount a volume here so a container restart doesn't pay
# the compile cost again.
ENV CANDLE_ROCM_CACHE_DIR=/cache/candle-rocm
VOLUME ["/cache/candle-rocm"]

# crane-serve's own CLI defaults are already --host 0.0.0.0 --port 8080,
# i.e. reachable from outside the container once the port is published.
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
