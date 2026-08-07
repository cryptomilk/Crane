# Podman container

NEVER COMMIT THIS DOCUMENT
NEVER ADD REFERENCES TO THIS FILE IN CODE OR COMMIT MESSAGES
NEVER REFERENCE ANY UPSTREAM CODE

I have an AMD 9700XT GPU. I'm currently unable to run ROCm locally. So I would like to use a rocm container to run Crane inside with Qwen 3.5 LLM.

So we need a Containerfile which does all of that for me.

It should run crane-serve inside on a TCP port 8080 and be reachable from outside.

There are two options:

1. https://github.com/huggingface/candle/pull/3801
   Feat/rocm backend for candle

2. Or using ZLUDA

## Findings (research only, no Containerfile yet)

### Hardware note

Confirmed via `lspci`:
```
03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 48 [Radeon RX 9070/9070 XT/9070 GRE] (rev c0)
```
i.e. **Radeon RX 9070 XT** (Navi 48, **RDNA4**, `gfx1201`, 16GB GDDR6,
304W TDP). This is `gfx1201`/RDNA4, **not** RDNA3, which matters for
both options below. It shares its Navi 48 die with the workstation
**Radeon AI PRO R9700** (32GB variant of the same silicon), and there are
open reports against that sibling card (e.g. `ollama/ollama#14686`,
March 2026) of the ROCm backend failing to *initialize* even with the
official HIP SDK 7.1 installed.

**Update: base ROCm is confirmed working on this exact card** — you
trained a Kokoro German voice inside a `rocm/pytorch` container, which
means the HIP driver, rocBLAS and MIOpen all function correctly on
`gfx1201` here. The `ollama` report above is evidently not universal (or
was specific to a different driver/kernel/ROCm version combination). This
removes the "does ROCm even initialize on this GPU" risk for both options
below — the remaining risk for each is specific to *candle*/*Crane*
running on top of that already-working ROCm stack, not the stack itself.

### Current state of Crane itself

- Zero references to ROCm/HIP/AMD anywhere in the repo (code, `Cargo.toml`,
  docs) — confirmed by grep.
- `cuda` is a first-class feature threaded through `crane-core`, `crane`,
  and `crane-serve`'s `Cargo.toml`s, enabling `candle-core/cuda`,
  `candle-nn/cuda`, `candle-transformers/cuda` and pulling in
  `bindgen_cuda`/`nvml-wrapper`.
- `crane-core/build.rs` uses `bindgen_cuda` (a wrapper around `nvcc`) to
  compile Crane's **own** custom kernels (`kernels/cuda/**/*.cu` — fused
  top-k, argmax, SiLU-mul, Snake, Atan2, GDN) to PTX, separately from
  candle's own kernel crate.
- Device selection lives in `crane-serve/src/lib.rs` (~line 286-304):
  `--cpu` flag forces CPU, else `cuda_if_available(0)` under the `cuda`
  feature, else `Device::new_metal` on macOS, else CPU. About 25 files
  across `crane-core`/`crane`/`crane-serve` gate logic on
  `#[cfg(feature = "cuda")]` or pattern-match `Device::Cuda(_)`.

### Option 1: candle PR #3801 ("Feat/rocm backend") — chosen direction

- **Open**, unmerged: `state=OPEN`, `mergeable=CONFLICTING`,
  `mergeStateStatus=DIRTY` against current `huggingface/candle` main;
  128 files changed, +18400/-617, from a personal fork
  (`xmiksay/candle`, branch `feat/rocm-backend`). The fork's root
  `Cargo.toml` still declares `workspace.package.version = "0.11.0"` —
  the same version Crane already pins (`candle-core = "0.11"`), so this
  is version-compatible, just needs to be sourced via git (branch/commit)
  instead of crates.io.
- Adds a `rocm` feature (default path: im2col + rocBLAS GEMM) and an
  opt-in `miopen` feature to `candle-core`/`candle-nn`/`candle-transformers`,
  mirroring how `cudnn` layers over `cuda`. Pulls in `rocm-rs` 0.5.x
  (`default-features = false`, `features = ["macros"]` — its default
  `gpu-sort` feature needs a nightly `-Zbuild-std` invocation candle
  never calls into, so it's deliberately left off).
- **Kernels compile at runtime, not build time** — this is the biggest
  structural difference from the CUDA path and matters for container
  design. `candle-rocm-kernels` embeds the same `candle-kernels/src/*.cu`
  sources the CUDA backend uses, but instead of precompiling them (like
  `bindgen_cuda`/`nvcc` do for CUDA, producing embedded PTX at build
  time), it shells out to `hipcc --genco` + `clang-offload-bundler`
  **on first use of each kernel module, at runtime**, and caches the
  resulting code objects on disk (default `~/.cache/candle-rocm/`,
  overridable via `CANDLE_ROCM_CACHE_DIR`; falls back to a per-uid temp
  dir if `~/.cache` is missing/read-only, e.g. in a container running as
  a service account). Their own README states the requirement plainly:
  *"ROCm 6.2 or newer, with `hipcc` and `clang-offload-bundler` available
  at runtime."* **This means the final container image needs the full
  ROCm dev/compiler toolchain (`hipcc`, `clang-offload-bundler` from
  `$ROCM_PATH`, default `/opt/rocm`) present at runtime, not just
  runtime shared libraries** — ruling out a slim multi-stage build that
  only ships `librocblas.so`-style runtime packages the way a CUDA image
  might ship only `libcublas.so` after an nvcc-at-build-time compile.
  Expect ~3-5s compile per kernel module on first use (`quantized.cu`,
  at 4,845 lines, takes ~70s), one-time per cache directory.
- **Correction from an earlier pass of this doc**: the quantized-matmul
  tile-geometry dispatch is **not** Ampere-default for our card. Per
  `candle-rocm-kernels/README.md`, the compile flag is `-DRDNA2` for
  `gfx103x` and **`-DRDNA3` for `gfx11xx/gfx12xx`** — i.e. our `gfx1201`
  card *does* get the RDNA-tuned tile geometry (`nwarps = 8`), not the
  Ampere fallback. What's genuinely unvalidated for `gfx1201` is
  **performance tuning only**: the MMQ-vs-dense crossover thresholds and
  the transpose-then-GEMM reorientation heuristic were measured
  exclusively on `gfx1101` (RDNA3, RX 7800 XT) and the README calls them
  "specific to it" — so the same code path should run correctly on our
  card, just not necessarily at re-tuned-optimal thresholds.
- **A downstream crate (i.e. Crane) can compile its own `.cu` kernels
  through the same runtime pipeline without a build.rs/HIP toolchain of
  its own.** `candle_core::rocm_backend::RocmDevice::get_or_load_custom_func(name, module, source)`
  is the public, documented counterpart to `CudaDevice::get_or_load_custom_func`
  — it takes raw `.cu` source text (e.g. via `include_str!`), compiles it
  through the same shim/cache/hipcc pipeline as candle's own kernels, and
  is namespaced apart from the built-ins. This substantially **de-risks**
  the earlier estimate: Crane's own fused ops
  (`crane-core/kernels/cuda/**/*.cu` — top-k, argmax, SiLU-mul, Snake,
  Atan2, GDN) would *not* need a second `hipcc`-based `build.rs` pipeline
  written from scratch; a `rocm_fwd` impl could pass the existing
  `.cu` source straight into this runtime API instead of the CUDA path's
  build-time-PTX-then-`cudarc`-load pattern.
- **MoE gap is irrelevant to our target model.** The PR's dense
  (non-quantized) fused MoE GEMM (`moe_gemm`, and the prefill half of
  `moe_gemm_gguf`) is explicitly **not implemented** — it needs an
  `nvcuda::wmma`-based host library ROCm has no build step for. Quantized
  MoE (`moe_gemm_gguf`'s decode path, `FusedMoeGGUF`) *is* implemented.
  This would matter for a quantized MoE model, but **Crane's
  `crane-core/src/models/qwen3_5/` has no MoE/expert code at all**
  (confirmed by grep) — Qwen 3.5 in Crane is a dense transformer with GDN
  hybrid layers, so this gap doesn't block the target model.
- Remaining op coverage directly relevant to running an LLM: `matmul` via
  rocBLAS (`f32`/`f64`/`f16`/`bf16`, strided-batched), and `candle-nn`'s
  `softmax_last_dim`/`rms_norm`/`layer_norm`/`sigmoid`/`rope`/`rope_i`/`rope_thd`
  all have `rocm_fwd` shared-kernel implementations (f16/bf16/f32/f64).
  Quantized GGUF matmul (MMVQ/DMMV/MMQ/dequantize+GEMM) is fully
  implemented, so a quantized (GGUF) Qwen 3.5 would also work, not just
  full-precision safetensors.
- To actually use it in Crane, this is not just a container change, it's
  new engineering on top of an experimental, unmerged branch (though the
  `gfx1201` code-path risk is now understood to be about performance
  tuning, not correctness):
  - Point the workspace's `candle-core`/`candle-nn`/`candle-transformers`
    deps at this fork/branch (git dependency or `[patch]`) instead of
    crates.io `"0.11"`, and carry the unresolved merge conflicts against
    current candle main ourselves.
  - Add a brand-new `rocm` feature to `crane-core`/`crane`/`crane-serve`'s
    `Cargo.toml`s (doesn't exist today) mirroring the `cuda` wiring.
  - Add ROCm `rocm_fwd` impls for Crane's own custom ops
    (`crane-core/src/ops/fused_ops/`, `ops/gdn/`) using
    `RocmDevice::get_or_load_custom_func` at runtime against the existing
    `.cu` sources (see above) — no new build-time HIP toolchain needed in
    `crane-core/build.rs` itself for this part.
  - Add `Device::new_rocm`/`rocm_if_available` branches in
    `crane-serve/src/lib.rs`'s device selection and thread
    `#[cfg(feature = "rocm")]` through the ~25 existing
    `#[cfg(feature = "cuda")]`/`Device::Cuda(_)` call sites (engine memory
    queries, sampling GPU-native path, SDK client device selection, etc).
  - Ensure the **runtime** container image (not just a build stage) ships
    `hipcc`, `clang-offload-bundler`, and the ROCm runtime libraries
    (`rocBLAS`, `rocRAND`, and `MIOpen` if `--features miopen` is used) —
    the project's own CI (`.github/workflows/ci_rocm.yaml`) only installs
    `-dev` header packages (`hip-dev`, `rocblas-dev`, `rocsolver-dev`,
    `rocfft-dev`, `rocsparse-dev`, `rocrand-dev`, `miopen-hip-dev` at
    ROCm 7.2.4 from `repo.radeon.com`) for a **GPU-free `cargo check`/`clippy`
    job**, which is not sufficient for an image that must also run and
    JIT-compile kernels.
  - Useful runtime env vars once running: `CANDLE_ROCM_ARCH` (defaults to
    autodetected `gfx1201` via `hipGetDeviceProperties`, override only if
    needed), `CANDLE_ROCM_CACHE_DIR` (worth pointing at a persistent
    volume so the container doesn't re-JIT every kernel on each restart),
    `CANDLE_ROCM_FORCE_DMMV`, `ROCM_PATH` (default `/opt/rocm`).

### Option 2: ZLUDA (CUDA-on-AMD translation layer)

- `vosen/ZLUDA` is active again (pushed within the last day as of this
  writing), 14.7k stars, dual Apache-2.0/MIT. Historically dropped AMD
  support in 2024 for legal reasons, now back with AMD as the primary
  target.
- Upstream's own quick-start doc leads with: *"This version of ZLUDA is
  under heavy development and will likely not work with your application
  yet."* Their FAQ lists PyTorch support (a far larger, better-funded
  effort than Crane) as still only their "top priority", targeted for
  Q4 2025 — i.e. general ML-framework compatibility is bleeding edge, not
  a solved problem, as of their own docs.
- Officially supports "Radeon RX 5000 series and newer" (RDNA1+), so
  `gfx1201`/RDNA4 is nominally in scope, but there is no GPU-specific
  validation list and no mention of RDNA4 anywhere in their docs.
- Mechanism: `LD_LIBRARY_PATH`/`LD_AUDIT`-inject ZLUDA's `libcuda.so` (and
  `libnvidia-ml.so`, `libcublas.so`, etc.) ahead of the real ones so an
  **unmodified** CUDA binary's driver-API calls get intercepted and
  JIT-translated: PTX → AMDGPU ISA, cuBLAS/cuBLASLt calls → rocBLAS/hipBLASLt,
  cuDNN → MIOpen.
- Practically, this would mean: build `crane-serve --features cuda`
  normally (needs `nvcc`/CUDA toolkit at build time only — no physical
  NVIDIA GPU required to compile PTX), then at runtime run it under ZLUDA
  inside a ROCm container. **No Crane or candle source changes needed if
  it works** — this is the cheap option to *try*.
- The risk is entirely at runtime and binary (works, or silently
  misbehaves/crashes): candle's CUDA backend drives raw driver calls via
  `cudarc`, cuBLAS/cuBLASLt for matmul, and custom PTX kernels for
  rope/rmsnorm/etc.; Crane adds its *own* custom PTX kernels on top
  (fused top-k, argmax, SiLU-mul, Snake, Atan2, GDN). None of this is on
  ZLUDA's documented/validated list — only `llama.cpp` is documented, and
  even that doc recommends compiling for one specific NVIDIA arch (e.g.
  `86`) to keep cuBLAS in the loop, and is written from a Windows-first
  perspective (the HIP SDK install steps target Windows; Linux usage is
  comparatively thin in their docs, though it should map onto an
  already-installed ROCm container).

### Decision: Option 1 (candle PR #3801)

Chosen direction: build Crane against the `xmiksay/candle` `feat/rocm-backend`
branch's `rocm` feature rather than trying ZLUDA. Base ROCm-on-`gfx1201` is
already confirmed working (Kokoro training via `rocm/pytorch`), and the
deeper look above shows `gfx1201` gets the same RDNA-tuned kernel tile
geometry as the tested `gfx1101` card (only perf-tuning thresholds are
unvalidated for this exact GPU, not correctness), and that Crane's own
custom kernels can reuse candle's runtime JIT pipeline
(`RocmDevice::get_or_load_custom_func`) instead of needing a from-scratch
HIP build step.

### Implementation plan (not started — planning only, no code/Containerfile changes made)

**Environment constraint that shapes the ordering below**: this working
sandbox has **no Rust toolchain and no ROCm installed at all**
(`cargo`/`rustc` not found, no `/opt/rocm`, no `hipcc`). Every build/check
step therefore has to happen either on your host machine or inside a ROCm
container *you* run — nothing here can compile-check candle's `rocm`
feature (which needs ROCm headers on disk for `rocm-rs`'s bindgen even for
a plain `cargo check`). Per this repo's own rule, `crane-serve` also must
never be launched by the agent, even to test — you run it and share
logs/output.

**Commit policy**: one commit per step below, and each commit must stand on
its own (builds cleanly, no half-finished code) per this repo's commit
standards — not one giant commit at the end. Since this sandbox can't
compile anything ROCm-related, the loop for each patch step is: make the
change here → you build-check it on your machine/ROCm container → report
back pass/fail → commit once it's confirmed clean → move to the next step.
That means steps 1-4 and 7 each pause for your build confirmation before
being committed; nothing gets committed unverified.

Ordered steps:

1. ✅ **Done (`fd56051`). Pin the candle fork as a git dependency.** Point
   the workspace's `candle-core`/`candle-nn`/`candle-transformers` deps at
   a specific **commit SHA** of `xmiksay/candle`'s `feat/rocm-backend`
   branch (not a floating branch ref — it's actively force-pushed per its
   commit history, so a branch name alone isn't reproducible). Verified
   with `cargo check --workspace` (default features) before committing.
2. ✅ **Done (`007f1f6`). Add the `rocm` feature** to
   `crane-core`/`crane`/`crane-serve`'s `Cargo.toml`s, mirroring the
   existing `cuda` feature wiring (`candle-core/rocm`, `candle-nn/rocm`,
   `candle-transformers/rocm`). Verified with `cargo check -p crane-core
   -p crane-serve --features rocm` inside a
   `rocm/dev-ubuntu-24.04:7.14.0-full` container (needed `libssl-dev`,
   `pkg-config` and `libclang-dev` installed first — see below).
3. ⏭️ **Deferred, not required.** Originally planned as "implement
   `rocm_fwd` for Crane's own custom ops (`fused_ops/*`, `ops/gdn/`) via
   `RocmDevice::get_or_load_custom_func`" — investigated the actual call
   sites before writing any of it and found none of it is load-bearing for
   Qwen 3.5:
   - `fused_silu_mul`/`snake`/`atan2`/`gpu_argmax`/`topk_indices` are used
     by `hunyuan_dense`, the ONNX evaluator, and `models/modules/ffn.rs` —
     **not** by Qwen 3.5, whose own `Mlp` (`qwen3_5/modeling.rs:515`) calls
     plain `candle_nn::ops::silu()` directly.
   - `crane-serve/src/engine/sampling.rs`'s GPU-fast-paths (lines 166, 200)
     already gate on `logits.device().is_cuda()`, which is `false` on
     ROCm, so they already fall through to the portable sampling path —
     no change needed, it was written to degrade gracefully for any
     non-CUDA device already.
   - The GDN recurrence's fused CUDA kernel
     (`crane-core/src/ops/gdn/backend.rs:174-177`) is behind
     `#[cfg(feature = "cuda")] if q.device().is_cuda()`; under
     `--features rocm` that whole branch compiles out, unconditionally
     falling back to `gated_delta_rule_recurrence`, the portable
     per-timestep implementation written in plain Candle tensor ops —
     already device-generic, already runs on ROCm.

   Writing a hand-rolled, unverified HIP kernel launch (shared-memory
   tuning, register-specialized variants) that can't be run on real
   hardware from this sandbox would be pure performance work carrying
   real correctness risk, for a path that already works today via the
   portable fallback. **Decision: skip for now; revisit only as a
   profiled performance optimization after step 8 confirms the portable
   path works end-to-end.**
4. ✅ **Done (`cad2a95`). Extend device selection** in
   `crane-serve/src/lib.rs`: `Device::new_rocm(0)` (falling back to CPU on
   error, no `rocm_if_available` convenience constructor exists) when
   built with `--features rocm`, and `resolve_dtype` treats ROCm like CUDA
   (BF16 default). Verified with `cargo check -p crane-serve --features
   rocm`.

   Scoped narrowly after tracing actual call sites — **not** touched,
   left as follow-ups:
   - The TTS/ASR/VLM branches' separate `use_cpu` logic (forces CPU
     whenever `cuda` isn't compiled in — pre-existing, affects Metal too).
     Plain Qwen 3.5 isn't `is_vlm()`/`is_tts()`/`is_asr()`, so it's
     unaffected — it goes through the final `else` branch that uses the
     top-level `device`/`dtype` directly, which this step already fixes.
   - `engine/mod.rs`'s `query_total_gpu_memory`/`query_gpu_memory_usage`
     (used by `--gpu-memory-limit` and the stats endpoint) already
     degrade gracefully to `0`/unlimited for any non-CUDA device — no
     hard failure, just no ROCm memory figures yet.
   - `crane` SDK client device selection (`crane/src/llm/client.rs` and
     friends) and the `example` binaries — out of scope for the
     `crane-serve` server story.
5. *(folded into steps 1-4 above, not a separate commit)* — compile-check
   on a machine with ROCm headers, e.g. `cargo check -p crane-core -p
   crane-serve --features rocm`. Cannot be done in this sandbox.
6. *(folded into steps 1-4 above, not a separate commit)* — iterate on
   whatever a given step's build-check surfaces, before that step's
   commit.
7. ✅ **Done (`809a325`). Containerfile written and built.**
   `Containerfile` (repo root) — multi-stage, `rocm/dev-ubuntu-24.04:7.14.0-full` for both
   stages (runtime stage can't be slimmer per the JIT-compilation finding
   above). Builder stage compiles `crane-serve --features rocm --release`;
   runtime stage additionally bakes the Qwen 3.5 9B GGUF model into the
   image at build time (via `uv run data/crane-model-download`, cleaning
   `~/.cache/huggingface` in the same layer to avoid a duplicate multi-GB
   copy), so the container is self-contained rather than needing a model
   volume mount. `QUANT` is a build-arg (default `Q5_K_M`, per the sizing
   section above). `entrypoint.sh` builds the model path from `$QUANT` at
   container start (a static Dockerfile `CMD` can't reference a build-arg
   dynamically) and forwards any extra `podman run` args to `crane-serve`.

   Also added `compose.yaml` (Compose Specification format, works with
   `podman compose`/`podman-compose`) so `podman compose up -d --build`
   replaces having to remember the full `podman run` invocation (device
   passthrough, group access, port, volume). `group_add: keep-groups`
   verified against `podman-compose`'s own source — it forwards each
   `group_add` entry straight through as `--group-add <item>` with no
   transformation, so this works exactly like the plain `podman run`
   command's `--group-add keep-groups`.

   Verified with `podman build -t crane-serve-rocm .` on the target
   machine — builds cleanly.

   **Layer-ordering lesson learned during step 8 below**: the model
   download must be the *first* thing in the runtime stage, not merely
   ahead of `COPY --from=builder`. Buildah/Podman layer caching
   invalidates a layer and everything after it once any earlier
   instruction's text changes — an early revision put the
   `LD_LIBRARY_PATH` fix (added while debugging the `libamdhip64.so`
   error below) above the download block, and editing that one line
   alone forced a full multi-GB model redownload on the next build, even
   though nothing about the model or the download command had changed.
   Reordered so only the download's own inputs (`QUANT`, `HF_TOKEN`, the
   script, `uv`) precede it; everything else in the stage now comes
   after, free to change without ever invalidating it again.
8. 🚧 **In progress, blocked on an upstream bug.** You run the container /
   `crane-serve`; I read back logs/output and iterate from there — I will
   not launch it myself even inside a container. Any fixes this surfaces
   get their own follow-up commit(s), same policy as above — not folded
   silently back into an earlier step's already-committed change.

   Progress so far:
   - Fixed `libamdhip64.so.7: cannot open shared object file` at container
     startup — `rocm-rs`'s build.rs links against `/opt/rocm/lib` via an
     explicit `-L` search path at *build* time, which doesn't add an
     rpath or register the path with the runtime dynamic linker; added
     `ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64` to the runtime
     stage.
   - Along the way, learned `podman compose up --build` can rebuild the
     image without recreating the running container (stale-container
     gotcha) — `podman compose down` before `up --build` avoids it. Also
     learned the model-download layer-caching lesson recorded above.
   - With those fixed: **device selection works** —
     `Device: Rocm(RocmDevice(DeviceId(1))), dtype: BF16` in the logs —
     and the GGUF model loads (`GGUF loaded: 427 tensors, 46 metadata
     entries`).
   - **Now blocked** on a real bug in the pinned candle branch, hit while
     JIT-compiling the `quantized` kernel module on first inference:
     `candle-rocm-kernels`'s HIP compatibility shim unconditionally
     defines `atomicAdd(__half)`/`atomicAdd(__hip_bfloat16)`, assuming
     HIP itself lacks them — true against the ROCm 7.2.4 the shim was
     developed against, **not true on our ROCm 7.14.0**, whose
     `amd_hip_fp16.h`/`amd_hip_bf16.h` already define these, causing
     `error: redefinition of 'atomicAdd'`. Full writeup, root cause, and
     the exact AMD header guard involved: **`BUG_CANDLE_ROCM.md`**.
     Not yet reported upstream or patched — next decision is whether to
     file it on PR #3801 and wait, or vendor+patch
     `candle-rocm-kernels` locally via a Cargo `[patch]` entry to unblock
     ourselves now.

### Verifying steps 2-4 before the real Containerfile exists (step 7)

You can't build with ROCm headers on the host directly (no local ROCm), and
we don't have the real Containerfile yet — that's step 7, and writing it
early just to unblock earlier steps would get the ordering backwards. The
practical answer is an **ad-hoc container**, not a real Containerfile yet,
just for compile-checking steps 2-4 on the base image already decided on
above:

```bash
podman run --rm -it \
  -v "$(pwd)":/workspace:Z \
  -v crane-rocm-cargo-registry:/root/.cargo/registry \
  -v crane-rocm-cargo-git:/root/.cargo/git \
  -v crane-rocm-target:/workspace/target \
  -w /workspace \
  rocm/dev-ubuntu-24.04:7.14.0-full bash
```

Then inside the container (one-time per container image, cached via the
named volumes above so repeat checks don't redownload crates or
recompile from scratch):

```bash
apt-get update && apt-get install -y --no-install-recommends libssl-dev pkg-config libclang-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
cargo check -p crane-core -p crane-serve --features rocm
```

`libssl-dev`/`pkg-config` (the Ubuntu apt names — "openssl devel" is the
RPM-family name for the same thing) are needed because `reqwest` and/or
`hf-hub` pull in `openssl-sys` through `reqwest`'s default `default-tls`
feature (native-tls on Linux), which links against system OpenSSL and
needs its headers to build `openssl-sys` from source. This isn't ROCm- or
rocm-feature-specific — a plain `cargo build -p crane-serve` with no GPU
features at all would hit the same thing on a bare container, so **the
real Containerfile in step 7 needs `libssl-dev pkg-config` in its apt
install list regardless of the ROCm packages**.

`libclang-dev` is ROCm-specific: `rocm-rs`'s `build.rs` runs `bindgen` over
the HIP headers to generate its Rust bindings, and `bindgen` needs
`libclang.so` at runtime (distinct from having the `clang` compiler binary
itself — `rocm/dev-ubuntu-24.04` ships enough of LLVM/clang for `hipcc` to
work, but not necessarily `libclang-dev`). Without it: `thread 'main'
panicked ... Unable to find libclang`. **The real Containerfile's build
stage needs `libclang-dev` too**, alongside `libssl-dev`/`pkg-config`.

No `--device /dev/kfd --device /dev/dri` needed for this — `cargo check`
only needs the ROCm *headers* on disk for `rocm-rs`'s bindgen, not actual
GPU access. Those device flags (plus `--group-add keep-groups` or
equivalent render-group access) only matter once we're *running*
`crane-serve` against the real GPU in step 8.

### Base image for step 7 — decided

**Decision: `rocm/dev-ubuntu-24.04:7.14.0-full`**, not
`rocm/pytorch`, and **not `rocm/dev-ubuntu-26.04`** despite that tag
existing on Docker Hub (same digest/size as the 24.04 one, published the
same day) — pushing back on this specifically because AMD's own ROCm
7.14.0 compatibility docs
(`rocm.docs.amd.com/.../compatibility/compatibility-matrix.html`, via the
system-requirements page) are explicit that our exact GPU is scoped to a
short list of operating systems:

> AMD Radeon (RX 9070 XT, RX 9070 GRE, RX 9070, RX 9060 XT LP, RX 9060 XT,
> RX 9060, RX 7900 XTX, ...) only support **Ubuntu 24.04.4, Ubuntu 22.04.5,
> RHEL 10.1, and RHEL 9.7**.

Ubuntu 26.04 does not appear anywhere in ROCm 7.14.0's supported-OS table
at all (it lists 24.04.4, 22.04.5, several RHEL/SLES/Debian/Rocky/Oracle
versions — no 26.04). The `rocm/dev-ubuntu-26.04:7.14.0-full` image
existing on Docker Hub doesn't mean AMD has validated ROCm-on-RDNA4
against it; it's not clear from the tag alone whether it's even installing
packages built for 26.04's own apt suite or reusing 24.04's under the
hood. Given we're already carrying one unvalidated-version risk (running
the candle `rocm` feature, itself only tested on ROCm 7.2.4, on top of
7.14), stacking a second, officially-unsupported-for-this-GPU OS version
on top isn't worth it — **sticking to `dev-ubuntu-24.04` keeps the OS
variable inside AMD's own documented support matrix for the RX 9070 XT**,
leaving `gfx1201`-on-7.14 as the one already-accepted variable rather than
two compounding ones.

- You confirmed your host's current ROCm is **7.14** — the same version you
  already proved works end-to-end on this exact `gfx1201` card (Kokoro
  training). Matching that version in the container is safer than matching
  the version candle's `feat/rocm-backend` branch happened to be
  developed/tuned against (**ROCm 7.2.4**, and on `gfx1101`/RDNA3, not
  `gfx1201`) — RDNA4 support has been visibly evolving fast across ROCm
  releases (recall the `ollama` `gfx1201`-init-failure report was against
  HIP SDK 7.1), so leaning on your own newer, proven-working version is
  the lower-risk bet, even though it means the candle branch's `rocm`
  feature is being run on a ROCm version nobody upstream has specifically
  validated it against. Worth watching for in steps 5/8.
- `rocm/dev-ubuntu-*` images are built from AMD's own
  `ROCm/ROCm-docker` Dockerfiles. The plain `dev-ubuntu-24.04` tag installs
  just the `rocm-dev` meta-package (this is what provides `hipcc` and
  `clang-offload-bundler`); the `-complete` suffix (renamed `-full` in
  newer releases — `7.14.0-full` is the only 7.14 tag published, no bare
  `7.14.0`) additionally installs `rocm-libs`, the runtime library
  meta-package (`rocBLAS`, `rocRAND`, `rocSPARSE`, `MIOpen`, etc.) that
  candle's `rocm`/`miopen` features actually link against. **We need the
  `-full`/`-complete` variant**, not the bare one, since candle needs both
  halves (the compiler for runtime JIT, the libraries for matmul/RNG/conv).
- `rocm/pytorch` is a superset of the same `rocm-dev` + `rocm-libs` layers
  plus a full PyTorch/Python/torchvision stack Crane (a pure Rust binary)
  doesn't need — its tags run **~19-29 GB** vs. `dev-ubuntu-24.04:7.14.0-full`'s
  **~7.9 GB**. No packaging or version advantage for our use case, just
  extra size and attack surface.
- Still need to add on top: a **Rust toolchain** (`rustup` or distro
  packages) — not present in `rocm/dev-ubuntu-*`. A multi-stage build can
  still help here (drop the Rust build toolchain, crates.io registry
  cache, and intermediate build artifacts from the final image), but the
  **final stage still has to be the same ROCm dev/full base**, not a
  slimmer runtime-only one, because `hipcc`/`clang-offload-bundler` are
  needed at container *runtime* for the first-use kernel JIT, not just at
  `cargo build` time.
- Separate from the base image, running under **Podman** will need
  `--device /dev/kfd --device /dev/dri` (plus render/video group access)
  passed to `podman run` for the container to see the GPU at all — a
  run-time flag, not something the base image itself provides.

No Containerfile has been written yet — this section is planning only.

## Which model/quant fits the RX 9070 XT (16GB)?

Crane's own `data/crane-model-download --list` already lists
`qwen3.5-9b-gguf` (`unsloth/Qwen3.5-9B-GGUF`, default `--quant Q4_K_M`) as a
supported target, confirming Qwen 3.5 9B is the intended model — and
`crane-core`'s Qwen3.5 implementation has a **full GGUF loading path**
(`crane-core/src/models/qwen3_5/model.rs:146-309`'s `from_gguf`, dispatched
via `ModelFormat::Gguf`), not just safetensors, so a quantized GGUF is a
first-class option, not a fallback.

### Architecture (from `Qwen/Qwen3.5-9B`'s `config.json`)

- 32 total layers, hybrid per `full_attention_interval: 4`: **8 full-attention
  layers** (every 4th) + **24 GDN linear-attention layers**.
- Full-attention: `num_key_value_heads: 4`, `head_dim: 256` (GQA).
- GDN layers: fixed-size recurrent state, not a growing KV cache — matches
  what the explore-agent found in `crane-core/src/models/qwen3_5/kv_cache.rs`
  (only full-attention layers allocate a KV cache).
- `hidden_size: 4096`, `vocab_size: 248320`, `max_position_embeddings: 262144`
  (256K, unlikely to be used at full length in practice).

**KV cache cost**: only the 8 full-attention layers pay it —
`2 (K+V) × 4 kv_heads × 256 head_dim × 8 layers = 16384 elements/token` ≈
**32 KB/token at f16** (Crane's `CRANE_KV_QUANT`: `int8` ≈ 16 KB/token,
`int4` ≈ 8 KB/token). E.g. 32K tokens of context ≈ 1 GB at f16, ≈ 500 MB at
int8 — small relative to the 16GB budget even at `crane-serve`'s default
`--max-concurrent 16` continuous-batching depth, unless most of those 16
slots are individually running near the 256K context ceiling.

### GGUF file sizes (`unsloth/Qwen3.5-9B-GGUF`, same `--quant` values
`crane-model-download --quant <...>` accepts)

| Quant | Size | Fits in 16GB w/ headroom for KV cache + `mmproj` (~0.9GB) + ROCm runtime? |
|---|---|---|
| `BF16` | 17.92 GB | **No** — larger than total VRAM |
| `UD-Q8_K_XL` | 12.97 GB | Tight (~3GB left) — fine for light/low-concurrency use, risky at high `--max-concurrent`/long context |
| `Q8_0` | 9.53 GB | Comfortable (~6GB left) — near-lossless quality |
| `UD-Q6_K_XL` | 8.76 GB | Comfortable |
| `Q6_K` | 7.46 GB | Comfortable |
| `UD-Q5_K_XL` | 6.74 GB | Comfortable, generous headroom |
| `Q5_K_M` | 6.58 GB | Comfortable, generous headroom |
| `Q5_K_S` | 6.36 GB | Comfortable, generous headroom |
| `UD-Q4_K_XL` | 5.97 GB | Very generous headroom |
| `Q4_K_M` (script default) | 5.68 GB | Very generous headroom |
| `Q4_K_S` | 5.39 GB | Very generous headroom |
| `IQ4_NL` / `IQ4_XS` | 5.37 / 5.17 GB | Very generous headroom |
| `Q3_K_M` / `Q3_K_S` / `UD-Q2/Q3` / `IQ2/IQ3` variants | 3.19-4.67 GB | Most headroom, more quality loss — not needed given how much VRAM is free at Q4-Q6 already |

### Recommendation

For a 9B model on a 16GB card, VRAM is not the binding constraint at any of
the mainstream quant levels — even `Q8_0` (9.53 GB) leaves ~6GB free for KV
cache, the vision projector, and ROCm/JIT overhead. Given that headroom:

- **`Q6_K` (7.46 GB) or `Q5_K_M` (6.58 GB)** is a good default: near the
  quality of `Q8_0` at meaningfully smaller size, leaving very comfortable
  room for `--max-concurrent` batching and longer contexts.
- **`Q8_0` (9.53 GB)** if prioritizing quality over concurrency/context
  headroom (e.g. mostly single-request use) — still comfortably fits.
- The script's own default, **`Q4_K_M` (5.68 GB)**, is safe but leaves the
  card under-used for a 9B model at 16GB — worth overriding with
  `--quant Q5_K_M` or `--quant Q6_K` unless minimizing download size or RAM
  matters more than quality.
- Avoid `BF16` (doesn't fit) and treat `UD-Q8_K_XL` as workable only for
  light/low-concurrency serving, since it leaves the least headroom for
  `crane-serve`'s continuous-batching KV cache under load.
