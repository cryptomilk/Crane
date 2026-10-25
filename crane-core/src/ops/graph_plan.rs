// SPDX-License-Identifier: MIT
//! Replay a fixed-shape forward pass as a captured GPU graph, with
//! explicitly owned mutable inputs. No exported graph/kernel assets or
//! Python runtime are involved.
//!
//! `GraphPlan` has a CUDA path (`cudarc`'s `CudaGraph`) and a `ROCm` path
//! (candle-rocm's `HipGraphCapture`/`HipGraph`/`HipGraphExec`), selected by
//! Cargo feature. The two share no code: HIP graph replay additionally
//! needs address-stable intermediate allocations across replays, which the
//! `ROCm` path gets by routing the captured region's allocations through a
//! `VmmPool` (see `RocmDevice::begin_graph_capture`) instead of the
//! bucketed allocator every other call uses.
#[cfg(feature = "rocm")]
use candle_core::DeviceLocation;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{
        CudaGraph, CudaStream,
        sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
    },
};
#[cfg(feature = "rocm")]
use candle_core::rocm_backend::{HipGraphExec, RocmDevice, VmmPool, WrapErr};
use candle_core::{Result, Tensor, Var};
use std::sync::Arc;

/// A captured, replayable GPU graph over a fixed-shape forward pass.
#[cfg(feature = "cuda")]
pub struct GraphPlan {
    inputs: Vec<Var>,
    outputs: Vec<Tensor>,
    graph: Option<CudaGraph>,
    stream: Arc<CudaStream>,
}
// CudaGraph itself is not Sync. This owner only exposes replay through &mut
// self, serializes destruction, and the stream binds its context on each call.
#[cfg(feature = "cuda")]
unsafe impl Send for GraphPlan {}

/// A captured, replayable GPU graph over a fixed-shape forward pass.
#[cfg(feature = "rocm")]
pub struct GraphPlan {
    inputs: Vec<Var>,
    outputs: Vec<Tensor>,
    exec: Option<HipGraphExec>,
    // Keeps the captured region's intermediate allocations at the addresses
    // baked into `exec`'s kernel launches; see the module doc. Held only for
    // its Drop side effect (freeing the VMM reservation once `exec` no
    // longer needs it), never read. `Drop` below takes `exec` before `pool`
    // explicitly, since `GraphCaptureSession::end`'s pool-lifetime contract
    // (candle-rocm's `alloc.rs`) requires `exec` be torn down first; this
    // must not depend on the two fields' declaration order here.
    pool: Option<Arc<VmmPool>>,
    device: RocmDevice,
}
// No manual Send/Sync impl needed here, unlike the CUDA path above:
// `HipGraphExec` and `VmmPool` are already `Send + Sync`, and `RocmDevice`
// is `Clone` over `Arc`-wrapped state, so every field is Send on its own.

#[cfg(feature = "cuda")]
impl GraphPlan {
    /// Warm kernels/metadata, capture, instantiate, and initialize outputs.
    ///
    /// # Safety
    /// External allocations referenced by `forward` (for example model
    /// weights) must outlive this plan. Captured operations must use the
    /// supplied device stream and must not depend on host input values.
    /// Mutable inputs are owned by the plan; replay returns independent copies.
    pub unsafe fn capture<F>(inputs: &[&Tensor], mut forward: F) -> Result<Self>
    where
        F: FnMut(&[Tensor]) -> Result<Vec<Tensor>>,
    {
        if inputs.is_empty()
            || !inputs[0].device().is_cuda()
            || inputs
                .iter()
                .any(|x| !x.device().same_device(inputs[0].device()))
        {
            candle_core::bail!("CUDA graph inputs must share one device/stream");
        }
        let dev = inputs[0].device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        let _params = dev.enable_cuda_graph_htod_cache();
        let variables = inputs
            .iter()
            .map(|x| {
                let variable = Var::zeros(x.shape().clone(), x.dtype(), x.device())?;
                variable.set(x)?;
                Ok(variable)
            })
            .collect::<Result<Vec<_>>>()?;
        let values = variables.iter().map(|x| x.detach()).collect::<Vec<_>>();
        for _ in 0..2 {
            let _ = forward(&values)?;
        }
        stream.synchronize().w()?;
        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .w()?;
        let outputs = forward(&values);
        // End capture even when the closure rejects an operation.
        let graph = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .w();
        let outputs = outputs?;
        let graph = graph?.ok_or_else(|| candle_core::Error::Msg("empty CUDA graph".into()))?;
        graph.upload().w()?;
        graph.launch().w()?;
        stream.synchronize().w()?;
        Ok(Self {
            inputs: variables,
            outputs,
            graph: Some(graph),
            stream,
        })
    }

    pub fn replay(&mut self, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        if !self.matches(inputs) {
            candle_core::bail!("CUDA graph input signature changed");
        }
        for (dst, src) in self.inputs.iter().zip(inputs) {
            dst.set(src)?;
        }
        self.graph.as_ref().expect("live CUDA graph").launch().w()?;
        // Queue copies before releasing exclusive access: later replays can
        // safely overwrite graph buffers without changing returned tensors.
        self.outputs.iter().map(Tensor::copy).collect()
    }
}

#[cfg(feature = "rocm")]
impl GraphPlan {
    /// Warm kernels/metadata, capture, instantiate, and initialize outputs.
    ///
    /// # Safety
    /// External allocations referenced by `forward` (for example model
    /// weights) must outlive this plan. Captured operations must use the
    /// supplied device stream and must not depend on host input values.
    /// Mutable inputs are owned by the plan; replay returns independent copies.
    ///
    /// `forward`'s rocBLAS calls must not race a concurrent rocBLAS handle
    /// teardown on another thread -- `HipGraphCapture`'s caller obligation.
    /// Crane satisfies this structurally rather than with a lock: a
    /// `RocmDevice` creates its one rocBLAS handle at construction and never
    /// tears it down before the device itself drops, so there is no
    /// concurrent handle lifecycle event to race during capture.
    ///
    /// # Errors
    /// Returns an error if `inputs` is empty, spans more than one device, or
    /// isn't on a `ROCm` device; if `forward` itself errors during warmup or
    /// capture; if the captured region records no GPU operations; or if any
    /// underlying HIP call (capture, instantiate, upload, launch, VMM pool
    /// allocation) fails.
    pub unsafe fn capture<F>(inputs: &[&Tensor], mut forward: F) -> Result<Self>
    where
        F: FnMut(&[Tensor]) -> Result<Vec<Tensor>>,
    {
        if inputs.is_empty()
            || !inputs[0].device().is_rocm()
            || inputs
                .iter()
                .any(|x| !x.device().same_device(inputs[0].device()))
        {
            candle_core::bail!("HIP graph inputs must share one device/stream");
        }
        let device = inputs[0].device().as_rocm_device()?.clone();
        let DeviceLocation::Rocm { gpu_id } = inputs[0].device().location() else {
            unreachable!("validated is_rocm() above");
        };
        // Device ordinals are small (bounded by the machine's GPU count),
        // far under i32::MAX, so this never truncates or wraps to a
        // negative value.
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let pool = Arc::new(VmmPool::new(gpu_id as i32).w()?);
        let variables = inputs
            .iter()
            .map(|x| {
                let variable = Var::zeros(x.shape().clone(), x.dtype(), x.device())?;
                variable.set(x)?;
                Ok(variable)
            })
            .collect::<Result<Vec<_>>>()?;
        let values = variables.iter().map(|x| x.detach()).collect::<Vec<_>>();
        for _ in 0..2 {
            let _ = forward(&values)?;
        }
        device.synchronize()?;
        // Routes the captured region's allocations through `pool` instead of
        // the bucketed free list, so the addresses this capture bakes into
        // `exec`'s kernel launches stay valid instead of being recycled the
        // moment `forward`'s intermediates drop.
        let session = device.begin_graph_capture(pool.clone())?;
        let outputs = forward(&values);
        // End capture even when the closure rejects an operation.
        let graph = session.end().w();
        let outputs = outputs?;
        let graph = graph?.ok_or_else(|| candle_core::Error::Msg("empty HIP graph".into()))?;
        // HIP returns a valid, zero-node graph for an empty capture instead
        // of `None` (unlike CUDA's `end_capture` above), so emptiness has to
        // be checked explicitly rather than inferred from the `Option`.
        // `session.end()` already committed `pool`'s bump-allocator
        // watermark by this point, so this bail does not roll it back; that's
        // fine here because `pool` goes out of scope on this `Err` return,
        // freeing the whole VMM reservation anyway.
        if graph.is_empty().w()? {
            candle_core::bail!("empty HIP graph");
        }
        let exec = graph.instantiate().w()?;
        exec.upload(device.stream()).w()?;
        exec.launch(device.stream()).w()?;
        device.synchronize()?;
        Ok(Self {
            inputs: variables,
            outputs,
            exec: Some(exec),
            pool: Some(pool),
            device,
        })
    }

    /// Replay the captured graph against `inputs`, returning fresh copies of
    /// the captured outputs.
    ///
    /// # Panics
    /// Panics if called on a plan whose graph was already torn down, which
    /// cannot happen through the public API: every `GraphPlan` returned by
    /// [`Self::capture`] holds a live graph until it is dropped.
    ///
    /// # Errors
    /// Returns an error if `inputs`' shapes/dtypes/device don't match the
    /// tensors this plan was captured against, or if the underlying HIP
    /// launch or output copy fails.
    pub fn replay(&mut self, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        if !self.matches(inputs) {
            candle_core::bail!("HIP graph input signature changed");
        }
        for (dst, src) in self.inputs.iter().zip(inputs) {
            dst.set(src)?;
        }
        self.exec
            .as_ref()
            .expect("live HIP graph")
            .launch(self.device.stream())
            .w()?;
        // Queue copies before releasing exclusive access: later replays can
        // safely overwrite graph buffers without changing returned tensors.
        self.outputs.iter().map(Tensor::copy).collect()
    }
}

impl GraphPlan {
    /// Whether `inputs` has the same shapes/dtypes/device as the tensors
    /// this plan was captured against, i.e. whether [`Self::replay`] can
    /// reuse the captured graph for `inputs` without re-capturing.
    #[must_use]
    pub fn matches(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == self.inputs.len()
            && self.inputs.iter().zip(inputs).all(|(a, b)| {
                a.dims() == b.dims() && a.dtype() == b.dtype() && a.device().same_device(b.device())
            })
    }
}

#[cfg(feature = "cuda")]
impl Drop for GraphPlan {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        self.outputs.clear();
        let _ = self.stream.synchronize();
        self.graph.take();
    }
}

#[cfg(feature = "rocm")]
impl Drop for GraphPlan {
    fn drop(&mut self) {
        let _ = self.device.synchronize();
        self.outputs.clear();
        let _ = self.device.synchronize();
        // `exec` must be destroyed before `pool`'s VMM reservation is freed
        // (its kernel launches reference addresses allocated from `pool`),
        // so both are torn down explicitly in order here rather than relying
        // on the struct's field declaration order to do it implicitly.
        self.exec.take();
        self.pool.take();
    }
}
