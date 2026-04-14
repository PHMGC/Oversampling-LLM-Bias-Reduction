"""GPU-aware parallel job dispatch for training and evaluation pipelines.

Each job runs in a daemon thread with its own torch.cuda.Stream. PyTorch
streams are thread-local, so the GPU can execute multiple streams concurrently
while the OS sees a single process (single nvidia-smi entry). Parallelism is
determined at runtime by probing free VRAM on every allowed GPU and computing
how many jobs fit per GPU based on the estimated memory weight per job.

Falls back to sequential execution in the current process only when no GPU slot
is available at all (total_slots == 0).

Usage
-----
    from src.parallel_utils import JobSpec, run_parallel_jobs, estimate_job_memory_gb

    weight_gb = estimate_job_memory_gb(MODEL_NAME)

    jobs = [JobSpec(job_id=did, args=(did, tr, MODEL, out, 3), weight_gb=weight_gb)
            for did, tr, out in ...]
    results = run_parallel_jobs(jobs, train_fn=train_one_job)
"""

from __future__ import annotations

import importlib.util
import logging
import os
import queue as _queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class JobSpec:
    """One unit of work dispatched to a subprocess.

    Attributes:
        job_id:    Human-readable identifier, e.g. "baseline/mcauley/cds_reviews".
        args:      Positional arguments forwarded to train_fn.
        kwargs:    Keyword arguments forwarded to train_fn.
        weight_gb: Estimated peak VRAM requirement in GB for this job. Used by
                   probe_gpu_memory to compute how many jobs fit per GPU. Obtain
                   this value from estimate_job_memory_gb() rather than guessing.
    """
    job_id: str
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    weight_gb: float = 0.0  # must be set; 0.0 triggers a warning in run_parallel_jobs


@dataclass
class GpuInfo:
    """Available parallelism for a single GPU."""
    index: int       # physical GPU index as reported by nvidia-smi / pynvml
    free_gb: float   # free VRAM in GB at probe time
    slots: int       # floor(free_gb / job_weight_gb)


@dataclass
class JobResult:
    """Outcome of a single dispatched job."""
    job_id: str
    gpu_index: int
    success: bool
    return_value: Any = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# GPU probing
# ---------------------------------------------------------------------------

def estimate_job_memory_gb(
    model_name_or_path: str,
    safety_factor: float = 2.5,
) -> float:
    """Estimate peak VRAM (GB) for fine-tuning a HuggingFace model with AdamW.

    Loads only the model config (no weights) to count parameters, then applies:
      - fp16 model weights:        params × 2 bytes
      - fp32 gradients:            params × 4 bytes
      - AdamW optimizer states:    params × 8 bytes  (m and v in fp32)
      Total fixed cost:            params × 14 bytes

    Multiplies by safety_factor (default 2.5) to account for activation memory,
    framework overhead, and dataset batch buffering.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        safety_factor:      Multiplier applied on top of the parameter-based
                            estimate to cover activations and overhead.

    Returns:
        Estimated peak VRAM in GB.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path)

    # Instantiate on the meta device: counts parameters without allocating memory.
    with torch.device("meta"):
        model = AutoModel.from_config(config)

    num_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 2 + 4 + 8   # fp16 weights + fp32 grads + AdamW states
    base_gb = (num_params * bytes_per_param) / (1024 ** 3)
    estimate = base_gb * safety_factor

    logger.info(
        "Memory estimate for %s: %.0fM params × %d bytes × %.1f safety = %.2f GB",
        model_name_or_path, num_params / 1e6, bytes_per_param, safety_factor, estimate,
    )
    return estimate


def probe_gpu_memory(
    job_weight_gb: float,
    min_free_gb: float = 2.0,
) -> List[GpuInfo]:
    """Return one GpuInfo per eligible GPU visible to this process.

    Only GPUs listed in CUDA_VISIBLE_DEVICES (if set) are considered. GPUs
    with less than min_free_gb of free memory are excluded.

    Slots per GPU are computed as floor(free_gb / job_weight_gb).

    Args:
        job_weight_gb: Estimated peak VRAM per job in GB. Use estimate_job_memory_gb()
                       to obtain this value from the model rather than guessing.
        min_free_gb:   Minimum free VRAM (GB) for a GPU to be considered usable.

    Returns:
        List of GpuInfo sorted by free_gb descending. May be empty if no GPU
        has enough free memory or if CUDA is unavailable.
    """
    allowed = _allowed_gpu_indices()

    raw = _probe_via_pynvml() if _pynvml_available() else _probe_via_nvidiasmi()

    infos: List[GpuInfo] = []
    for idx, free_gb in raw:
        if allowed is not None and idx not in allowed:
            continue
        if free_gb < min_free_gb:
            continue
        slots = max(0, int(free_gb // job_weight_gb))
        infos.append(GpuInfo(index=idx, free_gb=free_gb, slots=slots))

    infos.sort(key=lambda g: g.free_gb, reverse=True)
    return infos


def _allowed_gpu_indices() -> Optional[set]:
    """Parse CUDA_VISIBLE_DEVICES and return the set of allowed physical indices.

    Returns None if CUDA_VISIBLE_DEVICES is unset (all GPUs allowed).
    Returns an empty set if CUDA_VISIBLE_DEVICES is set to an invalid/empty value
    (meaning no GPU is allowed).
    """
    val = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not val:
        return None  # unset — all GPUs allowed
    try:
        return {int(x.strip()) for x in val.split(",") if x.strip()}
    except ValueError:
        logger.warning(
            "Could not parse CUDA_VISIBLE_DEVICES=%r — treating as no restriction", val
        )
        return None


def _pynvml_available() -> bool:
    return importlib.util.find_spec("pynvml") is not None


def _probe_via_pynvml() -> List[Tuple[int, float]]:
    """Return (physical_index, free_gb) pairs using pynvml."""
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    results = []
    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        results.append((i, mem.free / (1024 ** 3)))
    pynvml.nvmlShutdown()
    return results


def _probe_via_nvidiasmi() -> List[Tuple[int, float]]:
    """Return (physical_index, free_gb) pairs by parsing nvidia-smi output.

    Falls back to this when pynvml is not installed.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("nvidia-smi probe failed: %s", exc)
        return []

    results = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                idx = int(parts[0])
                free_mib = float(parts[1])
                results.append((idx, free_mib / 1024.0))
            except ValueError:
                continue
    return results


# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------

def run_parallel_jobs(
    jobs: List[JobSpec],
    train_fn: Callable[..., Any],
    min_free_gb: float = 2.0,
    result_timeout: Optional[float] = None,
    max_jobs_per_gpu: int = 1,
) -> List[JobResult]:
    """Dispatch jobs across available GPU slots and collect results.

    Probes visible GPUs and computes how many jobs fit per GPU using each
    JobSpec's weight_gb. max_jobs_per_gpu caps the concurrency per GPU
    regardless of available VRAM — GPU compute is a shared resource, and
    saturating it with too many concurrent jobs reduces total throughput.
    The default of 1 (sequential per GPU) is fastest for compute-bound jobs.

    Falls back to sequential in-process execution only when no GPU slot is
    available at all (total_slots == 0), i.e., CUDA is unavailable or all
    GPUs have insufficient free memory.

    Args:
        jobs:             Jobs to execute. Each must have weight_gb set.
        train_fn:         Callable invoked as train_fn(*job.args, **job.kwargs).
        min_free_gb:      GPUs with less free VRAM than this are excluded.
        result_timeout:   Seconds to wait for a result. None = infinite.
        max_jobs_per_gpu: Hard cap on concurrent jobs per GPU (default 1).

    Returns:
        List of JobResult in order of completion.
    """
    if max_jobs_per_gpu < 1:
        raise ValueError(f"max_jobs_per_gpu must be >= 1, got {max_jobs_per_gpu}")

    if not jobs:
        return []

    weights = [j.weight_gb for j in jobs]
    if any(w <= 0 for w in weights):
        logger.warning(
            "Some JobSpec instances have weight_gb <= 0. "
            "Use estimate_job_memory_gb() to set a meaningful value."
        )

    # Use the maximum weight across all jobs for conservative slot calculation.
    # This ensures the GPU is never overcommitted even for the heaviest job.
    max_weight = max(weights)
    gpu_infos = probe_gpu_memory(job_weight_gb=max_weight, min_free_gb=min_free_gb)
    total_slots = sum(g.slots for g in gpu_infos)

    if total_slots == 0:
        _log_fallback(gpu_infos)
        return _run_sequential(jobs, train_fn)

    logger.info(
        "Parallel dispatch: %d jobs across %d GPU(s), max_jobs_per_gpu=%d "
        "(weight=%.2f GB/job)",
        len(jobs), len(gpu_infos), max_jobs_per_gpu, max_weight,
    )

    return _dispatch(jobs, train_fn, gpu_infos, result_timeout, max_jobs_per_gpu)


# ---------------------------------------------------------------------------
# Sequential fallback
# ---------------------------------------------------------------------------

def _run_sequential(
    jobs: List[JobSpec],
    train_fn: Callable[..., Any],
) -> List[JobResult]:
    """Execute all jobs in the current process, sequentially."""
    results = []
    for job in jobs:
        start = time.monotonic()
        try:
            rv = train_fn(*job.args, **job.kwargs)
            elapsed = time.monotonic() - start
            gpu_idx = _current_gpu_index()
            results.append(JobResult(
                job_id=job.job_id,
                gpu_index=gpu_idx,
                success=True,
                return_value=rv,
                elapsed_seconds=elapsed,
            ))
            logger.info("  [sequential] %s finished in %.0fs", job.job_id, elapsed)
        except Exception as exc:
            elapsed = time.monotonic() - start
            results.append(JobResult(
                job_id=job.job_id,
                gpu_index=_current_gpu_index(),
                success=False,
                error=str(exc),
                elapsed_seconds=elapsed,
            ))
            logger.error("  [sequential] %s failed: %s", job.job_id, exc)
    return results


def _current_gpu_index() -> int:
    """Return the first physical GPU index from CUDA_VISIBLE_DEVICES, or -1."""
    allowed = _allowed_gpu_indices()
    if allowed:
        return min(allowed)
    return 0


# ---------------------------------------------------------------------------
# Parallel dispatch
# ---------------------------------------------------------------------------

def _physical_to_cuda_index(physical_index: int) -> int:
    """Map a physical GPU index to its CUDA-visible device index.

    When CUDA_VISIBLE_DEVICES=3, the only visible device is cuda:0 (not cuda:3).
    This function performs that remapping so streams are created on the correct
    CUDA device index as seen by the current process.
    """
    val = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not val:
        return physical_index
    try:
        visible = [int(x.strip()) for x in val.split(",") if x.strip()]
        return visible.index(physical_index)
    except (ValueError, IndexError):
        return 0


def _dispatch(
    jobs: List[JobSpec],
    train_fn: Callable[..., Any],
    gpu_infos: List[GpuInfo],
    result_timeout: Optional[float],
    max_jobs_per_gpu: int = 1,
) -> List[JobResult]:
    """Streaming slot-based dispatch using threads and per-thread CUDA streams.

    Each job runs in a daemon thread. The thread creates its own
    torch.cuda.Stream (thread-local in PyTorch), so the GPU can execute
    multiple streams concurrently while the OS sees a single process.
    max_jobs_per_gpu caps concurrency below what memory alone would allow.
    """
    result_queue: _queue.Queue = _queue.Queue()

    slot_used: Dict[int, int] = {g.index: 0 for g in gpu_infos}
    slot_max:  Dict[int, int] = {g.index: min(g.slots, max_jobs_per_gpu) for g in gpu_infos}
    for g in gpu_infos:
        logger.info("  GPU %d: %.1f GB free, memory allows %d slot(s), capped at %d",
                    g.index, g.free_gb, g.slots, slot_max[g.index])

    pending: deque[JobSpec] = deque(jobs)
    # thread_ident -> (JobSpec, gpu_index, Thread)
    active: Dict[int, Tuple[JobSpec, int, threading.Thread]] = {}

    results: List[JobResult] = []

    while pending or active:
        # Fill all available slots
        for g in gpu_infos:
            while pending and slot_used[g.index] < slot_max[g.index]:
                job = pending.popleft()
                t = threading.Thread(
                    target=_worker_entry,
                    args=(job, train_fn, g.index, result_queue),
                    daemon=True,
                )
                t.start()
                active[t.ident] = (job, g.index, t)
                slot_used[g.index] += 1
                logger.info(
                    "  [dispatch] started %s on GPU %d (tid=%d)",
                    job.job_id, g.index, t.ident,
                )

        if not active:
            break

        # Block until one result arrives
        result: JobResult = result_queue.get(timeout=result_timeout)
        results.append(result)

        # Free the slot for the finished job
        ident_to_remove = None
        for ident, (job, gpu_idx, t) in active.items():
            if job.job_id == result.job_id:
                t.join(timeout=30)
                slot_used[gpu_idx] -= 1
                ident_to_remove = ident
                break

        if ident_to_remove is not None:
            del active[ident_to_remove]

        status = "OK" if result.success else f"FAILED: {result.error}"
        logger.info(
            "  [dispatch] %s on GPU %d finished in %.0fs  %s",
            result.job_id, result.gpu_index, result.elapsed_seconds, status,
        )

    return results


# ---------------------------------------------------------------------------
# Thread worker
# ---------------------------------------------------------------------------

def _worker_entry(
    job: JobSpec,
    train_fn: Callable[..., Any],
    gpu_index: int,
    result_queue: _queue.Queue,
) -> None:
    """Entry point executed in each thread.

    Creates a dedicated torch.cuda.Stream for this thread. In PyTorch, the
    current stream is thread-local, so each thread's CUDA ops are routed to
    its own stream. The GPU can execute multiple streams concurrently, giving
    true parallelism inside a single OS process (single nvidia-smi entry).
    """
    import torch

    cuda_idx = _physical_to_cuda_index(gpu_index)
    stream = torch.cuda.Stream(device=cuda_idx)

    start = time.monotonic()
    try:
        with torch.cuda.stream(stream):
            rv = train_fn(*job.args, **job.kwargs)
        elapsed = time.monotonic() - start
        result_queue.put(JobResult(
            job_id=job.job_id,
            gpu_index=gpu_index,
            success=True,
            return_value=rv,
            elapsed_seconds=elapsed,
        ))
    except Exception as exc:
        elapsed = time.monotonic() - start
        result_queue.put(JobResult(
            job_id=job.job_id,
            gpu_index=gpu_index,
            success=False,
            error=str(exc),
            elapsed_seconds=elapsed,
        ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_fallback(gpu_infos: List[GpuInfo]) -> None:
    if not gpu_infos:
        logger.info(
            "No eligible GPU found (CUDA unavailable or all allowed GPUs below "
            "memory threshold). Running sequentially in current process."
        )
    else:
        logger.info(
            "No GPU has enough free VRAM for even one job across %d GPU(s). "
            "Running sequentially.",
            len(gpu_infos),
        )
