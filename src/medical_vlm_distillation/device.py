"""Pick the accelerator once: CUDA, then Apple Silicon (MPS), then CPU. Override with e.g. DEVICE=cpu."""

import os

import torch


def _pick() -> str:
    if env := os.environ.get("DEVICE"):
        return env
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick()
PIN_MEMORY = DEVICE == "cuda"  # pinned memory only helps host->CUDA copies; MPS warns and ignores it


def synchronize(device: str = DEVICE) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def empty_cache(device: str = DEVICE) -> None:
    """Return cached, unused allocator memory. Matters on MPS, where GPU memory is the system's RAM."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def device_name(device: str = DEVICE) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name()
    if device == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"
