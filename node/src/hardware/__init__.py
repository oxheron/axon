import logging
import os
import re
import socket
import subprocess

LOGGER = logging.getLogger(__name__)


def _detect_vram_gb_rocm_smi() -> float:
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "-d", "0"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0.0
    text = (proc.stdout or "") + (proc.stderr or "")
    match = re.search(r"VRAM Total Memory \(MiB\):\s*([0-9]+)", text)
    if match:
        return float(match.group(1)) / 1024.0
    match = re.search(r"VRAM Total Memory \(B\):\s*([0-9]+)", text)
    if match:
        return float(match.group(1)) / (1024**3)
    return 0.0


def detect_vram_gb() -> float:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.total / (1024**3)
    except Exception:  # noqa: BLE001
        pass

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
    except Exception:  # noqa: BLE001
        pass

    rocm_gb = _detect_vram_gb_rocm_smi()
    if rocm_gb > 0.0:
        return rocm_gb
    return 0.0


def detect_torch_accelerator() -> str:
    try:
        import torch
    except ImportError:
        return "none"
    if getattr(torch.version, "hip", None):
        return "rocm"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "none"


def _preflight_vllm_for_accelerator(accel: str) -> None:
    if accel != "rocm":
        return
    try:
        import vllm._rocm_C  # noqa: F401, PLC0415
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "ROCm PyTorch is active, but this vLLM build does not load `vllm._rocm_C` (%s). "
            "Install a ROCm-matched vLLM build; the default PyPI CUDA wheel will not run on this host.",
            exc,
        )
        return
    try:
        import amdsmi  # noqa: F401, PLC0415
    except ImportError:
        LOGGER.warning(
            "ROCm detected but Python package `amdsmi` is not installed; "
            "vLLM may not auto-detect the ROCm platform."
        )


def build_vllm_worker_environ(accel: str) -> dict[str, str]:
    env = os.environ.copy()
    if accel == "rocm":
        env.setdefault("VLLM_TARGET_DEVICE", "rocm")
    elif accel == "cuda":
        env.setdefault("VLLM_TARGET_DEVICE", "cuda")
    return env


def detect_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()
