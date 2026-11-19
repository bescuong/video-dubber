"""Automatic GPU / VRAM / Driver & CPU Hardware Profiler (`dubber/hw_profile.py`).

Detects:
- Active NVIDIA CUDA GPUs and classifies VRAM into optimal performance tiers:
    * `nvidia_high` (>= 10.5 GB VRAM): OmniVoice batch=6, Whisper fp16, NLLB fp16 batch=32
    * `nvidia_mid`  (5.0 - 10.4 GB VRAM, e.g. RTX 3050 6GB/8GB): OmniVoice batch=3..4, Whisper int8_float16, NLLB fp16 batch=16
    * `nvidia_low`  (< 5.0 GB VRAM, e.g. GTX 1060 3GB): OmniVoice batch=1 + CPU audio_tokenizer, Whisper int8, NLLB fp16 batch=8
- Physical NVIDIA GPUs whose driver is too old for bundled CUDA 12.6 (< 560.94) so the UI can warn
  the user immediately instead of silently running 10x slower on CPU!
- AMD / Intel GPUs for hardware video decoding/encoding (`d3d11va`, `h264_amf`, `h264_qsv`).
- Optimal worker thread counts for multi-threaded subtitle erasure (`Blurrer`) and CPU inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
import subprocess
import sys

CREATE_NO_WINDOW = 0x08000000 if hasattr(subprocess, "STARTUPINFO") else 0
MIN_CUDA126_DRIVER = 560.94


@dataclass
class HardwareProfile:
    tier: str                    # "nvidia_high" | "nvidia_mid" | "nvidia_low" | "amd_gpu" | "intel_gpu" | "cpu"
    vendor: str                  # "nvidia" | "amd" | "intel" | "cpu"
    gpu_name: str                # Display GPU name
    vram_gb: float               # Dedicated VRAM in GiB (0.0 if unknown/CPU)
    cuda_available: bool         # True if torch.cuda.is_available() succeeded
    compute_major: int           # CUDA sm major (e.g. 8 for RTX 30xx Ampere, 6 for GTX 10xx Pascal)
    driver_version: str          # NVIDIA driver string if detected (e.g. "566.14")
    driver_outdated: bool        # True if physical NVIDIA GPU exists on PC but CUDA failed to init!
    driver_warning: str          # Actionable Vietnamese message when driver_outdated is True
    # Auto-tuned AI & video parameters
    omni_batch_size: int         # Parallel sentences per OmniVoice diffusion call
    omni_tokenizer_cpu: bool     # Offload audio_tokenizer to CPU on <5GB GPUs
    whisper_compute: str         # "float16" | "int8_float16" | "int8"
    whisper_beam: int            # Beam size (3 = fast & accurate)
    nllb_batch_size: int         # NLLB-200 batch size
    ffmpeg_hwaccel: str          # "cuda" | "d3d11va" | ""
    blur_workers: int            # Parallel OpenCV threads for subtitle inpainting
    cpu_threads: int             # Optimal physical-core thread count for CPU torch/whisper
    badge_text: str              # Short human-readable badge for GUI status bar


_CACHED_PROFILE: HardwareProfile | None = None
_TORCH_OPT_APPLIED: bool = False


def _probe_nvidia_smi() -> tuple[str, str, float]:
    """Query `nvidia-smi` for `(gpu_name, driver_version, vram_gb)` even when PyTorch CUDA fails."""
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            creationflags=CREATE_NO_WINDOW,
        )
        if p.returncode == 0:
            line = p.stdout.decode("utf-8", errors="ignore").strip().splitlines()[0]
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3:
                vram_mb = float(re.findall(r"[\d.]+", parts[2])[0]) if re.findall(r"[\d.]+", parts[2]) else 0.0
                return parts[0], parts[1], round(vram_mb / 1024.0, 2)
    except Exception:
        pass
    return "", "", 0.0


def _probe_windows_display_adapters() -> list[str]:
    """Fast (~20ms) Windows Registry lookup of installed GPU adapter descriptions (works on Win10 & Win11 24H2)."""
    if sys.platform != "win32":
        return []
    names: list[str] = []
    try:
        import winreg
        cls_key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, cls_key) as root:
            idx = 0
            while True:
                try:
                    sub_name = winreg.EnumKey(root, idx)
                    idx += 1
                except OSError:
                    break
                if not sub_name.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub_name) as sk:
                        desc, _ = winreg.QueryValueEx(sk, "DriverDesc")
                        desc_s = str(desc).strip()
                        if desc_s and "basic" not in desc_s.lower() and desc_s not in names:
                            names.append(desc_s)
                except OSError:
                    continue
    except Exception:
        pass
    return names


def apply_torch_cuda_optimizations() -> None:
    """Enable TensorFloat-32 (TF32) & cuDNN autotuner on Ampere+ NVIDIA GPUs (RTX 30/40/50)."""
    global _TORCH_OPT_APPLIED
    if _TORCH_OPT_APPLIED:
        return
    try:
        import torch
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            torch.backends.cudnn.benchmark = True
            if major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
            _TORCH_OPT_APPLIED = True
    except Exception:
        pass


def configure_cpu_threads(profile: HardwareProfile | None = None) -> None:
    """Set PyTorch CPU thread count to physical core count to prevent hyper-threading contention."""
    prof = profile or get_hw_profile()
    try:
        import torch
        torch.set_num_threads(prof.cpu_threads)
    except Exception:
        pass


def get_hw_profile(force_refresh: bool = False) -> HardwareProfile:
    """Detect and return cached `HardwareProfile` for the current machine."""
    global _CACHED_PROFILE
    if _CACHED_PROFILE is not None and not force_refresh:
        return _CACHED_PROFILE

    logical_cpus = os.cpu_count() or 4
    cpu_threads = max(2, min(12, logical_cpus // 2 if logical_cpus >= 6 else logical_cpus))
    blur_workers = max(2, min(8, logical_cpus - 1 if logical_cpus > 2 else 2))

    smi_name, smi_drv, smi_vram = _probe_nvidia_smi()

    cuda_ok = False
    torch_gpu_name = ""
    torch_vram_gb = 0.0
    comp_major = 0
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            torch_gpu_name = props.name
            torch_vram_gb = round(props.total_memory / (1024 ** 3), 2)
            comp_major = int(props.major)
            apply_torch_cuda_optimizations()
    except Exception:
        cuda_ok = False

    if cuda_ok:
        vram = torch_vram_gb or smi_vram
        gname = torch_gpu_name or smi_name or "NVIDIA CUDA GPU"
        if vram >= 10.5:
            tier = "nvidia_high"
            omni_bs = 6 if comp_major >= 7 else 2
            omni_tok_cpu = False
            w_comp = "float16" if comp_major >= 7 else "int8"
            nllb_bs = 32
            tier_lbl = "Tối đa (High-VRAM)"
        elif vram >= 5.5:
            tier = "nvidia_mid"
            omni_bs = (4 if vram >= 7.2 else 3) if comp_major >= 7 else 1
            omni_tok_cpu = False
            w_comp = "int8_float16" if comp_major >= 7 else "int8"
            nllb_bs = 16
            tier_lbl = "Tăng tốc Cao (Mid-VRAM)"
        else:
            tier = "nvidia_low"
            omni_bs = 1
            omni_tok_cpu = True
            w_comp = "int8"
            nllb_bs = 8
            tier_lbl = "Tiết kiệm VRAM"

        prof = HardwareProfile(
            tier=tier,
            vendor="nvidia",
            gpu_name=gname,
            vram_gb=vram,
            cuda_available=True,
            compute_major=comp_major,
            driver_version=smi_drv,
            driver_outdated=False,
            driver_warning="",
            omni_batch_size=omni_bs,
            omni_tokenizer_cpu=omni_tok_cpu,
            whisper_compute=w_comp,
            whisper_beam=3,
            nllb_batch_size=nllb_bs,
            ffmpeg_hwaccel="cuda",
            blur_workers=blur_workers,
            cpu_threads=cpu_threads,
            badge_text=f"⚡ GPU: {gname} ({vram:.1f} GB · {tier_lbl})",
        )
        _CACHED_PROFILE = prof
        return prof

    # CUDA is NOT active -> inspect whether a physical NVIDIA / AMD / Intel GPU exists!
    adapters = _probe_windows_display_adapters()
    nv_adapter = smi_name or next((a for a in adapters if "nvidia" in a.lower() or "geforce" in a.lower() or "rtx" in a.lower() or "gtx" in a.lower()), "")
    amd_adapter = next((a for a in adapters if "amd" in a.lower() or "radeon" in a.lower()), "")
    intel_adapter = next((a for a in adapters if "intel" in a.lower() or "arc" in a.lower() or "iris" in a.lower()), "")

    if nv_adapter:
        drv_num = 0.0
        if smi_drv:
            m = re.findall(r"\d+(?:\.\d+)?", smi_drv)
            if m:
                drv_num = float(m[0])
        drv_info = f" (Driver hiện tại: {smi_drv})" if smi_drv else ""
        if 0 < drv_num < MIN_CUDA126_DRIVER:
            warn = (
                f"⚠ Phát hiện card {nv_adapter}{drv_info} nhưng Driver NVIDIA quá cũ (cần >= {MIN_CUDA126_DRIVER}) "
                f"nên CUDA 12.6 không thể kích hoạt! Ứng dụng đang phải chạy bằng CPU (chậm hơn 8–10 lần). "
                f"Hãy cập nhật NVIDIA Driver mới nhất để mở khóa toàn bộ tốc độ card rời!"
            )
        else:
            warn = (
                f"⚠ Phát hiện card {nv_adapter}{drv_info} nhưng CUDA chưa kích hoạt được! "
                f"Vui lòng cài đặt / cập nhật NVIDIA Game Ready hoặc Studio Driver mới nhất (>= 560.94) rồi mở lại ứng dụng."
            )
        prof = HardwareProfile(
            tier="cpu",
            vendor="nvidia",
            gpu_name=nv_adapter,
            vram_gb=smi_vram,
            cuda_available=False,
            compute_major=0,
            driver_version=smi_drv,
            driver_outdated=True,
            driver_warning=warn,
            omni_batch_size=1,
            omni_tokenizer_cpu=True,
            whisper_compute="int8",
            whisper_beam=3,
            nllb_batch_size=6,
            ffmpeg_hwaccel="d3d11va",
            blur_workers=blur_workers,
            cpu_threads=cpu_threads,
            badge_text=f"⚠ {nv_adapter} (Chưa bật được CUDA — Cần cập nhật Driver!)",
        )
        _CACHED_PROFILE = prof
        return prof

    if amd_adapter:
        prof = HardwareProfile(
            tier="amd_gpu",
            vendor="amd",
            gpu_name=amd_adapter,
            vram_gb=0.0,
            cuda_available=False,
            compute_major=0,
            driver_version="",
            driver_outdated=False,
            driver_warning="",
            omni_batch_size=1,
            omni_tokenizer_cpu=True,
            whisper_compute="int8",
            whisper_beam=3,
            nllb_batch_size=6,
            ffmpeg_hwaccel="d3d11va",
            blur_workers=blur_workers,
            cpu_threads=cpu_threads,
            badge_text=f"⚡ GPU: {amd_adapter} (Mã hóa phần cứng AMF)",
        )
        _CACHED_PROFILE = prof
        return prof

    if intel_adapter:
        prof = HardwareProfile(
            tier="intel_gpu",
            vendor="intel",
            gpu_name=intel_adapter,
            vram_gb=0.0,
            cuda_available=False,
            compute_major=0,
            driver_version="",
            driver_outdated=False,
            driver_warning="",
            omni_batch_size=1,
            omni_tokenizer_cpu=True,
            whisper_compute="int8",
            whisper_beam=3,
            nllb_batch_size=6,
            ffmpeg_hwaccel="d3d11va",
            blur_workers=blur_workers,
            cpu_threads=cpu_threads,
            badge_text=f"⚡ GPU: {intel_adapter} (Intel QSV)",
        )
        _CACHED_PROFILE = prof
        return prof

    prof = HardwareProfile(
        tier="cpu",
        vendor="cpu",
        gpu_name="CPU Mode",
        vram_gb=0.0,
        cuda_available=False,
        compute_major=0,
        driver_version="",
        driver_outdated=False,
        driver_warning="",
        omni_batch_size=1,
        omni_tokenizer_cpu=True,
        whisper_compute="int8",
        whisper_beam=3,
        nllb_batch_size=4,
        ffmpeg_hwaccel="",
        blur_workers=blur_workers,
        cpu_threads=cpu_threads,
        badge_text=f"💻 Chế độ CPU ({logical_cpus} luồng)",
    )
    _CACHED_PROFILE = prof
    return prof
