"""Vocal / background separation with Demucs (htdemucs) with live chunk progress & cancel support."""
import gc
import os
import sys
from pathlib import Path
import numpy as np


def _load_demucs_model():
    # 1. Try local models folder (next to package or next to executable)
    for base in [
        Path(sys.executable).parent,
        Path(__file__).parent.parent,
        Path(os.path.dirname(os.path.abspath(__file__))).parent,
        Path(getattr(sys, "_MEIPASS", ".")),
    ]:
        local_dir = base / "models" / "htdemucs"
        yaml_file = local_dir / "htdemucs.yaml"
        safetensors_file = local_dir / "955717e8.safetensors"
        if yaml_file.exists() and safetensors_file.exists():
            import yaml
            from demucs.hf import load_safetensors_model
            from demucs.repo import BagOfModels
            with open(yaml_file, "r", encoding="utf-8") as f:
                bag = yaml.safe_load(f)
            models = [load_safetensors_model(local_dir / f"{sig}.safetensors") for sig in bag["models"]]
            bag_model = BagOfModels(models, bag.get("weights"), bag.get("segment"))
            bag_model.eval()
            return bag_model

    # 2. Try Hugging Face Hub (adefossez/HTDemucs)
    try:
        from demucs.hf import get_hf_model
        model = get_hf_model("htdemucs")
        model.eval()
        return model
    except Exception:
        pass

    # 3. Fallback to default Demucs get_model
    from demucs.pretrained import get_model
    model = get_model("htdemucs")
    model.eval()
    return model


def separate(audio, log=print, use_gpu=True, progress=None, cancelled=None):
    """audio: float32 (2, n) at 44.1 kHz. Returns (vocals, background), same shape."""
    class _SafeStream:
        def write(self, s): pass
        def flush(self): pass
        def isatty(self): return False

    if sys.stdout is None or not hasattr(sys.stdout, "write"):
        sys.stdout = _SafeStream()
    if sys.stderr is None or not hasattr(sys.stderr, "write"):
        sys.stderr = _SafeStream()

    import torch
    import torch.hub

    _orig_load = torch.hub.load_state_dict_from_url
    def _safe_load_state_dict(url, *args, **kwargs):
        kwargs["progress"] = False
        return _orig_load(url, *args, **kwargs)
    torch.hub.load_state_dict_from_url = _safe_load_state_dict

    from demucs.apply import apply_model

    model = _load_demucs_model()
    model.eval()
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    wav = torch.from_numpy(audio)
    total_len = max(1, wav.shape[-1])
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav_n = (wav - mean) / std

    def _demucs_cb(info_dict):
        if cancelled and cancelled():
            raise InterruptedError("Đã hủy")
        if progress and isinstance(info_dict, dict) and info_dict.get("state") == "end":
            off = float(info_dict.get("segment_offset", 0))
            seg_samples = float((model.segment or 7.8) * model.samplerate)
            frac = min(0.98, (off + seg_samples) / total_len)
            progress(frac)

    def _run(dev):
        log(f"Demucs đang tách giọng & nhạc nền trên {dev.upper()}...")
        with torch.no_grad():
            return apply_model(
                model, wav_n[None], device=dev, split=True, overlap=0.25,
                shifts=0, progress=False, num_workers=0, callback=_demucs_cb
            )[0]

    try:
        sources = _run(device)
    except InterruptedError:
        raise
    except RuntimeError as e:  # CUDA OOM -> CPU
        if device == "cuda":
            log(f"GPU không đủ bộ nhớ ({e.__class__.__name__}), tự động chuyển sang CPU")
            torch.cuda.empty_cache()
            sources = _run("cpu")
        else:
            raise

    sources = sources * std + mean
    idx = model.sources.index("vocals")
    vocals = sources[idx].cpu().numpy()
    background = (sources.sum(0) - sources[idx]).cpu().numpy()
    del model, sources
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if progress:
        progress(1.0)
    return vocals.astype(np.float32), background.astype(np.float32)
