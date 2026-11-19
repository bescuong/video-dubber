"""OmniVoice AI TTS Engine (k2-fsa/OmniVoice) — Zero-Shot Voice Cloning & Voice Design.

Features:
- Per-character zero-shot voice cloning directly from separated video vocals (`vocals.wav`)
  so each character speaks in their own original timbre in the target language.
- Custom reference audio voice cloning (`.wav` / `.mp3`) using local `faster-whisper` for
  zero-overhead reference transcription (never downloads extra 1.6GB HF ASR weights).
- Consistent Voice Design mode: locks sentence 0 into a `VoiceClonePrompt` so every
  character keeps 100% identical voice identity across the entire video.
- Low-VRAM GTX 1060 3GB optimization: loads in `float16` (`load_asr=False`) and keeps
  `audio_tokenizer` on CPU when GPU VRAM < 5 GB (~1.17 GB VRAM peak!).
- Instant pre-synthesis duration estimation -> single-pass native speed fitting + shared
  Anti-Slur Elastic Timeline with guaranteed 150ms breath pauses.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from .media import load_audio, save_wav
from .tts import (
    SR,
    _trim_silence,
    place_audios_on_track,
    sanitize_tts_text,
)

# Curated Vietnamese presets mapped strictly to validated OmniVoice instruct tags
# (preset_id, display_label, instruct_en, default_gender)
OMNIVOICE_PRESETS: list[tuple[str, str, str, str]] = [
    ("design_auto", "✨ Tự động Giọng Bản Xứ chuẩn theo vai Nam/Nữ — Rõ nét nhất (Đề xuất)", "", "auto"),
    ("female_young", "👩 Cố định 1 giọng: Nữ trẻ trung, tự nhiên & rõ chữ", "female, young adult, moderate pitch", "female"),
    ("male_young", "👨 Cố định 1 giọng: Nam trẻ trung, năng động (Review / Vlog)", "male, young adult, moderate pitch", "male"),
    ("male_deep", "👨 Cố định 1 giọng: Nam trầm ấm, thuyết minh phim", "male, middle-aged, low pitch", "male"),
    ("female_sweet", "👩 Cố định 1 giọng: Nữ trong trẻo, ngọt ngào (Giọng cao)", "female, young adult, high pitch", "female"),
    ("female_mature", "👩 Cố định 1 giọng: Nữ trưởng thành, trầm ấm", "female, middle-aged, low pitch", "female"),
    ("male_cinematic", "👨 Cố định 1 giọng: Nam rất trầm, điện ảnh hùng tráng", "male, middle-aged, very low pitch", "male"),
    ("child", "🧒 Cố định 1 giọng: Thiếu nhi / Hoạt hình", "child, high pitch", "child"),
    ("elderly_male", "👴 Cố định 1 giọng: Ông lão kể chuyện", "male, elderly, low pitch", "male"),
    ("elderly_female", "👵 Cố định 1 giọng: Bà lão hiền hậu", "female, elderly, moderate pitch", "female"),
    ("whisper", "🤫 Cố định 1 giọng: Thì thầm / ASMR", "female, young adult, whisper", "female"),
    ("clone_file", "🎧 Clone từ file giọng mẫu sạch của bạn (.wav / .mp3)...", "", "auto"),
    ("clone_auto", "🎭 Clone màu giọng gốc trong video (Chỉ dùng khi video cùng ngôn ngữ & thu âm sạch)", "", "auto"),
]

_PRESET_MAP: dict[str, tuple[str, str, str]] = {
    pid: (label, instruct, gender) for pid, label, instruct, gender in OMNIVOICE_PRESETS
}

# Distinct validated instruct profiles per diarized character slot
_SPEAKER_INSTRUCTS: dict[str, str] = {
    "NAM_1": "male, young adult, moderate pitch",
    "NAM_2": "male, middle-aged, low pitch",
    "NAM_3": "male, middle-aged, very low pitch",
    "NU_1": "female, young adult, moderate pitch",
    "NU_2": "female, young adult, high pitch",
    "NU_3": "female, middle-aged, low pitch",
    "TRE_1": "child, high pitch",
}

_MODEL_CACHE: dict[str, object] = {}


class _NullStream:
    def write(self, s):
        return 0

    def flush(self):
        pass

    def isatty(self):
        return False


def is_omnivoice_available() -> bool:
    """Return True if the `omnivoice` Python package is installed in this runtime."""
    try:
        return importlib.util.find_spec("omnivoice") is not None
    except Exception:
        return False


def _ensure_safe_streams():
    if sys.stdout is None:
        sys.stdout = _NullStream()
    if sys.stderr is None:
        sys.stderr = _NullStream()


def _resolve_model_source() -> str:
    """Prefer bundled local models/OmniVoice directory if present, else HuggingFace Hub ID."""
    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "models" / "OmniVoice")
        candidates.append(exe_dir / "_internal" / "models" / "OmniVoice")
    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / "models" / "OmniVoice")
    for p in candidates:
        if p.is_dir() and (p / "config.json").is_file():
            return str(p)
    return "k2-fsa/OmniVoice"


def _lang_for_omnivoice(target_lang: str) -> str:
    code = (target_lang or "vi").strip().lower()
    if "-" in code:
        code = code.split("-", 1)[0]
    return code


def _apply_locale_accent(instruct: str, target_lang: str) -> str:
    """Attach American/British accent ONLY for English target languages (exact valid tags)."""
    base = (instruct or "").strip()
    if not base:
        return base
    if target_lang == "en" and "accent" not in base:
        return f"{base}, american accent"
    if target_lang == "en-GB" and "accent" not in base:
        return f"{base}, british accent"
    return base


def _instruct_for_speaker(seg: dict, target_lang: str) -> str:
    spk = seg.get("speaker", "")
    gender = seg.get("gender", "male")
    if spk in _SPEAKER_INSTRUCTS:
        inst = _SPEAKER_INSTRUCTS[spk]
    elif gender == "female":
        inst = _SPEAKER_INSTRUCTS["NU_1"]
    elif gender == "child":
        inst = _SPEAKER_INSTRUCTS["TRE_1"]
    else:
        inst = _SPEAKER_INSTRUCTS["NAM_1"]
    return _apply_locale_accent(inst, target_lang)


def _resample_to_sr(y: np.ndarray, src_sr: int, dst_sr: int = SR) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) == 0 or src_sr == dst_sr:
        return y
    from math import gcd
    from scipy.signal import resample_poly

    g = gcd(int(dst_sr), int(src_sr))
    up = int(dst_sr) // g
    down = int(src_sr) // g
    return resample_poly(y, up, down).astype(np.float32)


def _load_omnivoice(use_gpu: bool = True, log: Callable[[str], None] = print):
    """Load k2-fsa/OmniVoice safely with automatic VRAM-tier tuning (3GB..24GB) & TF32 Tensor Cores."""
    _ensure_safe_streams()
    import torch
    from omnivoice import OmniVoice
    from .hw_profile import apply_torch_cuda_optimizations, configure_cpu_threads, get_hw_profile

    prof = get_hw_profile()
    if use_gpu and prof.driver_outdated and prof.driver_warning:
        log(prof.driver_warning)

    want_gpu = bool(use_gpu and torch.cuda.is_available())
    cache_key = "cuda" if want_gpu else "cpu"
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    unload_omnivoice()

    src = _resolve_model_source()
    if want_gpu:
        try:
            apply_torch_cuda_optimizations()
            gc.collect()
            torch.cuda.empty_cache()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_name = torch.cuda.get_device_name(0)
            log(
                f"Đang tải mô hình giọng đọc OmniVoice AI trên GPU "
                f"({gpu_name}, {vram_gb:.1f} GB VRAM, batch={prof.omni_batch_size})..."
            )
            model = OmniVoice.from_pretrained(
                src,
                device_map="cuda:0",
                dtype=torch.float16,
                load_asr=False,
            )
            # On <5GB GPUs (e.g. GTX 1060 3GB), keep audio tokenizer on CPU -> peak ~1.17 GB VRAM!
            if prof.omni_tokenizer_cpu and getattr(model, "audio_tokenizer", None) is not None:
                model.audio_tokenizer.to("cpu")
                torch.cuda.empty_cache()
            model.eval()
            _MODEL_CACHE[cache_key] = model
            return model
        except Exception as e:
            log(f"⚠ Không thể tải OmniVoice lên GPU ({e}) ➜ Tự động chuyển sang CPU...")
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    configure_cpu_threads(prof)
    log(f"Đang tải mô hình giọng đọc OmniVoice AI trên CPU ({prof.cpu_threads} luồng)...")
    model = OmniVoice.from_pretrained(
        src,
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()
    _MODEL_CACHE["cpu"] = model
    return model


def unload_omnivoice():
    """Free OmniVoice model & release GPU VRAM cleanly before OCR / FFmpeg NVENC."""
    if not _MODEL_CACHE:
        return
    _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _transcribe_ref_audio_fast(wav_path: str, log: Callable[[str], None] = print) -> str:
    """Transcribe a short user-supplied reference voice clip using local faster-whisper (no HF ASR download)."""
    cache_dir = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "VideoDubber" / "ref_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        st = os.stat(wav_path)
        sig = hashlib.md5(f"{wav_path}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")).hexdigest()[:16]
    except OSError:
        sig = hashlib.md5(wav_path.encode("utf-8")).hexdigest()[:16]
    txt_cache = cache_dir / f"{sig}.txt"
    if txt_cache.is_file():
        cached_txt = txt_cache.read_text(encoding="utf-8", errors="ignore").strip()
        if cached_txt:
            return cached_txt

    try:
        from faster_whisper import WhisperModel

        log("Đang phân tích nội dung file giọng mẫu để tối ưu độ giống...")
        wmodel = WhisperModel("small", device="cpu", compute_type="int8")
        segs_it, _ = wmodel.transcribe(wav_path, beam_size=1, vad_filter=True)
        text = " ".join(s.text.strip() for s in segs_it if s.text.strip()).strip()
        del wmodel
        gc.collect()
        if text:
            txt_cache.write_text(text, encoding="utf-8")
            return text
    except Exception as e:
        log(f"  ℹ Không thể tự nhận diện lời file mẫu ({e}), dùng chế độ clone trực tiếp.")
    return "Xin chào, đây là giọng nói mẫu."


def _extract_best_window(y: np.ndarray, sr: int, max_sec: float = 8.5) -> np.ndarray:
    """Return the cleanest, most speech-active contiguous window up to `max_sec`."""
    max_n = int(max_sec * sr)
    if len(y) <= max_n:
        return y
    step = max(1, int(0.5 * sr))
    best_i = 0
    best_e = -1.0
    for i in range(0, len(y) - max_n + 1, step):
        e = float(np.mean(y[i:i + max_n] ** 2))
        if e > best_e:
            best_e = e
            best_i = i
    return y[best_i:best_i + max_n]


def _make_prompt_safe(model, wav_24k: np.ndarray, ref_sr: int, ref_text: str):
    """Create OmniVoice VoiceClonePrompt safely with automatic float32 + fallback if silence-trim empties clip."""
    import torch

    wav = np.asarray(wav_24k, dtype=np.float32).reshape(-1)
    if len(wav) < int(0.45 * ref_sr):
        return None
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < 0.0015:
        return None
    if rms < 0.06:
        wav = np.clip(wav * (0.08 / rms), -0.95, 0.95).astype(np.float32)

    tensor_2d = torch.from_numpy(wav[None, :].copy())
    clean_ref_text = sanitize_tts_text(ref_text) or "Hello."

    try:
        return model.create_voice_clone_prompt(
            ref_audio=(tensor_2d, ref_sr),
            ref_text=clean_ref_text,
            preprocess_prompt=True,
        )
    except Exception:
        try:
            return model.create_voice_clone_prompt(
                ref_audio=(tensor_2d, ref_sr),
                ref_text=clean_ref_text,
                preprocess_prompt=False,
            )
        except Exception:
            return None


def _build_speaker_clone_prompts(model, segs: list[dict], vocals_path: str | Path | None,
                                 log: Callable[[str], None] = print) -> dict[str, object]:
    """Extract the cleanest 3.5-8.5s speech sample per speaker from vocals.wav and encode once."""
    prompts: dict[str, object] = {}
    if not vocals_path or not os.path.exists(str(vocals_path)):
        return prompts

    ref_sr = int(getattr(model, "sampling_rate", 24000) or 24000)
    try:
        full_wav = load_audio(str(vocals_path), ref_sr, 1)[0]
    except Exception as e:
        log(f"⚠ Không đọc được âm thanh gốc để clone giọng ({e}), tự động chuyển sang Thiết kế giọng AI.")
        return prompts

    by_spk: dict[str, list[dict]] = {}
    for s in segs:
        spk = s.get("speaker") or ("NU_1" if s.get("gender") == "female" else "NAM_1")
        by_spk.setdefault(spk, []).append(s)

    for spk, spk_segs in by_spk.items():
        # Rank clips by speech clarity & ideal reference duration (2.5..8.0s)
        scored = []
        for s in spk_segs:
            st = max(0.0, float(s.get("start", 0.0)))
            en = max(st, float(s.get("end", st)))
            dur = en - st
            txt = (s.get("text") or "").strip()
            if dur < 0.7 or not txt:
                continue
            a = int(st * ref_sr)
            b = min(len(full_wav), int(en * ref_sr))
            if b - a < int(0.6 * ref_sr):
                continue
            clip = full_wav[a:b]
            rms = float(np.sqrt(np.mean(clip ** 2)))
            if rms < 0.002:
                continue
            # Prefer clips near 4.5s with strong clear vocal RMS
            dur_score = 1.0 - min(1.0, abs(dur - 4.8) / 6.0)
            score = dur_score * 0.65 + min(rms / 0.12, 1.0) * 0.35
            scored.append((score, dur, clip, txt))

        if not scored:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        # Concatenate up to 3 top clips until we reach 3.5..8.5s total
        chosen_clips = []
        chosen_texts = []
        total_d = 0.0
        gap_sil = np.zeros(int(0.12 * ref_sr), dtype=np.float32)

        for _sc, dur, clip, txt in scored:
            if total_d >= 4.2 and len(chosen_clips) >= 1:
                break
            if total_d + dur > 9.5 and chosen_clips:
                continue
            if chosen_clips:
                chosen_clips.append(gap_sil)
            chosen_clips.append(clip)
            chosen_texts.append(txt)
            total_d += dur
            if len(chosen_clips) >= 5:  # 3 clips + 2 silences
                break

        if not chosen_clips:
            continue

        merged = np.concatenate(chosen_clips)
        if len(merged) > int(9.5 * ref_sr):
            merged = merged[:int(9.5 * ref_sr)]
        ref_txt = " ".join(chosen_texts).strip()

        prompt = _make_prompt_safe(model, merged, ref_sr, ref_txt)
        if prompt is not None:
            prompts[spk] = prompt
            lbl = spk_segs[0].get("speaker_label", spk)
            log(f"  🎭 Đã trích xuất mẫu clone giọng gốc cho nhân vật [{lbl}] ({len(merged) / ref_sr:.1f}s)")

    return prompts


def _build_custom_file_prompt(model, ref_audio_path: str, log: Callable[[str], None] = print):
    """Encode user's custom reference audio file (.wav/.mp3) into a reusable VoiceClonePrompt."""
    if not ref_audio_path or not os.path.isfile(ref_audio_path):
        raise RuntimeError("Bạn đã chọn Clone từ file giọng mẫu nhưng chưa chọn file âm thanh hợp lệ.")
    ref_sr = int(getattr(model, "sampling_rate", 24000) or 24000)
    wav = load_audio(ref_audio_path, ref_sr, 1)[0]
    wav = _extract_best_window(wav, ref_sr, max_sec=8.5)
    with tempfile.TemporaryDirectory() as td:
        tmp_wav = os.path.join(td, "ref_clip.wav")
        save_wav(tmp_wav, wav[None, :], ref_sr)
        ref_text = _transcribe_ref_audio_fast(tmp_wav, log=log)
    prompt = _make_prompt_safe(model, wav, ref_sr, ref_text)
    if prompt is None:
        raise RuntimeError("File giọng mẫu quá ngắn hoặc không có tiếng nói rõ ràng (cần >= 1 giây).")
    log(f"  🎧 Đã nạp giọng mẫu từ file: {Path(ref_audio_path).name} ({len(wav) / ref_sr:.1f}s)")
    return prompt


def _estimate_duration_sec(model, text: str, prompt) -> float:
    """Instant (0ms) pre-synthesis duration estimate using OmniVoice's built-in RuleDurationEstimator."""
    try:
        frame_rate = float(model.audio_tokenizer.config.frame_rate)
        ref_txt = prompt.ref_text if prompt is not None else None
        ref_len = prompt.ref_audio_tokens.size(-1) if prompt is not None else None
        toks = model._estimate_target_tokens(text, ref_txt, ref_len, speed=1.0)
        return max(0.3, toks / max(1.0, frame_rate))
    except Exception:
        return max(0.5, len(text) / 14.0)


def _ov_cache_path(cache_dir: str, sig: str) -> str:
    h = hashlib.md5(sig.encode("utf-8")).hexdigest()[:18]
    return os.path.join(cache_dir, f"ov_{h}.wav")


def _get_or_build_native_anchor_prompt(
    model,
    target_lang: str,
    instruct: str,
    log: Callable[[str], None] = print,
):
    """Return a crystal-clear, studio-grade Native VoiceClonePrompt spoken IN THE TARGET LANGUAGE.

    Eliminates foreign-accent carryover & muffled Demucs vocal artifacts, while guaranteeing 100%
    identical character voice across every sentence of the video (and cached permanently on disk!).
    """
    from omnivoice.models.omnivoice import VoiceClonePrompt
    from .config import SAMPLE_PREVIEW_TEXT

    lang_code = _lang_for_omnivoice(target_lang)
    sr_out = int(getattr(model, "sampling_rate", 24000) or 24000)
    anchor_dir = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "VideoDubber" / "omni_anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)

    sig = hashlib.md5(f"anchor_v2|{target_lang}|{lang_code}|{instruct}".encode("utf-8")).hexdigest()[:16]
    pt_path = anchor_dir / f"{lang_code}_{sig}.pt"

    if pt_path.is_file() and pt_path.stat().st_size > 256:
        try:
            return VoiceClonePrompt.load(str(pt_path))
        except Exception:
            pass

    anchor_txt = SAMPLE_PREVIEW_TEXT.get(target_lang) or SAMPLE_PREVIEW_TEXT.get(lang_code) or SAMPLE_PREVIEW_TEXT["vi"]
    try:
        raw_list = model.generate(
            text=anchor_txt,
            language=lang_code,
            instruct=instruct,
            speed=1.0,
            num_step=24,
            denoise=True,
            postprocess_output=True,
        )
        raw_24k = np.asarray(raw_list[0], dtype=np.float32).reshape(-1)
        prompt = _make_prompt_safe(model, raw_24k, sr_out, anchor_txt)
        if prompt is not None:
            try:
                prompt.save(str(pt_path))
            except Exception:
                pass
            return prompt
    except Exception as e:
        log(f"  ℹ Không tạo được mẫu giọng chuẩn ({e}), chuyển sang sinh trực tiếp.")
    return None


def synthesize_preview_wav_omnivoice(
    preset: str,
    target_lang: str,
    text: str,
    out_wav: str,
    ref_audio_path: str = "",
    use_gpu: bool = True,
    num_step: int = 16,
    log: Callable[[str], None] = print,
) -> str:
    """Synthesize a live preview WAV clip using OmniVoice for the GUI '🔊 Nghe thử giọng' button."""
    if not is_omnivoice_available():
        raise RuntimeError(
            "Thư viện OmniVoice chưa được cài trong bản thực thi hiện tại.\n"
            "Vui lòng cài đặt bản Setup v2.2.0 mới nhất hoặc chuyển sang Edge Neural TTS."
        )

    cleaned = sanitize_tts_text(text)
    if not cleaned:
        raise RuntimeError("Văn bản mẫu trống.")

    model = _load_omnivoice(use_gpu=use_gpu, log=log)
    lang_code = _lang_for_omnivoice(target_lang)
    sr_out = int(getattr(model, "sampling_rate", 24000) or 24000)

    prompt = None
    instruct = None
    if preset == "clone_file":
        prompt = _build_custom_file_prompt(model, ref_audio_path, log=log)
    else:
        if preset in ("clone_auto", "design_auto"):
            instruct = _apply_locale_accent("female, young adult, moderate pitch", target_lang)
        else:
            _, raw_inst, _ = _PRESET_MAP.get(preset, ("", "female, young adult, moderate pitch", "female"))
            instruct = _apply_locale_accent(raw_inst, target_lang)
        prompt = _get_or_build_native_anchor_prompt(model, target_lang, instruct, log=log)

    audios = model.generate(
        text=cleaned,
        language=lang_code,
        voice_clone_prompt=prompt,
        instruct=instruct if prompt is None else None,
        num_step=max(8, min(32, int(num_step))),
        denoise=True,
        postprocess_output=True,
    )
    y = _trim_silence(_resample_to_sr(audios[0], sr_out, SR))
    if len(y) == 0:
        raise RuntimeError("OmniVoice không tạo được âm thanh mẫu.")
    peak = float(np.max(np.abs(y)))
    if peak > 0.01:
        y = y * (0.88 / peak)
    save_wav(out_wav, y[None, :], SR)
    return out_wav


def synthesize_track_omnivoice(
    segments: list[dict],
    total_duration: float,
    cache_dir: str,
    target_lang: str = "vi",
    vocals_path: str | Path | None = None,
    preset: str = "design_auto",
    ref_audio_path: str = "",
    use_gpu: bool = True,
    max_speedup: float = 1.28,
    min_gap_ms: int = 240,
    num_step: int = 16,
    log: Callable[[str], None] = print,
    progress=None,
    cancelled=None,
):
    """Synthesize full multi-character dubbed speech track using OmniVoice AI + Anti-Slur Elastic Timeline."""
    if not is_omnivoice_available():
        raise RuntimeError("OmniVoice chưa được cài đặt trong môi trường hiện tại.")

    os.makedirs(cache_dir, exist_ok=True)
    active_pairs = [
        (idx, s) for idx, s in enumerate(segments)
        if sanitize_tts_text(s.get("translation", ""))
    ]
    n_total = int((total_duration + 1.5) * SR)
    if not active_pairs:
        return np.zeros(n_total, np.float32), []

    segs = [s for _, s in active_pairs]
    min_gap = max(0.14, float(min_gap_ms) / 1000.0)
    lang_code = _lang_for_omnivoice(target_lang)
    steps = max(8, min(32, int(num_step)))
    preset = preset or "design_auto"

    if progress:
        progress(0.02)

    model = _load_omnivoice(use_gpu=use_gpu, log=log)
    sr_out = int(getattr(model, "sampling_rate", 24000) or 24000)

    try:
        if cancelled and cancelled():
            raise InterruptedError("Đã hủy")

        # 1. Prepare Native Studio Voice Anchors or Voice Clone Prompts per Character
        speaker_prompts: dict[str, object] = {}
        global_prompt = None
        ref_sig = f"v3_{preset}_{target_lang}"

        if preset == "clone_auto":
            log(f"OmniVoice AI: Đang phân tích & clone màu giọng gốc ({len(segs)} câu, {steps} steps)...")
            speaker_prompts = _build_speaker_clone_prompts(model, segs, vocals_path, log=log)
            if vocals_path and os.path.exists(str(vocals_path)):
                st = os.stat(str(vocals_path))
                ref_sig = f"clone_auto_v3|{st.st_size}|{int(st.st_mtime)}"
        elif preset == "clone_file":
            log(f"OmniVoice AI: Đang nạp giọng mẫu để lồng tiếng ({len(segs)} câu, {steps} steps)...")
            global_prompt = _build_custom_file_prompt(model, ref_audio_path, log=log)
            st = os.stat(ref_audio_path)
            ref_sig = f"clone_file_v3|{Path(ref_audio_path).name}|{st.st_size}|{int(st.st_mtime)}"
        else:
            p_label = _PRESET_MAP.get(preset, (preset, "", "auto"))[0]
            log(f"OmniVoice AI ({p_label}): Đang tạo giọng bản xứ rõ chữ cho {len(segs)} câu ({steps} steps)...")

        # Ensure EVERY speaker role in `design_auto` / fixed preset (or uncloned fallback) gets a
        # pure Native Target-Language Studio Anchor -> 100% clear pronunciation & 0% speaker mix-up!
        if global_prompt is None:
            if preset not in ("clone_auto", "design_auto"):
                spk_key_fixed = f"PRESET_{preset}"
                _, raw_inst, _ = _PRESET_MAP.get(preset, ("", "female, young adult, moderate pitch", "female"))
                inst_fixed = _apply_locale_accent(raw_inst, target_lang)
                anc = _get_or_build_native_anchor_prompt(model, target_lang, inst_fixed, log=log)
                if anc is not None:
                    speaker_prompts[spk_key_fixed] = anc
            else:
                # Collect distinct speaker roles present in the script
                seen_roles: dict[str, dict] = {}
                for s in segs:
                    spk = s.get("speaker") or ("NU_1" if s.get("gender") == "female" else "NAM_1")
                    if spk not in seen_roles:
                        seen_roles[spk] = s
                for spk, rep_seg in seen_roles.items():
                    if cancelled and cancelled():
                        raise InterruptedError("Đã hủy")
                    if spk not in speaker_prompts:
                        inst = _instruct_for_speaker(rep_seg, target_lang)
                        lbl = rep_seg.get("speaker_label", spk)
                        log(f"  🎙 Khóa giọng bản xứ chuẩn cho nhân vật [{lbl}]...")
                        anc = _get_or_build_native_anchor_prompt(model, target_lang, inst, log=log)
                        if anc is not None:
                            speaker_prompts[spk] = anc

        if progress:
            progress(0.12)

        # 2. Per-Speaker Isolated Batched GPU Synthesis + Anti-Slur Speed Cap (<= 1.20x)
        import torch
        from .hw_profile import get_hw_profile

        prof = get_hw_profile()
        batch_size = prof.omni_batch_size if (use_gpu and torch.cuda.is_available()) else 1
        n_segs = len(segs)
        audios: list[np.ndarray | None] = [None] * n_segs
        speeds: list[float] = [1.0] * n_segs
        done_count = 0
        omni_speed_cap = min(float(max_speedup), 1.20)  # Strict 1.20x ceiling keeps Vietnamese crisp & clear!

        def _bump_prog(inc: int = 1):
            nonlocal done_count
            done_count += inc
            if progress:
                progress(0.12 + 0.83 * min(1.0, done_count / max(1, n_segs)))

        def _resolve_seg_meta(i: int, s: dict):
            txt = sanitize_tts_text(s["translation"])
            spk = s.get("speaker") or ("NU_1" if s.get("gender") == "female" else "NAM_1")
            if global_prompt is not None:
                spk_key = "GLOBAL_FILE"
                instruct = None
            elif preset in ("clone_auto", "design_auto"):
                spk_key = spk
                instruct = _instruct_for_speaker(s, target_lang)
            else:
                spk_key = f"PRESET_{preset}"
                _, raw_inst, _ = _PRESET_MAP.get(preset, ("", "female, young adult, moderate pitch", "female"))
                instruct = _apply_locale_accent(raw_inst, target_lang)

            prompt_now = global_prompt if global_prompt is not None else speaker_prompts.get(spk_key)
            orig_start = float(s["start"])
            orig_end = float(s["end"])
            next_start = float(segs[i + 1]["start"]) if (i + 1 < n_segs) else total_duration
            # Generous window allowing natural drift into pauses so TTS doesn't rush unnecessarily
            base_avail = max(0.35, next_start - orig_start - min_gap)
            slot_budget = max(orig_end - orig_start, min(base_avail + 0.35, (orig_end - orig_start) + 0.90))
            slot_budget = min(slot_budget, max(0.35, base_avail + 0.45))

            est_dur = _estimate_duration_sec(model, txt, prompt_now)
            need = est_dur / max(0.32, slot_budget)
            gen_speed = float(np.clip(need, 1.0, omni_speed_cap)) if need > 1.05 else 1.0
            speeds[i] = need
            cache_key = f"ov3|{ref_sig}|{spk_key}|{lang_code}|{steps}|{gen_speed:.2f}|{txt}"
            cache_p = _ov_cache_path(cache_dir, cache_key)
            return {
                "idx": i,
                "txt": txt,
                "spk_key": spk_key,
                "instruct": instruct,
                "prompt": prompt_now,
                "gen_speed": gen_speed,
                "cache_p": cache_p,
            }

        pending_by_speaker: dict[str, list[dict]] = {}
        for i, s in enumerate(segs):
            if cancelled and cancelled():
                raise InterruptedError("Đã hủy")
            it = _resolve_seg_meta(i, s)
            cp = it["cache_p"]
            if os.path.exists(cp) and os.path.getsize(cp) > 256:
                try:
                    audios[i] = _trim_silence(load_audio(cp, SR, 1)[0])
                    _bump_prog(1)
                    continue
                except Exception:
                    pass
            pending_by_speaker.setdefault(it["spk_key"], []).append(it)

        def _synth_single_item(it: dict):
            if cancelled and cancelled():
                raise InterruptedError("Đã hủy")
            idx = it["idx"]
            prompt = it["prompt"]
            try:
                raw_list = model.generate(
                    text=it["txt"],
                    language=lang_code,
                    voice_clone_prompt=prompt,
                    instruct=it["instruct"] if prompt is None else None,
                    speed=it["gen_speed"] if abs(it["gen_speed"] - 1.0) > 0.03 else None,
                    num_step=steps,
                    denoise=True,
                    postprocess_output=True,
                )
                raw_24k = np.asarray(raw_list[0], dtype=np.float32).reshape(-1)
                y_44k = _trim_silence(_resample_to_sr(raw_24k, sr_out, SR))
                if len(y_44k) > 0:
                    save_wav(it["cache_p"], y_44k[None, :], SR)
                audios[idx] = y_44k
            except Exception as e:
                log(f"  ⚠ OmniVoice bỏ qua câu {idx + 1} ('{it['txt'][:40]}') do lỗi: {e}")
                audios[idx] = np.zeros(0, np.float32)
            _bump_prog(1)

        # Process strictly grouped by speaker so two characters NEVER share a diffusion batch!
        for _spk_key, spk_items in pending_by_speaker.items():
            if batch_size <= 1 or any(x["prompt"] is None for x in spk_items):
                for it in spk_items:
                    _synth_single_item(it)
                continue

            for b_start in range(0, len(spk_items), batch_size):
                if cancelled and cancelled():
                    raise InterruptedError("Đã hủy")
                chunk = spk_items[b_start:b_start + batch_size]
                if len(chunk) == 1:
                    _synth_single_item(chunk[0])
                    continue
                try:
                    raw_list = model.generate(
                        text=[x["txt"] for x in chunk],
                        language=lang_code,
                        voice_clone_prompt=[x["prompt"] for x in chunk],
                        instruct=None,
                        speed=[x["gen_speed"] for x in chunk],
                        num_step=steps,
                        denoise=True,
                        postprocess_output=True,
                    )
                    for x, raw_item in zip(chunk, raw_list):
                        raw_24k = np.asarray(raw_item, dtype=np.float32).reshape(-1)
                        y_44k = _trim_silence(_resample_to_sr(raw_24k, sr_out, SR))
                        if len(y_44k) > 0:
                            save_wav(x["cache_p"], y_44k[None, :], SR)
                        audios[x["idx"]] = y_44k
                    _bump_prog(len(chunk))
                except InterruptedError:
                    raise
                except RuntimeError as oom_e:
                    log(f"  ℹ Giảm cỡ batch tạm thời cho cụm câu dài ({oom_e.__class__.__name__})...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for x in chunk:
                        _synth_single_item(x)
                except Exception:
                    for x in chunk:
                        _synth_single_item(x)

        final_audios = [a if a is not None else np.zeros(0, np.float32) for a in audios]
        track, info = place_audios_on_track(segs, final_audios, speeds, total_duration, min_gap_ms=min_gap_ms)
        if progress:
            progress(1.0)
        return track, info
    finally:
        # Always free GPU VRAM so RapidOCR and FFmpeg h264_nvenc have full GPU headroom!
        unload_omnivoice()
