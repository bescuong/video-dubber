"""End-to-end dubbing pipeline.

Stage 1  analyze(): extract audio -> separate vocals -> speech recognition -> translation -> speaker/character analysis
Stage 2  render():  TTS fitted to timing -> mix with background -> OCR old subs -> erase/cover + new subs -> mp4
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .config import LANGUAGES, Settings
from . import media
from .media import CREATE_NO_WINDOW

SR = 44100

# Canonical stage identifiers (used by GUI stepper & weighted overall progress tracker)
STAGE_SEPARATE = "Tách âm thanh"
STAGE_ASR = "Nhận dạng giọng nói"
STAGE_TRANSLATE = "Dịch thuật"
STAGE_DIARIZE = "Phân tích giọng & nhân vật"
STAGE_TTS = "Lồng tiếng (TTS)"
STAGE_MIX = "Trộn âm thanh"
STAGE_OCR = "Quét phụ đề cũ (OCR)"
STAGE_RENDER = "Xuất video"


class Callbacks:
    def __init__(self, log=None, progress=None, cancelled=None):
        self.log = log or (lambda m: print(m, flush=True))
        self._progress = progress or (lambda stage, frac: None)
        self.cancelled = cancelled or (lambda: False)

    def stage(self, name: str):
        self.log(f"▶ {name}")
        self._progress(name, 0.0)
        return lambda f: self._progress(name, float(np.clip(f, 0.0, 1.0)))

    def check(self):
        if self.cancelled():
            raise InterruptedError("Đã hủy")


def work_dir(video):
    v = Path(video).resolve()
    d = v.parent / f"{v.stem}_dub"
    d.mkdir(exist_ok=True)
    return d


def load_project(video):
    p = work_dir(video) / "project.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_project(video, settings, segments, src_lang):
    p = work_dir(video) / "project.json"
    p.write_text(
        json.dumps(
            {"settings": settings.to_dict(), "source_lang": src_lang, "segments": segments},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def _match_lang_code(code: str, fallback: str = "en") -> str:
    if not code:
        return fallback
    if code in LANGUAGES:
        return code
    base = code.split("-")[0]
    if base in LANGUAGES:
        return base
    return fallback


# ============================================================================ STAGE 1: ANALYZE
def analyze(video, settings: Settings, cb: Callbacks = None):
    cb = cb or Callbacks()
    t0 = time.time()
    stage_times: list[str] = []
    video = str(Path(video).resolve())
    wd = work_dir(video)
    info = media.probe(video)
    mode = getattr(settings, "process_mode", "dub_audio")

    if mode == "dub_audio" and not info["has_audio"]:
        cb.log("ℹ Video không có kênh âm thanh — tự động chuyển sang chế độ Quét chữ phụ đề trên video (OCR).")
        mode = "ocr_sub_only"

    cb.log(f"Video {info['width']}x{info['height']} @ {info['fps']:.3f} fps, {info['duration']:.1f}s")

    if mode in ("ocr_sub_only", "ocr_sub_dub"):
        # OCR-First Branch: scan burned-in text directly from movable ROI (works on silent or BGM-only videos!)
        ts = time.time()
        p = cb.stage(STAGE_OCR)
        from . import subtitles
        acc_key = ""
        if getattr(settings, "translation_engine", "gemini") == "gemini":
            try:
                from . import auth
                acc_key = auth.get_account_gemini_key()
            except Exception:
                acc_key = ""

        segs, samples, detected = subtitles.extract_ocr_subtitles(
            video,
            info,
            settings,
            gemini_api_key=acc_key,
            log=cb.log,
            progress=p,
            cancelled=cb.cancelled,
        )
        # Cache OCR detection boxes so Stage 2 (render) erases old subs immediately without rescanning
        try:
            cache = wd / "ocr_subs.json"
            cache.write_text(
                json.dumps({"key": subtitles.roi_cache_key(settings), "samples": samples}),
                encoding="utf-8",
            )
        except Exception:
            pass
        stage_times.append(f"Quét OCR {time.time() - ts:.1f}s")
        p(1.0)
        cb.check()
        if not segs:
            raise RuntimeError(
                "Không quét thấy văn bản phụ đề nào trong khung chọn!\n"
                "👉 Hãy kéo khung màu xanh trên màn hình xem trước đặt đúng vào vùng chứa phụ đề gốc rồi thử lại."
            )
    else:
        # 1. Audio extraction & Demucs separation (cached)
        ts = time.time()
        p = cb.stage(STAGE_SEPARATE)
        vocals_p, bg_p = wd / "vocals.wav", wd / "background.wav"
        if settings.separate_vocals:
            if not (vocals_p.exists() and bg_p.exists()):
                from .separate import separate
                audio = media.load_audio(video, SR, 2)
                p(0.05)
                cb.check()
                vocals, bg = separate(
                    audio, log=cb.log, use_gpu=settings.use_gpu,
                    progress=lambda f: p(0.05 + 0.92 * f), cancelled=cb.cancelled
                )
                media.save_wav(vocals_p, vocals, SR)
                media.save_wav(bg_p, bg, SR)
            else:
                cb.log("Dùng lại kết quả tách giọng & nhạc nền đã lưu")
            asr_src = vocals_p
        else:
            asr_src = video
        stage_times.append(f"Tách nhạc {time.time() - ts:.1f}s")
        p(1.0)
        cb.check()

        # 2. Speech recognition (Whisper)
        ts = time.time()
        p = cb.stage(STAGE_ASR)
        wav16 = wd / "asr16k.wav"
        media.run(["ffmpeg", "-v", "error", "-y", "-i", str(asr_src), "-ac", "1", "-ar", "16000", str(wav16)])
        from .transcribe import transcribe
        lang = None if settings.source_lang == "auto" else LANGUAGES.get(settings.source_lang, LANGUAGES["en"]).whisper_code
        segs, detected = transcribe(
            wav16, settings.whisper_model, lang, settings.use_gpu, cb.log, p, info["duration"]
        )
        cb.log(f"Nhận dạng được {len(segs)} câu thoại")
        stage_times.append(f"Whisper {time.time() - ts:.1f}s")
        cb.check()
        if not segs:
            raise RuntimeError(
                "Không nhận dạng được lời thoại nào trong video.\n"
                "👉 Nếu video chỉ có chữ phụ đề hiển thị (không có tiếng nói), hãy đổi 'Chế độ xử lý' "
                "sang '📝 Chỉ dịch phụ đề chữ trên video (OCR)'."
            )

        # 3. Speaker Gender / Age / Multi-Character Diarization (runs BEFORE translation so AI knows speaker roles!)
        ts = time.time()
        p = cb.stage(STAGE_DIARIZE)
        try:
            from .gender import analyze_speakers
            y16 = media.load_audio(wav16, 16000, 1)[0]
            p(0.3)
            spk_infos = analyze_speakers(y16, segs)
            unique_labels = sorted({x["speaker_label"] for x in spk_infos})
            cb.log(f"Phân tích giọng đọc: phát hiện {len(unique_labels)} nhóm nhân vật ({', '.join(unique_labels)})")
            for s, sp in zip(segs, spk_infos):
                s["gender"] = sp["gender"]
                s["speaker"] = sp["speaker"]
                s["speaker_label"] = sp["speaker_label"]
        except Exception as e:
            cb.log(f"Không phân tích chi tiết được nhân vật ({e}), dùng mặc định Nam")
            for s in segs:
                s["gender"] = "male"
                s["speaker"] = "NAM_1"
                s["speaker_label"] = "Nam 1"
        stage_times.append(f"Phân vai {time.time() - ts:.1f}s")
        p(1.0)
        cb.check()

    # 4. Script Translation (Gemini AI High->Low Cascade or Offline NLLB-200)
    ts = time.time()
    p = cb.stage(STAGE_TRANSLATE)
    src_code = _match_lang_code(detected, fallback=(settings.source_lang if settings.source_lang != "auto" else "en"))
    if detected not in LANGUAGES and src_code != detected:
        cb.log(f"Ngôn ngữ '{detected}' được ánh xạ về '{src_code}'")
    tgt = LANGUAGES.get(settings.target_lang, LANGUAGES["vi"])
    src_obj = LANGUAGES.get(src_code, LANGUAGES["en"])

    if src_obj.nllb == tgt.nllb and settings.source_lang == settings.target_lang:
        translations = [s["text"] for s in segs]
        p(1.0)
    elif getattr(settings, "translation_engine", "gemini") == "gemini":
        from . import auth, gemini_translate
        acc_key = auth.get_account_gemini_key()
        if not acc_key:
            raise RuntimeError(
                "Bạn đang chọn dịch kịch bản bằng Gemini AI nhưng tài khoản chưa lưu khóa API!\n"
                "👉 Vui lòng bấm nút '🔑 Gemini API' ở góc trên thanh tiêu đề để nhập khóa API "
                "(hoặc đổi sang 'NLLB Offline' trong Tùy chỉnh nâng cao)."
            )
        try:
            translations = gemini_translate.translate_script_with_gemini(
                segments=segs,
                src_lang_name=src_obj.name,
                tgt_lang_name=tgt.name,
                raw_api_keys=acc_key,
                style_code=getattr(settings, "translation_style", "auto_natural"),
                custom_notes=getattr(settings, "translation_notes", ""),
                log=cb.log,
                progress=p,
                cancelled=cb.cancelled,
            )
        except InterruptedError:
            raise
        except Exception as gem_err:
            if getattr(settings, "gemini_fallback_nllb", True) and "không hợp lệ" not in str(gem_err).lower():
                cb.log(f"⚠ Gemini AI hết lượt sử dụng ({gem_err}) ➜ Tự động chuyển sang mô hình dự phòng NLLB-200 Offline...")
                from .translate import translate
                translations = translate(
                    [s["text"] for s in segs], src_obj.nllb, tgt.nllb,
                    settings.nllb_model, settings.use_gpu, cb.log, p
                )
            else:
                raise
    else:
        from .translate import translate
        translations = translate(
            [s["text"] for s in segs], src_obj.nllb, tgt.nllb,
            settings.nllb_model, settings.use_gpu, cb.log, p
        )

    for s, t in zip(segs, translations):
        s["translation"] = t
    stage_times.append(f"Dịch {time.time() - ts:.1f}s")
    cb.check()

    save_project(video, settings, segs, src_code)
    cb.log(f"⏱ Thống kê tốc độ: {' · '.join(stage_times)}")
    cb.log(f"✔ Hoàn tất phân tích sau {time.time() - t0:.0f}s. Bạn có thể kiểm tra/sửa lời dịch trước khi xuất.")
    return segs, src_code


# ============================================================================ STAGE 2: RENDER
def _rms_active(x, thr=0.005):
    m = np.abs(x) > thr
    return float(np.sqrt(np.mean(x[m] ** 2))) if m.sum() > 100 else 0.0


def _mix(dub, wd, info, settings, video, cb):
    n = len(dub)
    vocals_p, bg_p = wd / "vocals.wav", wd / "background.wav"
    if not info.get("has_audio", True):
        bg = np.zeros((2, n), np.float32)
        vocals = np.zeros((2, n), np.float32)
    elif settings.separate_vocals and bg_p.exists() and vocals_p.exists():
        bg = media.load_audio(bg_p, SR, 2)
        vocals = media.load_audio(vocals_p, SR, 2)
    elif getattr(settings, "process_mode", "dub_audio") == "ocr_sub_dub":
        # Keep original background music cleanly when dubbing over OCR text video
        bg = media.load_audio(video, SR, 2) * 0.55
        vocals = np.zeros((2, bg.shape[1]), np.float32)
    else:
        bg = np.zeros((2, n), np.float32)
        vocals = media.load_audio(video, SR, 2)
    m = max(n, bg.shape[1], vocals.shape[1])
    pad = lambda a: np.pad(a, ((0, 0), (0, m - a.shape[1]))) if a.ndim == 2 else np.pad(a, (0, m - len(a)))
    bg, vocals, dub = pad(bg), pad(vocals), pad(dub)
    ref = _rms_active(vocals.mean(0))
    cur = _rms_active(dub)
    gain = float(np.clip(ref / cur, 0.5, 2.8)) if cur > 0 and ref > 0 else 1.0
    out = bg + vocals * settings.original_vocals_volume + dub[None, :] * gain * settings.dub_volume
    peak = float(np.abs(out).max()) if out.size else 0
    if peak > 0.98:
        out *= 0.98 / peak
    dur_n = int(info["duration"] * SR)
    if dur_n > 0:
        out = out[:, :dur_n]
    return out.astype(np.float32)


_CACHED_GPU_ENCODER: str | None = None


def _pick_encoder(use_gpu: bool, W: int = 1920, H: int = 1080) -> str:
    global _CACHED_GPU_ENCODER
    if not use_gpu or W > 4096 or H > 4096:
        return "libx264"
    if _CACHED_GPU_ENCODER is not None:
        return _CACHED_GPU_ENCODER

    # Probe native GPU encoders first, then Windows MediaFoundation Hardware MFT (`h264_mf`),
    # which activates NVIDIA NVENC / AMD AMF / Intel QSV on ALL Windows driver versions!
    candidates = [
        ("h264_nvenc", ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p"]),
        ("h264_amf", ["-c:v", "h264_amf", "-pix_fmt", "yuv420p"]),
        ("h264_qsv", ["-c:v", "h264_qsv", "-pix_fmt", "nv12"]),
        ("h264_mf", ["-c:v", "h264_mf", "-hw_encoding", "true", "-rate_control", "u_vbr", "-b:v", "4000k", "-pix_fmt", "nv12"]),
    ]
    for enc_name, enc_test_args in candidates:
        try:
            media.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=640x360:d=0.2:r=25",
                *enc_test_args, "-f", "mp4", os.devnull, "-y"
            ])
            _CACHED_GPU_ENCODER = enc_name
            return enc_name
        except Exception:
            continue

    _CACHED_GPU_ENCODER = "libx264"
    return "libx264"


def _enc_args(encoder: str, crf: int, W: int = 1920, H: int = 1080, fps: float = 30.0) -> list[str]:
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(crf + 1), "-b:v", "0", "-pix_fmt", "yuv420p"]
    if encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf + 1), "-pix_fmt", "yuv420p"]
    if encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-preset", "faster", "-global_quality", str(crf + 1), "-pix_fmt", "nv12"]
    if encoder == "h264_mf":
        # Windows Hardware MFT (NVIDIA H.264 Encoder MFT / AMD / Intel QSV): scale VBR bitrate cleanly by resolution
        bpp = 0.12 * (2.0 ** ((20.0 - float(crf)) / 6.0))
        kbps = int(np.clip(W * H * min(float(fps or 30.0), 60.0) * bpp / 1000.0, 2500, 28000))
        return ["-c:v", "h264_mf", "-hw_encoding", "true", "-rate_control", "u_vbr", "-b:v", f"{kbps}k", "-pix_fmt", "nv12"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]


def render(video, settings: Settings, segments, cb: Callbacks = None, output=None):
    cb = cb or Callbacks()
    t0 = time.time()
    stage_times: list[str] = []
    video = str(Path(video).resolve())
    wd = work_dir(video)
    info = media.probe(video)
    W, H = info["width"], info["height"]
    tgt = LANGUAGES.get(settings.target_lang, LANGUAGES["vi"])
    mode = getattr(settings, "process_mode", "dub_audio")
    need_tts = mode in ("dub_audio", "ocr_sub_dub")
    from .hw_profile import get_hw_profile
    prof = get_hw_profile()
    save_project(video, settings, segments, (load_project(video) or {}).get("source_lang", "auto"))

    audio_p = None
    if need_tts:
        # 1. TTS with OmniVoice AI (Zero-Shot Clone / Voice Design) or Edge Neural TTS + Elastic Timeline
        ts = time.time()
        p = cb.stage(STAGE_TTS)
        from .gender import resolve_voice_for_segment
        from . import omnivoice_tts
        from .tts import synthesize_track

        for s in segments:
            s["voice"] = resolve_voice_for_segment(s, settings)

        want_omni = getattr(settings, "tts_engine", "omnivoice") == "omnivoice"
        if want_omni and not omnivoice_tts.is_omnivoice_available():
            cb.log(
                "⚠ Phiên bản cài đặt hiện tại chưa tích hợp gói thư viện OmniVoice AI — "
                "tự động chuyển sang Edge Neural TTS (vui lòng chạy file Setup v2.4.0 để dùng OmniVoice)."
            )
            want_omni = False

        if want_omni:
            vocals_p = wd / "vocals.wav"
            if vocals_p.exists():
                ref_src = vocals_p
            elif info["has_audio"] and mode == "dub_audio":
                ref_src = video
            else:
                ref_src = None
            dub, tinfo = omnivoice_tts.synthesize_track_omnivoice(
                segments,
                info["duration"],
                str(wd / "tts_omni"),
                target_lang=settings.target_lang,
                vocals_path=ref_src,
                preset=getattr(settings, "omnivoice_preset", "clone_auto"),
                ref_audio_path=getattr(settings, "omnivoice_ref_audio", ""),
                use_gpu=settings.use_gpu,
                max_speedup=settings.max_speedup,
                min_gap_ms=getattr(settings, "min_gap_ms", 150),
                num_step=int(getattr(settings, "omnivoice_steps", 16)),
                log=cb.log,
                progress=p,
                cancelled=cb.cancelled,
            )
        else:
            dub, tinfo = synthesize_track(
                segments,
                info["duration"],
                str(wd / "tts"),
                max_speedup=settings.max_speedup,
                min_gap_ms=getattr(settings, "min_gap_ms", 150),
                log=cb.log,
                progress=p,
                cancelled=cb.cancelled,
            )
        stage_times.append(f"Giọng đọc {time.time() - ts:.1f}s")
        fast = sum(1 for t in tinfo if t["speed"] > settings.max_speedup * 1.05)
        if fast:
            cb.log(f"ℹ Có {fast} câu thoại dịch khá dài đã được tự động cân chỉnh giãn nhịp.")
        cb.check()

        # 2. Mix Audio
        ts = time.time()
        p = cb.stage(STAGE_MIX)
        mix = _mix(dub, wd, info, settings, video, cb)
        audio_p = wd / "dub_audio.wav"
        media.save_wav(audio_p, mix, SR)
        stage_times.append(f"Trộn nhạc {time.time() - ts:.1f}s")
        p(1.0)
        cb.check()
    else:
        # Subtitle-only mode: keep exact burned-in frame timings (no TTS drift!)
        for s in segments:
            s.pop("dub_start", None)
            s.pop("dub_end", None)
        cb.log("📝 Chế độ chỉ dịch phụ đề: giữ nguyên âm thanh gốc (hoặc không tiếng), không chèn giọng đọc.")

    # 3. Old Subtitle Detection (OCR inside movable 4-sided ROI)
    samples = []
    if settings.blur_old_subs or (settings.burn_new_subs and settings.sub_place_on_old):
        ts = time.time()
        p = cb.stage(STAGE_OCR)
        from .subtitles import detect_subtitles, roi_cache_key
        key = roi_cache_key(settings)
        cache = wd / "ocr_subs.json"
        data = {}
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        if data.get("key") == key:
            samples = data["samples"]
            cb.log("Dùng lại kết quả quét vùng phụ đề OCR đã lưu")
        else:
            samples = detect_subtitles(video, info, settings, cb.log, p)
            cache.write_text(json.dumps({"key": key, "samples": samples}), encoding="utf-8")
        stage_times.append(f"OCR {time.time() - ts:.1f}s")
        p(1.0)
        cb.check()

    # 4. Generate Clean Synced ASS & SRT Subtitles
    from .subtitles import build_ass, build_srt, Blurrer
    (wd / "subs.srt").write_text(build_srt(segments), encoding="utf-8")
    ass_name = "subs.ass"
    (wd / ass_name).write_text(
        build_ass(segments, samples, settings.ocr_fps, W, H, tgt.font, settings),
        encoding="utf-8-sig",
    )

    # 5. Video Encode & Subtitle Burn (GPU hwaccel decode + Multi-threaded Blurrer + GPU NVENC/AMF/QSV encode!)
    ts = time.time()
    p = cb.stage(STAGE_RENDER)
    suffix = "sub" if mode == "ocr_sub_only" else "dub"
    out = Path(output) if output else Path(video).with_name(f"{Path(video).stem}_{settings.target_lang}_{suffix}.mp4")
    out = out.resolve()
    encoder = _pick_encoder(settings.use_gpu, W, H)
    enc_flags = _enc_args(encoder, settings.video_crf, W, H, info["fps"])
    hwaccel = prof.ffmpeg_hwaccel if settings.use_gpu else ""
    vf = [f"subtitles={ass_name}"] if settings.burn_new_subs else []
    use_blur = settings.blur_old_subs and any(s["blocks"] for s in samples)
    enc_label = f"{encoder} (GPU Hardware MFT)" if encoder == "h264_mf" else encoder
    extra_enc_note = f" · giải mã {hwaccel} · {prof.blur_workers} luồng xóa sub" if (use_blur and hwaccel) else ""
    cb.log(f"Mã hóa video đầu ra bằng {enc_label}{extra_enc_note}")
    dur_sec = max(0.1, float(info["duration"]))
    log_f = open(wd / "ffmpeg_render.log", "wb")
    try:
        if not use_blur:
            cmd = ["ffmpeg", "-y", "-progress", "pipe:1", "-nostats"]
            if hwaccel:
                cmd += ["-hwaccel", "auto"]
            cmd += ["-i", video]
            if audio_p is not None:
                cmd += ["-i", str(audio_p)]
            if vf:
                cmd += ["-vf", ",".join(vf)]
            if audio_p is not None:
                cmd += ["-map", "0:v:0", "-map", "1:a:0"] + enc_flags + [
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)
                ]
            elif info["has_audio"]:
                cmd += ["-map", "0:v:0", "-map", "0:a:0"] + enc_flags + [
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)
                ]
            else:
                cmd += ["-map", "0:v:0", "-an"] + enc_flags + [
                    "-movflags", "+faststart", str(out)
                ]

            proc = subprocess.Popen(
                cmd, cwd=str(wd), stderr=log_f, stdout=subprocess.PIPE,
                text=True, encoding="utf-8", errors="ignore", creationflags=CREATE_NO_WINDOW
            )
            try:
                for line in proc.stdout:
                    if cb.cancelled():
                        proc.kill()
                        raise InterruptedError("Đã hủy")
                    if line.startswith("out_time_us="):
                        try:
                            us = int(line.split("=", 1)[1].strip())
                            p(min(0.99, (us / 1_000_000.0) / dur_sec))
                        except ValueError:
                            pass
            finally:
                if proc.stdout:
                    proc.stdout.close()
            proc.wait()
        else:
            blur = Blurrer(samples, settings.ocr_fps, W, H, settings.blur_style)
            fps_str = info["fps_str"] or f"{info['fps']}"
            cmd = [
                "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{W}x{H}", "-r", fps_str, "-i", "-"
            ]
            if audio_p is not None:
                cmd += ["-i", str(audio_p)]
            elif info["has_audio"]:
                cmd += ["-i", video]
            if vf:
                cmd += ["-vf", ",".join(vf)]
            if audio_p is not None:
                cmd += ["-map", "0:v:0", "-map", "1:a:0"] + enc_flags + [
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)
                ]
            elif info["has_audio"]:
                cmd += ["-map", "0:v:0", "-map", "1:a:0"] + enc_flags + [
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)
                ]
            else:
                cmd += ["-map", "0:v:0", "-an"] + enc_flags + [
                    "-movflags", "+faststart", str(out)
                ]

            proc = subprocess.Popen(
                cmd, cwd=str(wd), stdin=subprocess.PIPE, stderr=log_f,
                stdout=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW,
                bufsize=W * H * 3 * 2,
            )
            total = max(1, int(info["duration"] * info["fps"]))
            fps = info["fps"]
            n_workers = max(1, int(prof.blur_workers))
            max_inflight = n_workers * 2

            def _process_frame_bytes(frm_copy, t_sec: float) -> bytes:
                return blur.apply(frm_copy, t_sec).tobytes()

            try:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    pending_q = deque()
                    written = 0
                    for i, frame in media.read_frames(video, fps_str, W, H, hwaccel=hwaccel):
                        t_sec = i / fps
                        if not blur.rects_at(t_sec):
                            # Fast path: no subtitle on this frame -> zero copy/blur overhead!
                            raw_b = frame.tobytes()
                            pending_q.append(pool.submit(lambda b=raw_b: b))
                        else:
                            # Copy buffer slice before read_frames advances & process in parallel C++ threads
                            pending_q.append(pool.submit(_process_frame_bytes, frame.copy(), t_sec))

                        if len(pending_q) >= max_inflight:
                            proc.stdin.write(pending_q.popleft().result())
                            written += 1
                            if written % 15 == 0:
                                p(min(0.99, written / total))
                                if cb.cancelled():
                                    proc.kill()
                                    raise InterruptedError("Đã hủy")

                    while pending_q:
                        proc.stdin.write(pending_q.popleft().result())
                        written += 1
                        if written % 15 == 0:
                            p(min(0.99, written / total))
                            if cb.cancelled():
                                proc.kill()
                                raise InterruptedError("Đã hủy")
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            proc.wait()
    finally:
        log_f.close()

    if proc.returncode != 0:
        tail = (wd / "ffmpeg_render.log").read_text("utf-8", "ignore")[-2000:]
        raise RuntimeError("FFmpeg lỗi khi xuất video:\n" + tail)
    stage_times.append(f"Xuất video {time.time() - ts:.1f}s")
    p(1.0)
    cb.log(f"⏱ Thống kê tốc độ: {' · '.join(stage_times)}")
    cb.log(f"✔ Hoàn tất xuất video sau {time.time() - t0:.0f}s: {out}")
    return str(out)
