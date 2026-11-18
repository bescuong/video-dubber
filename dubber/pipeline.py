"""End-to-end dubbing pipeline.

Stage 1  analyze(): extract audio -> separate vocals -> speech recognition -> translation -> speaker/character analysis
Stage 2  render():  TTS fitted to timing -> mix with background -> OCR old subs -> erase/cover + new subs -> mp4
"""
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
    video = str(Path(video).resolve())
    wd = work_dir(video)
    info = media.probe(video)
    if not info["has_audio"]:
        raise RuntimeError("Video không có kênh âm thanh")
    cb.log(f"Video {info['width']}x{info['height']} @ {info['fps']:.3f} fps, {info['duration']:.1f}s")

    # 1. Audio extraction & Demucs separation (cached)
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
    p(1.0)
    cb.check()

    # 2. Speech recognition (Whisper)
    p = cb.stage(STAGE_ASR)
    wav16 = wd / "asr16k.wav"
    media.run(["ffmpeg", "-v", "error", "-y", "-i", str(asr_src), "-ac", "1", "-ar", "16000", str(wav16)])
    from .transcribe import transcribe
    lang = None if settings.source_lang == "auto" else LANGUAGES.get(settings.source_lang, LANGUAGES["en"]).whisper_code
    segs, detected = transcribe(
        wav16, settings.whisper_model, lang, settings.use_gpu, cb.log, p, info["duration"]
    )
    cb.log(f"Nhận dạng được {len(segs)} câu thoại")
    cb.check()
    if not segs:
        raise RuntimeError("Không nhận dạng được lời thoại nào trong video")

    # 3. Machine Translation (NLLB-200)
    p = cb.stage(STAGE_TRANSLATE)
    src_code = _match_lang_code(detected, fallback=(settings.source_lang if settings.source_lang != "auto" else "en"))
    if detected not in LANGUAGES and src_code != detected:
        cb.log(f"Ngôn ngữ '{detected}' được ánh xạ về '{src_code}'")
    tgt = LANGUAGES.get(settings.target_lang, LANGUAGES["vi"])
    src_obj = LANGUAGES.get(src_code, LANGUAGES["en"])
    if src_obj.nllb == tgt.nllb:
        translations = [s["text"] for s in segs]
        p(1.0)
    else:
        from .translate import translate
        translations = translate(
            [s["text"] for s in segs], src_obj.nllb, tgt.nllb,
            settings.nllb_model, settings.use_gpu, cb.log, p
        )
    for s, t in zip(segs, translations):
        s["translation"] = t
    cb.check()

    # 4. Speaker Gender / Age / Multi-Character Diarization
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
    p(1.0)

    save_project(video, settings, segs, src_code)
    cb.log(f"✔ Hoàn tất phân tích sau {time.time() - t0:.0f}s. Bạn có thể kiểm tra/sửa lời dịch trước khi xuất.")
    return segs, src_code


# ============================================================================ STAGE 2: RENDER
def _rms_active(x, thr=0.005):
    m = np.abs(x) > thr
    return float(np.sqrt(np.mean(x[m] ** 2))) if m.sum() > 100 else 0.0


def _mix(dub, wd, info, settings, video, cb):
    n = len(dub)
    vocals_p, bg_p = wd / "vocals.wav", wd / "background.wav"
    if settings.separate_vocals and bg_p.exists():
        bg = media.load_audio(bg_p, SR, 2)
        vocals = media.load_audio(vocals_p, SR, 2)
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


def _pick_encoder(use_gpu):
    if use_gpu:
        try:
            media.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=320x240:d=0.2",
                "-c:v", "h264_nvenc", "-f", "null", "-"
            ])
            return "h264_nvenc"
        except Exception:
            pass
    return "libx264"


def _enc_args(encoder, crf):
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(crf + 1), "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]


def render(video, settings: Settings, segments, cb: Callbacks = None, output=None):
    cb = cb or Callbacks()
    t0 = time.time()
    video = str(Path(video).resolve())
    wd = work_dir(video)
    info = media.probe(video)
    W, H = info["width"], info["height"]
    tgt = LANGUAGES.get(settings.target_lang, LANGUAGES["vi"])
    save_project(video, settings, segments, (load_project(video) or {}).get("source_lang", "auto"))

    # 1. TTS with Character Voices & Anti-Slur Elastic Timeline
    p = cb.stage(STAGE_TTS)
    from .gender import resolve_voice_for_segment
    from .tts import synthesize_track

    for s in segments:
        s["voice"] = resolve_voice_for_segment(s, settings)

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
    fast = sum(1 for t in tinfo if t["speed"] > settings.max_speedup * 1.05)
    if fast:
        cb.log(f"ℹ Có {fast} câu thoại dịch khá dài đã được tự động cân chỉnh giãn nhịp.")
    cb.check()

    # 2. Mix Audio
    p = cb.stage(STAGE_MIX)
    mix = _mix(dub, wd, info, settings, video, cb)
    audio_p = wd / "dub_audio.wav"
    media.save_wav(audio_p, mix, SR)
    p(1.0)
    cb.check()

    # 3. Old Subtitle Detection (OCR)
    samples = []
    if settings.blur_old_subs or (settings.burn_new_subs and settings.sub_place_on_old):
        p = cb.stage(STAGE_OCR)
        key = f"v2_{settings.sub_region_top:.3f}_{settings.sub_region_bottom:.3f}_{settings.ocr_fps:g}"
        cache = wd / "ocr_subs.json"
        data = {}
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        if data.get("key") == key:
            samples = data["samples"]
            cb.log("Dùng lại kết quả quét phụ đề OCR đã lưu")
        else:
            from .subtitles import detect_subtitles
            samples = detect_subtitles(video, info, settings, cb.log, p)
            cache.write_text(json.dumps({"key": key, "samples": samples}), encoding="utf-8")
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

    # 5. Video Encode & Subtitle Burn
    p = cb.stage(STAGE_RENDER)
    out = Path(output) if output else Path(video).with_name(f"{Path(video).stem}_{settings.target_lang}_dub.mp4")
    out = out.resolve()
    encoder = _pick_encoder(settings.use_gpu)
    cb.log(f"Mã hóa video đầu ra bằng {encoder}")
    vf = [f"subtitles={ass_name}"] if settings.burn_new_subs else []
    common_out = (
        ["-map", "0:v:0", "-map", "1:a:0"]
        + _enc_args(encoder, settings.video_crf)
        + ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)]
    )
    use_blur = settings.blur_old_subs and any(s["blocks"] for s in samples)
    dur_sec = max(0.1, float(info["duration"]))
    log_f = open(wd / "ffmpeg_render.log", "wb")
    try:
        if not use_blur:
            cmd = ["ffmpeg", "-y", "-progress", "pipe:1", "-nostats", "-i", video, "-i", str(audio_p)]
            if vf:
                cmd += ["-vf", ",".join(vf)]
            cmd += common_out
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
                "-s", f"{W}x{H}", "-r", fps_str, "-i", "-", "-i", str(audio_p)
            ]
            if vf:
                cmd += ["-vf", ",".join(vf)]
            cmd += common_out
            proc = subprocess.Popen(
                cmd, cwd=str(wd), stdin=subprocess.PIPE, stderr=log_f,
                stdout=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW
            )
            total = max(1, int(info["duration"] * info["fps"]))
            fps = info["fps"]
            try:
                for i, frame in media.read_frames(video, fps_str, W, H):
                    proc.stdin.write(blur.apply(frame, i / fps).tobytes())
                    if i % 15 == 0:
                        p(min(0.99, i / total))
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
    p(1.0)
    cb.log(f"✔ Hoàn tất xuất video sau {time.time() - t0:.0f}s: {out}")
    return str(out)
