"""Text-to-speech (Microsoft Edge neural voices) + shared anti-slur elastic timeline fitting."""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
import unicodedata

import numpy as np

from .config import decode_voice_spec
from .media import load_audio, save_wav, run

SR = 44100

# Music symbols, zero-width chars, and non-spoken control characters that trigger Edge TTS NoAudioReceived
_STAGE_DIR_RE = re.compile(
    r"^\s*[\[\(（【]\s*(?:âm nhạc|nhạc nền|tiếng động|music|applause|laughter|bgm|sfx|nhạc|im lặng|silence)\s*[\]\)）】]\s*$",
    re.IGNORECASE,
)
_STRIP_SYMBOLS_RE = re.compile(r"[♪♫♬♩♭♯★☆✦✧✨🔥🎬🎙🔊⚡✅❌⚠👉▶⏹⏳🔄🔑👤🚪📂💾➕➖🧹📝🎵🌐🎭🎧👩👨🧒👴👵🤫]+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\ufeff]+")


def sanitize_tts_text(text: str) -> str:
    """Clean text for TTS engines (Edge TTS & OmniVoice). Returns '' if nothing pronounceable remains."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = _CTRL_RE.sub("", s)
    s = _STRIP_SYMBOLS_RE.sub(" ", s)
    # Remove pictographic emojis (Unicode category So) that break SSML or cause silence
    s = "".join(" " if unicodedata.category(ch) == "So" else ch for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s or _STAGE_DIR_RE.match(s):
        return ""
    # Must contain at least one letter or digit (prevents '...' or '?!' crashing Edge TTS)
    if not any(ch.isalnum() for ch in s):
        return ""
    return s


def _cache_name(cache_dir: str, text: str, voice_spec: str, rate: str) -> str:
    h = hashlib.md5(f"v3|{voice_spec}|{rate}|{text}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{h}.mp3")


_drm_synced = False


def _sync_drm_clock_once(force: bool = False):
    global _drm_synced
    if _drm_synced and not force:
        return
    try:
        import urllib.request
        from edge_tts.drm import DRM
        with urllib.request.urlopen("https://www.microsoft.com", timeout=4) as r:
            srv_date = r.headers.get("Date")
            if srv_date:
                srv_ts = DRM.parse_rfc2616_date(srv_date)
                if srv_ts:
                    DRM.adj_clock_skew_seconds(srv_ts - DRM.get_unix_timestamp())
        _drm_synced = True
    except Exception:
        pass


async def _synth_one(sem, text: str, voice_spec: str, rate: str, path: str,
                     on_done=None, cancelled=None, retries: int = 6,
                     allow_fail: bool = False, log=None):
    import edge_tts

    if cancelled and cancelled():
        raise InterruptedError("Đã hủy")

    cleaned = sanitize_tts_text(text)
    if not cleaned:
        if on_done:
            on_done()
        return ""

    if os.path.exists(path) and os.path.getsize(path) > 128:
        if on_done:
            on_done()
        return path

    voice_id, pitch = decode_voice_spec(voice_spec)
    last_err: Exception | None = None

    async with sem:
        if cancelled and cancelled():
            raise InterruptedError("Đã hủy")
        for attempt in range(retries):
            tmp = f"{path}.{attempt}.part"
            try:
                if attempt > 0:
                    await asyncio.sleep(0.7 * attempt)

                # Progressive fallback across retries:
                # 0-1: full (voice_id, rate, pitch)
                # 2-3: drop pitch (fixes Edge SSML pitch rejection on vi/ja/ko voices)
                # 4:   drop both pitch & rate (pure default voice)
                # 5:   re-sync Microsoft clock & pure default voice
                kwargs = {}
                if attempt < 4 and rate and rate != "+0%":
                    kwargs["rate"] = rate
                if attempt < 2 and pitch and pitch != "+0Hz":
                    kwargs["pitch"] = pitch
                if attempt == retries - 1:
                    _sync_drm_clock_once(force=True)

                await edge_tts.Communicate(cleaned, voice_id, **kwargs).save(tmp)
                if not os.path.exists(tmp) or os.path.getsize(tmp) < 128:
                    raise RuntimeError("Edge TTS trả về âm thanh rỗng (NoAudioReceived)")
                os.replace(tmp, path)
                await asyncio.sleep(0.08)  # gentle pacing to avoid Edge WS burst throttle
                if on_done:
                    on_done()
                return path
            except InterruptedError:
                raise
            except Exception as e:
                last_err = e
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                if attempt == retries - 1:
                    if allow_fail:
                        if log:
                            log(f"  ⚠ Edge TTS không thể tạo giọng cho câu '{cleaned[:45]}' ({e}) — tự động để khoảng lặng.")
                        if on_done:
                            on_done()
                        return ""
                    raise last_err


def _synth_many(jobs, concurrency: int = 2, progress_cb=None, cancelled=None,
                allow_fail: bool = False, log=None):
    """jobs: list of (text, voice_spec, rate, path). Returns list of output paths (or '' if skipped)."""
    total = len(jobs)
    if total == 0:
        return []
    _sync_drm_clock_once()
    done_cnt = 0

    def _step():
        nonlocal done_cnt
        done_cnt += 1
        if progress_cb:
            progress_cb(done_cnt / total)

    async def main():
        sem = asyncio.Semaphore(concurrency)
        tasks = [
            asyncio.create_task(
                _synth_one(sem, *j, on_done=_step, cancelled=cancelled, allow_fail=allow_fail, log=log)
            )
            for j in jobs
        ]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise

    return asyncio.run(main())


def _apply_micro_fades(y: np.ndarray, fade_in_ms: float = 14.0, fade_out_ms: float = 30.0) -> np.ndarray:
    """Apply smooth half-cosine micro-fades to prevent clicks and harsh syllable boundaries."""
    if len(y) < 64:
        return y
    y = y.copy()
    fi = min(len(y) // 3, int(SR * fade_in_ms / 1000.0))
    fo = min(len(y) // 3, int(SR * fade_out_ms / 1000.0))
    if fi > 4:
        curve_in = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fi, dtype=np.float32)))
        y[:fi] *= curve_in
    if fo > 4:
        curve_out = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fo, dtype=np.float32)))
        y[-fo:] *= curve_out
    return y


def _trim_silence(y: np.ndarray) -> np.ndarray:
    """Envelope-based gentle silence trim that preserves soft initial/final consonants."""
    if len(y) == 0:
        return y
    win = max(64, int(0.010 * SR))  # 10ms RMS window
    sq = y.astype(np.float32) ** 2
    kernel = np.ones(win, dtype=np.float32) / win
    env = np.sqrt(np.convolve(sq, kernel, mode="same"))
    peak = float(np.max(env))
    if peak < 1e-4:
        return y[:0]

    # Very gentle threshold so unvoiced fricatives/stops (s, th, kh, ph, t, k) are never clipped
    thr = min(0.0035, max(0.0015, peak * 0.02))
    idx = np.where(env > thr)[0]
    if len(idx) == 0:
        return y[:0]

    pre_pad = int(0.038 * SR)   # 38ms pre-roll
    post_pad = int(0.068 * SR)  # 68ms natural release tail
    a = max(0, int(idx[0]) - pre_pad)
    b = min(len(y), int(idx[-1]) + post_pad)
    return _apply_micro_fades(y[a:b])


def _atempo(y: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 0.015 or len(y) < int(0.08 * SR):
        return y
    factor = float(np.clip(factor, 0.75, 1.22))
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "a.wav"), os.path.join(td, "b.wav")
        save_wav(src, y[None], SR)
        run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter:a", f"atempo={factor:.4f}", dst])
        out = load_audio(dst, SR, 1)[0]
        return _apply_micro_fades(out, fade_in_ms=10.0, fade_out_ms=24.0)


def _load_tts(path: str) -> np.ndarray:
    if not path or not os.path.exists(path) or os.path.getsize(path) < 128:
        return np.zeros(0, np.float32)
    try:
        return _trim_silence(load_audio(path, SR, 1)[0])
    except Exception:
        return np.zeros(0, np.float32)


def _rate_str(speed: float) -> str:
    pct = int(round((speed - 1.0) * 100))
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


def compute_elastic_budgets(segs: list[dict], nat_durs: list[float], total_duration: float,
                            min_gap_ms: int = 150) -> list[float]:
    """Compute natural elastic duration budgets for each segment before next sentence."""
    min_gap = max(0.08, float(min_gap_ms) / 1000.0)
    budgets = []
    for i, s in enumerate(segs):
        orig_start = float(s["start"])
        orig_end = float(s["end"])
        next_start = float(segs[i + 1]["start"]) if (i + 1 < len(segs)) else total_duration

        base_avail = max(0.25, next_start - orig_start - min_gap)
        borrow = 0.0
        if i + 1 < len(segs):
            next_slot = max(0.2, float(segs[i + 1]["end"]) - next_start)
            next_slack = max(0.0, next_slot - nat_durs[i + 1])
            after_next = float(segs[i + 2]["start"]) if (i + 2 < len(segs)) else total_duration
            gap_after_next = max(0.0, after_next - float(segs[i + 1]["end"]) - min_gap)
            borrow = min(0.30, 0.6 * (next_slack + gap_after_next))

        slot = max(orig_end - orig_start, min(base_avail + borrow, (orig_end - orig_start) + 0.85))
        max_allow = max(0.25, (next_start + borrow) - orig_start - min_gap)
        budgets.append(min(slot, max_allow))
    return budgets


def place_audios_on_track(segs: list[dict], audios: list[np.ndarray], speeds: list[float],
                          total_duration: float, min_gap_ms: int = 240):
    """Place per-segment 44.1kHz mono audio clips onto final dub track with guaranteed breath pauses.

    Anti-Slur v2.5 Guarantees:
    - Enforces at least `min_gap_ms` (default 240ms) clean silence between EVERY pair of sentences.
    - Caps secondary `_atempo` at <= 1.10x (never crushes syllables into fast-forward gibberish).
    - Allows smooth elastic forward drift into upcoming natural pauses instead of chopping words mid-sentence!
    - Writes exact 'dub_start' and 'dub_end' back onto each segment dict so subtitles match 1:1.
    """
    n_total = int((total_duration + 2.5) * SR)
    track = np.zeros(n_total, np.float32)
    min_gap = max(0.14, float(min_gap_ms) / 1000.0)

    info = []
    cursor = 0.0  # earliest allowed start time (in seconds) for next sentence
    for i, s in enumerate(segs):
        y = audios[i]
        orig_start = float(s["start"])
        orig_end = float(s["end"])
        next_orig_start = float(segs[i + 1]["start"]) if (i + 1 < len(segs)) else total_duration

        if len(y) == 0:
            s["dub_start"] = round(orig_start, 3)
            s["dub_end"] = round(orig_end, 3)
            info.append({"start": orig_start, "end": orig_end, "speed": 1.0, "stretch": 1.0})
            continue

        actual_start = max(orig_start, cursor)
        # Measure natural window from original slot (plus gentle 0.55s elastic cushion), NOT penalized by prior drift!
        natural_window = max(0.40, orig_end - orig_start, (next_orig_start - orig_start - min_gap) + 0.55)
        if i + 1 < len(segs):
            hard_ceil_end = max(actual_start + 0.40, (next_orig_start + 0.85) - min_gap)
            avail = max(0.40, min(natural_window, hard_ceil_end - actual_start))
        else:
            avail = max(0.40, total_duration + 0.35 - actual_start)

        dur = len(y) / SR
        stretched = 1.0

        # Ultra-gentle secondary compression capped strictly at 1.10x so voice NEVER sounds slurred!
        if dur > avail * 1.04:
            stretched = min(dur / avail, 1.10)
            y = _atempo(y, stretched)
            dur = len(y) / SR

        # Only trim cleanly at the very end of the entire video file (never chop mid-video sentences!)
        video_hard_end = total_duration + 0.35
        max_end_samples = max(int(0.25 * SR), int((video_hard_end - actual_start) * SR))
        if len(y) > max_end_samples:
            y = y[:max_end_samples].copy()
            fo = min(len(y) // 2, int(0.095 * SR))
            if fo > 4:
                y[-fo:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fo, dtype=np.float32)))
            dur = len(y) / SR

        a = max(0, int(round(actual_start * SR)))
        b = min(n_total, a + len(y))
        if b > a:
            track[a:b] += y[:b - a]

        actual_end = actual_start + (b - a) / SR
        cursor = actual_end + min_gap

        s["dub_start"] = round(actual_start, 3)
        s["dub_end"] = round(actual_end, 3)

        info.append({
            "start": actual_start,
            "end": actual_end,
            "speed": speeds[i] if i < len(speeds) else 1.0,
            "stretch": stretched,
        })
    return track, info


def synthesize_preview_wav(voice_spec: str, text: str, out_wav: str) -> str:
    """Generate a short WAV file for the GUI 'Nghe thử giọng' button (Edge TTS)."""
    with tempfile.TemporaryDirectory() as td:
        mp3_p = os.path.join(td, "preview.mp3")
        res = _synth_many([(text, voice_spec, "+0%", mp3_p)], concurrency=1, allow_fail=False)
        y = _load_tts(res[0] if res else "")
        if len(y) == 0:
            raise RuntimeError("Không tạo được âm thanh mẫu")
        peak = float(np.max(np.abs(y)))
        if peak > 0.01:
            y = y * (0.88 / peak)
        save_wav(out_wav, y[None], SR)
    return out_wav


def synthesize_track(segments: list[dict], total_duration: float, cache_dir: str,
                     max_speedup: float = 1.50, min_gap_ms: int = 150,
                     log=print, progress=None, cancelled=None):
    """Synthesize and place dubbed speech with Edge TTS, guaranteed breath pauses & elastic alignment.

    Also writes 'dub_start' and 'dub_end' onto each segment dict so subtitles match 1:1.
    """
    os.makedirs(cache_dir, exist_ok=True)
    active_pairs = [
        (idx, s) for idx, s in enumerate(segments)
        if sanitize_tts_text(s.get("translation", ""))
    ]
    n_total = int((total_duration + 1.5) * SR)
    if not active_pairs:
        return np.zeros(n_total, np.float32), []

    segs = [s for _, s in active_pairs]

    # Pass 1: Natural +0% speed synthesis
    jobs1 = [
        (sanitize_tts_text(s["translation"]), s["voice"], "+0%",
         _cache_name(cache_dir, sanitize_tts_text(s["translation"]), s["voice"], "+0%"))
        for s in segs
    ]
    log(f"Tạo giọng đọc tự nhiên cho {len(jobs1)} câu thoại (Edge Neural TTS)...")
    paths1 = _synth_many(
        jobs1,
        concurrency=2,
        progress_cb=(lambda f: progress(0.50 * f)) if progress else None,
        cancelled=cancelled,
        allow_fail=True,
        log=log,
    )

    audios = [_load_tts(p) for p in paths1]
    nat_durs = [len(y) / SR for y in audios]
    budgets = compute_elastic_budgets(segs, nat_durs, total_duration, min_gap_ms=min_gap_ms)

    # Pass 2: Re-synthesize cleanly via neural rate ONLY when natural speech exceeds elastic budget
    speeds = []
    jobs2 = {}
    for i, s in enumerate(segs):
        need = nat_durs[i] / max(0.22, budgets[i]) if nat_durs[i] > 0 else 1.0
        speeds.append(need)
        if nat_durs[i] > 0 and need > 1.03:
            sp = min(need * 1.03, max_speedup)
            rate = _rate_str(sp)
            txt = sanitize_tts_text(s["translation"])
            jobs2[i] = (txt, s["voice"], rate, _cache_name(cache_dir, txt, s["voice"], rate))

    if jobs2:
        log(f"Tinh chỉnh nhịp độ nói cho {len(jobs2)} câu dài để không bị dính chữ...")
        job2_keys = list(jobs2.keys())
        job2_list = [jobs2[k] for k in job2_keys]
        paths2 = _synth_many(
            job2_list,
            concurrency=2,
            progress_cb=(lambda f: progress(0.50 + 0.38 * f)) if progress else None,
            cancelled=cancelled,
            allow_fail=True,
            log=log,
        )
        for k, p2 in zip(job2_keys, paths2):
            y2 = _load_tts(p2)
            if len(y2) > 0:
                audios[k] = y2

    track, info = place_audios_on_track(segs, audios, speeds, total_duration, min_gap_ms=min_gap_ms)

    if progress:
        progress(1.0)
    return track, info
