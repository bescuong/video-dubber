"""Text-to-speech (Microsoft Edge neural voices) + anti-slur elastic timeline fitting."""
from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile

import numpy as np

from .config import decode_voice_spec
from .media import load_audio, save_wav, run

SR = 44100


def _cache_name(cache_dir: str, text: str, voice_spec: str, rate: str) -> str:
    h = hashlib.md5(f"v2|{voice_spec}|{rate}|{text}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{h}.mp3")


_drm_synced = False


def _sync_drm_clock_once():
    global _drm_synced
    if _drm_synced:
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
                     on_done=None, cancelled=None, retries: int = 6):
    import edge_tts

    if cancelled and cancelled():
        raise InterruptedError("Đã hủy")

    if os.path.exists(path) and os.path.getsize(path) > 128:
        if on_done:
            on_done()
        return path

    voice_id, pitch = decode_voice_spec(voice_spec)
    async with sem:
        if cancelled and cancelled():
            raise InterruptedError("Đã hủy")
        for attempt in range(retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(0.8 * attempt)
                tmp = path + ".part"
                kwargs = {"rate": rate}
                if pitch and pitch != "+0Hz":
                    kwargs["pitch"] = pitch
                await edge_tts.Communicate(text, voice_id, **kwargs).save(tmp)
                os.replace(tmp, path)
                await asyncio.sleep(0.08)  # gentle pacing to avoid Edge WS burst throttle
                if on_done:
                    on_done()
                return path
            except InterruptedError:
                raise
            except Exception:
                if attempt == retries - 1:
                    raise


def _synth_many(jobs, concurrency: int = 2, progress_cb=None, cancelled=None):
    """jobs: list of (text, voice_spec, rate, path)."""
    total = len(jobs)
    if total == 0:
        return
    _sync_drm_clock_once()
    done_cnt = 0

    def _step():
        nonlocal done_cnt
        done_cnt += 1
        if progress_cb:
            progress_cb(done_cnt / total)

    async def main():
        sem = asyncio.Semaphore(concurrency)
        tasks = [asyncio.create_task(_synth_one(sem, *j, on_done=_step, cancelled=cancelled)) for j in jobs]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise

    asyncio.run(main())


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
    return _trim_silence(load_audio(path, SR, 1)[0])


def _rate_str(speed: float) -> str:
    pct = int(round((speed - 1.0) * 100))
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


def synthesize_preview_wav(voice_spec: str, text: str, out_wav: str) -> str:
    """Generate a short WAV file for the GUI 'Nghe thử giọng' button."""
    with tempfile.TemporaryDirectory() as td:
        mp3_p = os.path.join(td, "preview.mp3")
        _synth_many([(text, voice_spec, "+0%", mp3_p)], concurrency=1)
        y = _load_tts(mp3_p)
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
    """Synthesize and place dubbed speech with guaranteed breath pauses and elastic alignment.

    Also writes 'dub_start' and 'dub_end' onto each segment dict so subtitles match 1:1.
    """
    os.makedirs(cache_dir, exist_ok=True)
    active_pairs = [(idx, s) for idx, s in enumerate(segments) if s.get("translation", "").strip()]
    n_total = int((total_duration + 1.5) * SR)
    track = np.zeros(n_total, np.float32)
    if not active_pairs:
        return track, []

    segs = [s for _, s in active_pairs]
    min_gap = max(0.08, float(min_gap_ms) / 1000.0)

    # Pass 1: Natural +0% speed synthesis
    jobs1 = [
        (s["translation"].strip(), s["voice"], "+0%",
         _cache_name(cache_dir, s["translation"].strip(), s["voice"], "+0%"))
        for s in segs
    ]
    log(f"Tạo giọng đọc tự nhiên cho {len(jobs1)} câu thoại (Edge Neural TTS)...")
    _synth_many(
        jobs1,
        concurrency=2,
        progress_cb=(lambda f: progress(0.50 * f)) if progress else None,
        cancelled=cancelled,
    )

    audios = [_load_tts(j[3]) for j in jobs1]
    nat_durs = [len(y) / SR for y in audios]

    # Compute elastic time budgets so sentences don't speed up unnecessarily or collide
    budgets = []
    for i, s in enumerate(segs):
        orig_start = float(s["start"])
        orig_end = float(s["end"])
        next_start = float(segs[i + 1]["start"]) if (i + 1 < len(segs)) else total_duration

        # Natural window before next sentence minus mandatory breath pause
        base_avail = max(0.25, next_start - orig_start - min_gap)

        # Check if next sentence has slack we can gently borrow (up to 0.30s)
        borrow = 0.0
        if i + 1 < len(segs):
            next_slot = max(0.2, float(segs[i + 1]["end"]) - next_start)
            next_slack = max(0.0, next_slot - nat_durs[i + 1])
            after_next = float(segs[i + 2]["start"]) if (i + 2 < len(segs)) else total_duration
            gap_after_next = max(0.0, after_next - float(segs[i + 1]["end"]) - min_gap)
            borrow = min(0.30, 0.6 * (next_slack + gap_after_next))

        slot = max(orig_end - orig_start, min(base_avail + borrow, (orig_end - orig_start) + 0.85))
        # Never exceed hard ceiling to next utterance + max allowed nudge
        max_allow = max(0.25, (next_start + borrow) - orig_start - min_gap)
        budgets.append(min(slot, max_allow))

    # Pass 2: Re-synthesize cleanly via neural rate ONLY when natural speech exceeds elastic budget
    speeds = []
    jobs2 = {}
    for i, s in enumerate(segs):
        need = nat_durs[i] / max(0.22, budgets[i])
        speeds.append(need)
        if need > 1.03:
            # Add tiny 3% margin so neural rate synthesis fits without secondary atempo
            sp = min(need * 1.03, max_speedup)
            rate = _rate_str(sp)
            txt = s["translation"].strip()
            jobs2[i] = (txt, s["voice"], rate, _cache_name(cache_dir, txt, s["voice"], rate))

    if jobs2:
        log(f"Tinh chỉnh nhịp độ nói cho {len(jobs2)} câu dài để không bị dính chữ...")
        job2_list = list(jobs2.values())
        _synth_many(
            job2_list,
            concurrency=2,
            progress_cb=(lambda f: progress(0.50 + 0.38 * f)) if progress else None,
            cancelled=cancelled,
        )
        for i, j in jobs2.items():
            audios[i] = _load_tts(j[3])

    # Final Elastic Placement with strictly enforced breath gap (`min_gap`)
    info = []
    cursor = 0.0  # earliest allowed start time (in seconds) for next sentence
    for i, s in enumerate(segs):
        y = audios[i]
        orig_start = float(s["start"])
        next_orig_start = float(segs[i + 1]["start"]) if (i + 1 < len(segs)) else total_duration

        # Elastic start: never overlap previous sentence + breath gap, cap drift cleanly
        actual_start = max(orig_start, cursor)
        # Max end time before next sentence (allowing up to 0.36s elastic push on next sentence)
        if i + 1 < len(segs):
            hard_ceil_end = max(actual_start + 0.22, (next_orig_start + 0.36) - min_gap)
        else:
            hard_ceil_end = max(actual_start + 0.22, total_duration - 0.03)

        avail = max(0.22, hard_ceil_end - actual_start)
        dur = len(y) / SR
        stretched = 1.0

        # Gentle secondary time-compression ONLY if still slightly over budget (capped at 1.18x to avoid slurring)
        if dur > avail * 1.02:
            stretched = min(dur / avail, 1.18)
            y = _atempo(y, stretched)
            dur = len(y) / SR

        # Absolute safety ceiling so it never collides into the next sentence even on extreme 3x text
        abs_limit_end = (next_orig_start + 0.45 - min_gap * 0.75) if (i + 1 < len(segs)) else (total_duration + 0.2)
        max_samples = max(int(0.20 * SR), int((abs_limit_end - actual_start) * SR))
        if len(y) > max_samples:
            y = y[:max_samples].copy()
            fo = min(len(y) // 2, int(0.085 * SR))
            if fo > 4:
                y[-fo:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fo, dtype=np.float32)))
            dur = len(y) / SR

        a = max(0, int(round(actual_start * SR)))
        b = min(n_total, a + len(y))
        if b > a:
            track[a:b] += y[:b - a]

        actual_end = actual_start + (b - a) / SR
        cursor = actual_end + min_gap

        # Sync exact spoken window back to segment for 100% accurate non-overlapping subtitles
        s["dub_start"] = round(actual_start, 3)
        s["dub_end"] = round(actual_end, 3)

        info.append({
            "start": actual_start,
            "end": actual_end,
            "speed": speeds[i],
            "stretch": stretched,
        })

    if progress:
        progress(1.0)
    return track, info
