"""Speech recognition with faster-whisper (runs locally) with natural pause-aware segmentation."""
import gc
import os
import sys

SENT_END_PUNCT = tuple(".?!。？！…")
CLAUSE_PUNCT = tuple(";；,，、:：—-")


def _add_cuda_dlls():
    """Let CTranslate2 find cuBLAS/cuDNN DLLs shipped with the PyTorch wheel."""
    if sys.platform != "win32":
        return
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def _split_words(words, max_len=7.5):
    """Split a long whisper segment at natural sentence/clause punctuation or largest breath pause."""
    if not words:
        return []

    pieces = []
    cur = []
    for idx, w in enumerate(words):
        cur.append(w)
        dur = cur[-1].end - cur[0].start
        txt = w.word.strip()

        # Check gap to next word
        gap_after = (words[idx + 1].start - w.end) if (idx + 1 < len(words)) else 9.9

        # 1. Clean sentence-ending punctuation
        if dur >= 1.8 and txt.endswith(SENT_END_PUNCT):
            pieces.append(cur)
            cur = []
            continue

        # 2. Clause punctuation or clear natural breath pause (> 0.35s)
        if dur >= 3.8 and (txt.endswith(CLAUSE_PUNCT) or gap_after >= 0.35):
            pieces.append(cur)
            cur = []
            continue

        # 3. Hard length limit: backtrack up to 5 words to split at the widest pause
        if dur >= max_len and len(cur) >= 4:
            best_k = len(cur) - 1
            best_gap = -1.0
            lookback = max(2, len(cur) - 6)
            for k in range(lookback, len(cur)):
                prev_txt = cur[k - 1].word.strip()
                g = cur[k].start - cur[k - 1].end
                bonus = 0.25 if prev_txt.endswith(SENT_END_PUNCT + CLAUSE_PUNCT) else 0.0
                if g + bonus > best_gap:
                    best_gap = g + bonus
                    best_k = k
            pieces.append(cur[:best_k])
            cur = cur[best_k:]

    if cur:
        if pieces and (cur[-1].end - cur[0].start) < 0.85:
            pieces[-1].extend(cur)
        else:
            pieces.append(cur)

    out = []
    for p in pieces:
        text = "".join(x.word for x in p).strip()
        if text:
            out.append({"start": float(p[0].start), "end": float(p[-1].end), "text": text})
    return out


def _clean_and_merge_fragments(segs: list[dict], max_merged_dur: float = 7.5) -> list[dict]:
    """Merge ultra-short non-punctuated fragments and enforce clean non-overlapping boundaries."""
    if not segs:
        return []

    cleaned = []
    for s in segs:
        txt = s["text"].strip()
        if not txt:
            continue
        st = max(0.0, float(s["start"]))
        en = max(st + 0.15, float(s["end"]))
        if cleaned:
            prev = cleaned[-1]
            # Fix slight timestamp overlap from ASR
            if st < prev["end"]:
                mid = 0.5 * (prev["end"] + st)
                prev["end"] = max(prev["start"] + 0.12, mid - 0.02)
                st = mid + 0.02
                en = max(st + 0.15, en)

            gap = st - prev["end"]
            prev_dur = prev["end"] - prev["start"]
            cur_dur = en - st
            prev_ends_sent = prev["text"].endswith(SENT_END_PUNCT)

            # Merge tiny dangling fragment (< 0.55s) into neighbour if close (< 0.20s)
            if (not prev_ends_sent and gap < 0.20 and
                    (prev_dur < 0.55 or cur_dur < 0.50) and
                    (en - prev["start"]) <= max_merged_dur):
                sep = "" if prev["text"][-1:] in " \u3000" else " "
                prev["text"] = (prev["text"] + sep + txt).strip()
                prev["end"] = en
                continue

        cleaned.append({"start": round(st, 3), "end": round(en, 3), "text": txt})

    return cleaned


def transcribe(wav16k_path, model_name="medium", language=None, use_gpu=True, log=print,
               progress=None, duration=None):
    _add_cuda_dlls()
    from faster_whisper import WhisperModel

    if language and "-" in language:
        language = language.split("-")[0]

    model = None
    if use_gpu:
        try:
            model = WhisperModel(model_name, device="cuda", compute_type="int8")
            log(f"Whisper '{model_name}' chạy trên GPU (int8)")
        except Exception as e:
            log(f"Không chạy được Whisper trên GPU ({e}); chuyển sang CPU")
    if model is None:
        model = WhisperModel(model_name, device="cpu", compute_type="int8",
                             cpu_threads=max(4, (os.cpu_count() or 4) - 2))
        log(f"Whisper '{model_name}' chạy trên CPU (int8)")

    segments_iter, info = model.transcribe(
        str(wav16k_path),
        language=language,
        vad_filter=True,
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_parameters={"min_silence_duration_ms": 350, "speech_pad_ms": 180},
    )
    log(f"Ngôn ngữ nguồn: {info.language} (độ tin cậy {info.language_probability:.2f})")
    raw_out = []
    for seg in segments_iter:
        if seg.words and (seg.end - seg.start > 7.5):
            raw_out.extend(_split_words(seg.words, max_len=7.5))
        else:
            t = seg.text.strip()
            if t:
                raw_out.append({"start": float(seg.start), "end": float(seg.end), "text": t})
        if progress and duration:
            progress(min(1.0, seg.end / duration))

    del model
    gc.collect()
    out = _clean_and_merge_fragments(raw_out)
    return out, info.language
