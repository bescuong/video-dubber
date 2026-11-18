"""Detect hard-coded (burned-in) subtitles with OCR, cleanly erase/cover them, and build steady ASS subtitles."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .media import read_frames

OCR_WIDTH = 960


# =============================================================================
# 1. SUBTITLE DETECTION & TEMPORAL STABILIZATION
# =============================================================================
def _get_detector():
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()

    def detect(img):
        res = engine.text_det(img)
        boxes = res[0] if isinstance(res, tuple) else res
        if boxes is None:
            return []
        out = []
        for b in boxes:
            b = np.asarray(b)
            if b.ndim == 2:
                out.append([float(b[:, 0].min()), float(b[:, 1].min()),
                            float(b[:, 0].max()), float(b[:, 1].max())])
            else:
                out.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        return out

    return detect


def _merge_lines(boxes, W=None, H=None):
    """Merge word boxes on the same text line, then stack adjacent lines into subtitle blocks."""
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    lines = []
    for b in boxes:
        h = b[3] - b[1]
        for L in lines:
            lh = L[3] - L[1]
            ov = min(L[3], b[3]) - max(L[1], b[1])
            gap = max(L[0], b[0]) - min(L[2], b[2])
            if ov > 0.45 * min(h, lh) and gap < 2.4 * max(h, lh):
                L[0], L[1], L[2], L[3] = min(L[0], b[0]), min(L[1], b[1]), max(L[2], b[2]), max(L[3], b[3])
                break
        else:
            lines.append(list(b))

    lines.sort(key=lambda b: b[1])
    blocks = []
    for L in lines:
        lh = L[3] - L[1]
        for B in blocks:
            hov = min(B["r"][2], L[2]) - max(B["r"][0], L[0])
            vgap = L[1] - B["r"][3]
            if vgap < 0.85 * lh and hov > -0.5 * lh:
                r = B["r"]
                B["r"] = [min(r[0], L[0]), min(r[1], L[1]), max(r[2], L[2]), max(r[3], L[3])]
                B["lh"].append(lh)
                break
        else:
            blocks.append({"r": list(L), "lh": [lh]})

    valid_blocks = []
    for B in blocks:
        r = B["r"]
        bw = r[2] - r[0]
        med_lh = float(np.median(B["lh"]))

        # Allow short utterances ("OK!", "Hả?", "네!") down to 1.1x line height
        if bw < 1.1 * med_lh:
            continue

        if W is not None:
            if bw < 0.028 * W:
                continue
            # Filter small corner watermarks/logos hugging extreme left/right screen edges
            is_near_edge = (r[0] < 0.035 * W) or (r[2] > 0.965 * W)
            if bw < 0.15 * W and is_near_edge:
                continue

        valid_blocks.append({"rect": r, "line_h": med_lh})

    return valid_blocks


def _remove_static(samples, W, H, min_fraction=0.18):
    """Drop boxes that stay fixed in one spot for >= 18% of the video (logos, channel watermarks)."""
    if len(samples) < 30:
        return samples
    qw = max(1, int(0.015 * W))
    qh = max(1, int(0.015 * H))
    q = lambda r: (int(r[0] / qw), int(r[1] / qh), int(r[2] / qw), int(r[3] / qh))

    counts = {}
    for s in samples:
        for B in s["blocks"]:
            k = q(B["rect"])
            counts[k] = counts.get(k, 0) + 1

    threshold = max(6, int(min_fraction * len(samples)))
    static = {k for k, c in counts.items() if c >= threshold}
    if static:
        for s in samples:
            s["blocks"] = [B for B in s["blocks"] if q(B["rect"]) not in static]
    return samples


def _temporal_stabilize(samples, W, H):
    """Preserve short subtitles on the main subtitle band, remove off-band 1-frame noise,
    and bridge 1-frame OCR dropouts so old subtitles never flash unblurred."""
    n = len(samples)
    if n < 3:
        return samples

    def v_overlap(r1, r2):
        ov = min(r1[3], r2[3]) - max(r1[1], r2[1])
        mh = max(1.0, min(r1[3] - r1[1], r2[3] - r2[1]))
        return ov >= 0.35 * mh

    # Step 1: Identify multi-frame confirmed blocks to learn the video's true Subtitle Y-Band
    confirmed_cys = []
    confirmed_lhs = []
    for i in range(n):
        prev_b = samples[i - 1]["blocks"] if i > 0 else []
        next_b = samples[i + 1]["blocks"] if i < n - 1 else []
        for B in samples[i]["blocks"]:
            r = B["rect"]
            if any(v_overlap(r, p["rect"]) for p in prev_b) or any(v_overlap(r, nx["rect"]) for nx in next_b):
                confirmed_cys.append(0.5 * (r[1] + r[3]))
                confirmed_lhs.append(B["line_h"])

    has_band = len(confirmed_cys) >= 3
    band_cy = float(np.median(confirmed_cys)) if has_band else 0.86 * H
    band_lh = float(np.median(confirmed_lhs)) if has_band else 0.045 * H

    # Step 2: Filter isolated 1-frame noise ONLY if outside the main subtitle band
    for i in range(n):
        curr = samples[i]["blocks"]
        if not curr:
            continue
        prev_b = samples[i - 1]["blocks"] if i > 0 else []
        next_b = samples[i + 1]["blocks"] if i < n - 1 else []
        kept = []
        for B in curr:
            r = B["rect"]
            has_neighbor = (
                any(v_overlap(r, p["rect"]) for p in prev_b) or
                any(v_overlap(r, nx["rect"]) for nx in next_b)
            )
            if has_neighbor:
                kept.append(B)
            elif has_band:
                # Keep short 1-frame exclamation subtitles ("OK!", "Hả?") if centered on subtitle Y-band
                cy = 0.5 * (r[1] + r[3])
                cx = 0.5 * (r[0] + r[2])
                if abs(cy - band_cy) <= 1.6 * band_lh and (0.20 * W <= cx <= 0.80 * W):
                    kept.append(B)
        samples[i]["blocks"] = kept

    # Step 3: Bridge 1-frame OCR dropouts (when i-1 and i+1 have subtitles at the same Y-band, fill i)
    for i in range(1, n - 1):
        if samples[i]["blocks"]:
            continue
        prev_b = samples[i - 1]["blocks"]
        next_b = samples[i + 1]["blocks"]
        if prev_b and next_b:
            bridged = []
            for p in prev_b:
                for nx in next_b:
                    if v_overlap(p["rect"], nx["rect"]):
                        pr, nr = p["rect"], nx["rect"]
                        bridged.append({
                            "rect": [min(pr[0], nr[0]), min(pr[1], nr[1]),
                                     max(pr[2], nr[2]), max(pr[3], nr[3])],
                            "line_h": 0.5 * (p["line_h"] + nx["line_h"]),
                        })
            if bridged:
                samples[i]["blocks"] = bridged

    return samples


def detect_subtitles(video, info, settings, log=print, progress=None):
    """Return list of samples: {"t": seconds, "blocks": [{"rect":[x1,y1,x2,y2], "line_h": px}]}
    in full-resolution coordinates."""
    detect = _get_detector()
    W, H = info["width"], info["height"]
    sfps = float(settings.ocr_fps)
    total = max(1, int(info["duration"] * sfps))
    top_pct = int(settings.sub_region_top * 100)
    bot_pct = int(settings.sub_region_bottom * 100)
    log(f"Quét phụ đề cũ bằng OCR ({sfps:g} fps, vùng: {top_pct}% → {bot_pct}% chiều cao, ~{total} khung)...")

    samples = []
    for i, frame in read_frames(video, info["fps"], W, H, scale_width=OCR_WIDTH, sample_fps=sfps):
        h, w = frame.shape[:2]
        scale = W / w
        y0, y1 = int(settings.sub_region_top * h), int(settings.sub_region_bottom * h)
        crop = frame[y0:y1]
        boxes = []
        if crop.size:
            for x1, a, x2, b in detect(crop):
                bh = b - a
                bw = x2 - x1
                if not (0.016 * h <= bh <= 0.16 * h):
                    continue
                if bw < 0.9 * bh:
                    continue
                boxes.append([x1, a + y0, x2, b + y0])

        blocks = _merge_lines(boxes, W=w, H=h) if boxes else []
        for B in blocks:
            B["rect"] = [v * scale for v in B["rect"]]
            B["line_h"] *= scale
        samples.append({"t": i / sfps, "blocks": blocks})
        if progress and i % 8 == 0:
            progress(min(1.0, i / total))

    samples = _remove_static(samples, W, H)
    samples = _temporal_stabilize(samples, W, H)

    n_found = sum(1 for s in samples if s["blocks"])
    log(f"Phát hiện phụ đề gốc ở {n_found}/{len(samples)} khung mẫu")
    return samples


# =============================================================================
# 2. CLEAN SUBTITLE ERASURE / COVERING (ZERO GHOST STROKES)
# =============================================================================
def _merge_overlapping_rects(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge overlapping or closely adjacent rectangles so we cover smoothly without seams."""
    if len(rects) <= 1:
        return rects
    boxes = [list(r) for r in sorted(rects, key=lambda r: (r[1], r[0]))]
    merged = []
    for b in boxes:
        for m in merged:
            ov_x = min(m[2], b[2]) - max(m[0], b[0])
            ov_y = min(m[3], b[3]) - max(m[1], b[1])
            if ov_x >= -16 and ov_y >= -10:
                m[0], m[1] = min(m[0], b[0]), min(m[1], b[1])
                m[2], m[3] = max(m[2], b[2]), max(m[3], b[3])
                break
        else:
            merged.append(b)
    return [tuple(m) for m in merged]


class Blurrer:
    def __init__(self, samples, ocr_fps, W, H, style="inpaint_blur", pad_x=None, pad_y=None):
        self.samples = samples
        self.sfps = float(ocr_fps)
        self.W, self.H = W, H
        self.style = style or "inpaint_blur"
        # Generous padding to completely cover outlines, drop shadows, punctuation & accents
        self.pad_x = pad_x if pad_x is not None else max(18, int(0.028 * W))
        self.pad_y = pad_y if pad_y is not None else max(12, int(0.020 * H))
        # Lookaround window (±0.22s) so first & last frames of every subtitle are 100% covered
        self.lookaround_sec = max(0.22, 1.15 / max(1.0, self.sfps))
        self._morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def rects_at(self, t: float) -> list[tuple[int, int, int, int]]:
        if not self.samples:
            return []
        i_min = max(0, int(math.floor((t - self.lookaround_sec) * self.sfps)))
        i_max = min(len(self.samples) - 1, int(math.ceil((t + self.lookaround_sec) * self.sfps)))

        raw_rects = []
        px, py = self.pad_x, self.pad_y
        for i in range(i_min, i_max + 1):
            if abs(self.samples[i]["t"] - t) <= self.lookaround_sec + 1e-4:
                for B in self.samples[i]["blocks"]:
                    x1, y1, x2, y2 = B["rect"]
                    raw_rects.append((
                        max(0, int(x1 - px)),
                        max(0, int(y1 - py)),
                        min(self.W, int(x2 + px)),
                        min(self.H, int(y2 + py)),
                    ))
        return _merge_overlapping_rects(raw_rects)

    @staticmethod
    def _feather_blend(orig_roi: np.ndarray, covered_roi: np.ndarray, border: int = 8) -> np.ndarray:
        """Softly blend edges of covered ROI into original frame so no hard box seam is visible."""
        h, w = orig_roi.shape[:2]
        b = min(border, h // 4, w // 4)
        if b < 2:
            return covered_roi
        wy = np.ones(h, dtype=np.float32)
        wx = np.ones(w, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, b, dtype=np.float32)
        wy[:b] = ramp
        wy[-b:] = ramp[::-1]
        wx[:b] = ramp
        wx[-b:] = ramp[::-1]
        alpha = (wy[:, None] * wx[None, :])[:, :, None]
        return (orig_roi.astype(np.float32) * (1.0 - alpha) + covered_roi.astype(np.float32) * alpha).astype(np.uint8)

    def _inpaint_and_blur_roi(self, roi: np.ndarray) -> np.ndarray:
        """Erase bright/dark text strokes via fast downscaled Telea inpainting + smooth background blur."""
        h, w = roi.shape[:2]
        # Downscale ROI so morphological stroke detection + Telea inpaint runs >200 fps
        target_h = min(h, 68)
        scale = target_h / max(1, h)
        sw = max(8, int(round(w * scale)))
        sh = max(8, int(round(h * scale)))
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Catch both bright white/yellow characters (tophat) and black outlines/dark text (blackhat)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self._morph_k)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self._morph_k)
        stroke_energy = cv2.add(tophat, blackhat)

        # Also catch solid thick white cores (> 215 luminance)
        _, mask_edges = cv2.threshold(stroke_energy, 26, 255, cv2.THRESH_BINARY)
        _, mask_bright = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask_edges, mask_bright)
        mask = cv2.dilate(mask, self._dilate_k, iterations=1)

        # Keep outer 2px border of small ROI unmasked so Telea anchors cleanly to background colors
        mask[:2, :] = 0
        mask[-2:, :] = 0
        mask[:, :2] = 0
        mask[:, -2:] = 0

        erased = cv2.inpaint(small, mask, 3, cv2.INPAINT_TELEA)
        smoothed = cv2.GaussianBlur(erased, (0, 0), max(2.2, sh / 8.5))
        upscaled = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_LINEAR)
        return self._feather_blend(roi, upscaled, border=max(6, h // 7))

    def apply(self, frame: np.ndarray, t: float) -> np.ndarray:
        rects = self.rects_at(t)
        if not rects:
            return frame
        frame = frame.copy()
        for x1, y1, x2, y2 in rects:
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            roi = frame[y1:y2, x1:x2]
            h, w = roi.shape[:2]

            if self.style == "pixelate":
                f = max(6, h // 4)
                small = cv2.resize(roi, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
                cov = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                roi[:] = self._feather_blend(roi, cov, border=max(4, h // 9))
            elif self.style == "box":
                # Frosted dark cinema backdrop — 100% hides old subs & boosts new sub contrast
                small = cv2.resize(roi, (max(1, w // 6), max(1, h // 6)), interpolation=cv2.INTER_AREA)
                blurred = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3.0), (w, h), interpolation=cv2.INTER_LINEAR)
                darkened = (blurred.astype(np.float32) * 0.24 + np.array([20.0, 20.0, 26.0], dtype=np.float32) * 0.76).astype(np.uint8)
                roi[:] = self._feather_blend(roi, darkened, border=max(6, h // 7))
            elif self.style == "blur":
                # Heavy multi-pass frosted blur
                small = cv2.resize(roi, (max(1, w // 6), max(1, h // 6)), interpolation=cv2.INTER_AREA)
                sigma = max(4.0, small.shape[0] / 3.0)
                small = cv2.GaussianBlur(small, (0, 0), sigma)
                cov = cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), max(4.0, h / 6.0))
                roi[:] = self._feather_blend(roi, cov, border=max(6, h // 7))
            else:
                # Default "inpaint_blur": removes white/yellow strokes completely + smooth feathered blend
                roi[:] = self._inpaint_and_blur_roi(roi)

        return frame


# =============================================================================
# 3. STEADY, NON-OVERLAPPING ASS & SRT GENERATION
# =============================================================================
def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _wrap_text(text: str, max_chars: int = 42) -> str:
    """Balanced 2-line wrapping supporting both space-separated scripts and CJK."""
    text = text.strip()
    if "\n" in text or len(text) <= max_chars:
        return text

    words = text.split()
    # CJK text without spaces (Japanese / Chinese)
    if len(words) <= 2 and len(text) > max_chars:
        mid = len(text) // 2
        # Prefer splitting near CJK punctuation if close to middle
        for offset in range(min(8, mid)):
            for pos in (mid + offset, mid - offset):
                if 2 < pos < len(text) - 2 and text[pos - 1] in "、，。！？；：,.!? ":
                    return text[:pos].strip() + "\n" + text[pos:].strip()
        return text[:mid] + "\n" + text[mid:]

    if len(words) < 4:
        return text
    mid = len(text) // 2
    best_i, best_d = 1, 10**9
    acc = 0
    for i, w in enumerate(words[:-1], 1):
        acc += len(w) + (1 if i > 1 else 0)
        d = abs(acc - mid)
        if d < best_d:
            best_i, best_d = i, d
    return " ".join(words[:best_i]) + "\n" + " ".join(words[best_i:])


def _ass_escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def _normalized_sub_intervals(segments: list[dict]) -> list[tuple[float, float, str]]:
    """Extract (start, end, translation) synced to TTS timing and strictly prevent overlaps."""
    items = []
    for s in segments:
        raw = s.get("translation", "").strip()
        if not raw:
            continue
        st = float(s.get("dub_start", s["start"]))
        en = float(s.get("dub_end", s["end"]))
        en = max(st + 0.25, en)
        items.append([st, en, raw])

    # Strictly eliminate any timestamp overlap so two subtitles NEVER stack on screen
    for i in range(len(items) - 1):
        if items[i][1] > items[i + 1][0] - 0.04:
            items[i][1] = max(items[i][0] + 0.15, items[i + 1][0] - 0.04)
            if items[i + 1][0] < items[i][1]:
                items[i + 1][0] = items[i][1] + 0.02
                items[i + 1][1] = max(items[i + 1][0] + 0.20, items[i + 1][1])

    return [(st, en, txt) for st, en, txt in items]


def build_ass(segments, samples, ocr_fps, W, H, font, settings):
    """Build clean, rock-steady ASS subtitles with unified vertical placement and font size."""
    # Compute ONE unified vertical center & font size across the whole video (no per-line jumping!)
    all_cys = []
    all_lhs = []
    if samples and settings.sub_place_on_old:
        for smp in samples:
            if smp["blocks"]:
                dom = max(smp["blocks"], key=lambda B: (B["rect"][2] - B["rect"][0]) * (B["rect"][3] - B["rect"][1]))
                r = dom["rect"]
                all_cys.append(0.5 * (r[1] + r[3]))
                all_lhs.append(dom["line_h"])

    if settings.sub_font_size > 0:
        global_fs = int(settings.sub_font_size)
    elif all_lhs:
        med_lh = float(np.median(all_lhs))
        global_fs = int(np.clip(round(med_lh * 0.88), H * 0.038, H * 0.062))
    else:
        global_fs = int(round(H * 0.048))

    global_cy = None
    if all_cys and settings.sub_place_on_old:
        global_cy = int(np.clip(round(float(np.median(all_cys))), H * 0.08, H * 0.92))

    margin_v = int(H * 0.065)
    outline = max(2.4, H / 260.0)
    shadow = max(1.2, outline * 0.45)

    # BorderStyle=3 draws a sleek semi-transparent dark box behind subtitle text; BorderStyle=1 draws crisp outline
    use_bg_box = bool(getattr(settings, "sub_bg_box", False))
    border_style = 3 if use_bg_box else 1
    back_colour = "&H55000000" if use_bg_box else "&H78000000"
    outline_colour = "&H55101014" if use_bg_box else "&H00000000"
    box_pad = max(6.0, global_fs * 0.22) if use_bg_box else outline

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {W}",
        f"PlayResY: {H}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font},{global_fs},&H00FFFFFF,&H000000FF,{outline_colour},{back_colour},"
        f"-1,0,0,0,100,100,0,0,{border_style},{box_pad:.1f},{0.0 if use_bg_box else shadow:.1f},"
        f"2,{int(W * 0.06)},{int(W * 0.06)},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    max_chars = (26 if font in ("Yu Gothic", "Microsoft YaHei", "Malgun Gothic") else 42) if W >= H else 24
    pos_tag = f"{{\\an5\\pos({W // 2},{global_cy})}}" if global_cy is not None else ""

    for st, en, raw in _normalized_sub_intervals(segments):
        text = _wrap_text(raw, max_chars=max_chars)
        lines.append(
            f"Dialogue: 0,{_ass_time(st)},{_ass_time(en)},Default,,0,0,0,,{pos_tag}{_ass_escape(text)}"
        )

    return "\n".join(lines) + "\n"


def build_srt(segments):
    def ts(t):
        ms = int(round(max(0, t) * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    out = []
    for n, (st, en, txt) in enumerate(_normalized_sub_intervals(segments), 1):
        out.append(f"{n}\n{ts(st)} --> {ts(en)}\n{txt}\n")
    return "\n".join(out)
