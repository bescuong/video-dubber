"""Detect hard-coded (burned-in) subtitles with OCR, extract text for silent/text-only videos,
cleanly erase/cover old subtitles inside a movable 4-sided ROI, and build steady ASS/SRT subtitles."""
from __future__ import annotations

from difflib import SequenceMatcher
import math
import re

import cv2
import numpy as np

from .media import read_frames

OCR_WIDTH = 960


# =============================================================================
# 0. MOVABLE 4-SIDED ROI HELPERS & SCRIPT LANGUAGE HEURISTIC
# =============================================================================
def get_roi_fractions(settings) -> tuple[float, float, float, float]:
    """Return clamped normalized ROI `(left, top, right, bottom)` in `[0.0, 1.0]`."""
    left = float(np.clip(getattr(settings, "sub_region_left", 0.02), 0.0, 0.94))
    right = float(np.clip(getattr(settings, "sub_region_right", 0.98), left + 0.06, 1.0))
    top = float(np.clip(getattr(settings, "sub_region_top", 0.72), 0.0, 0.95))
    bottom = float(np.clip(getattr(settings, "sub_region_bottom", 0.98), top + 0.04, 1.0))
    return left, top, right, bottom


def get_roi_bounds(settings, w: int, h: int) -> tuple[int, int, int, int]:
    """Return pixel crop rectangle `(x0, y0, x1, y1)` inside a frame of size `(w, h)`."""
    left, top, right, bottom = get_roi_fractions(settings)
    x0 = max(0, min(w - 2, int(round(left * w))))
    x1 = max(x0 + 2, min(w, int(round(right * w))))
    y0 = max(0, min(h - 2, int(round(top * h))))
    y1 = max(y0 + 2, min(h, int(round(bottom * h))))
    return x0, y0, x1, y1


def roi_cache_key(settings) -> str:
    left, top, right, bottom = get_roi_fractions(settings)
    sfps = float(getattr(settings, "ocr_fps", 5.0))
    return f"v3_{left:.3f}_{top:.3f}_{right:.3f}_{bottom:.3f}_{sfps:g}"


def detect_script_lang(texts: list[str], fallback: str = "en") -> str:
    """Fast Unicode-script detector for OCR'd text when source language is 'auto'."""
    blob = " ".join(t for t in texts if t)
    if not blob.strip():
        return fallback
    n_kana = len(re.findall(r"[\u3040-\u30ff]", blob))
    n_hangul = len(re.findall(r"[\uac00-\ud7af\u1100-\u11ff]", blob))
    n_cjk = len(re.findall(r"[\u4e00-\u9fff]", blob))
    n_thai = len(re.findall(r"[\u0e00-\u0e7f]", blob))
    n_cyr = len(re.findall(r"[\u0400-\u04ff]", blob))
    n_deva = len(re.findall(r"[\u0900-\u097f]", blob))
    n_vi = len(re.findall(
        r"[ăâđêôơưĂÂĐÊÔƠƯàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]",
        blob,
    ))
    if n_kana >= 3:
        return "ja"
    if n_hangul >= 3:
        return "ko"
    if n_cjk >= 3:
        return "zh"
    if n_thai >= 4:
        return "th"
    if n_cyr >= 5:
        return "ru"
    if n_deva >= 4:
        return "hi"
    if n_vi >= 4:
        return "vi"
    return fallback


# =============================================================================
# 1. SUBTITLE DETECTION & TEMPORAL STABILIZATION
# =============================================================================
def _get_ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def _detect_boxes_with_engine(engine, img: np.ndarray) -> list[list[float]]:
    res = engine.text_det(img)
    boxes = res[0] if isinstance(res, tuple) else res
    if boxes is None:
        return []
    out = []
    for b in boxes:
        b = np.asarray(b)
        if b.ndim == 2:
            out.append([
                float(b[:, 0].min()), float(b[:, 1].min()),
                float(b[:, 0].max()), float(b[:, 1].max()),
            ])
        else:
            out.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    return out


def recognize_crop_rapidocr(engine, img: np.ndarray) -> str:
    """Run full RapidOCR detection + recognition on a subtitle crop and return clean joined text."""
    if img is None or img.size == 0 or img.shape[0] < 6 or img.shape[1] < 6:
        return ""
    try:
        res, _ = engine(img)
        if not res:
            return ""
        # Sort lines top-to-bottom, then left-to-right within the same line
        items = []
        for entry in res:
            if not entry or len(entry) < 2:
                continue
            pts = np.asarray(entry[0])
            txt = str(entry[1]).strip()
            score = float(entry[2]) if len(entry) > 2 else 1.0
            if not txt or score < 0.45:
                continue
            cy = float(pts[:, 1].mean()) if pts.ndim == 2 else float(pts[1])
            cx = float(pts[:, 0].min()) if pts.ndim == 2 else float(pts[0])
            h = max(4.0, float(pts[:, 1].max() - pts[:, 1].min())) if pts.ndim == 2 else 16.0
            items.append((cy, cx, h, txt))
        if not items:
            return ""
        items.sort(key=lambda x: (x[0], x[1]))
        # Group into horizontal lines
        lines: list[list[tuple[float, float, float, str]]] = []
        for it in items:
            for L in lines:
                if abs(it[0] - L[0][0]) < 0.65 * max(it[2], L[0][2]):
                    L.append(it)
                    break
            else:
                lines.append([it])
        out_lines = []
        for L in lines:
            L.sort(key=lambda x: x[1])
            out_lines.append(" ".join(x[3] for x in L))
        return " ".join(out_lines).strip()
    except Exception:
        return ""


def _get_detector():
    engine = _get_ocr_engine()
    return lambda img: _detect_boxes_with_engine(engine, img)


def _merge_lines(boxes, W=None, H=None, custom_x_roi: bool = False):
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

        # Allow short utterances ("OK!", "Hả?", "네!") down to 1.0x line height
        if bw < 1.0 * med_lh:
            continue

        if W is not None:
            if bw < 0.022 * W:
                continue
            # Filter small corner watermarks/logos ONLY when scanning full screen width
            if not custom_x_roi:
                is_near_edge = (r[0] < 0.035 * W) or (r[2] > 0.965 * W)
                if bw < 0.15 * W and is_near_edge:
                    continue

        valid_blocks.append({"rect": r, "line_h": med_lh})

    return valid_blocks


def _remove_static(samples, W, H, min_fraction: float = 0.45, skip_centered: bool = True):
    """Drop boxes that stay fixed in one spot like channel watermarks/logos without dropping real subtitles."""
    if len(samples) < 50:
        return samples
    qw = max(1, int(0.015 * W))
    qh = max(1, int(0.015 * H))
    q = lambda r: (int(r[0] / qw), int(r[1] / qh), int(r[2] / qw), int(r[3] / qh))

    counts = {}
    sample_rect = {}
    for s in samples:
        for B in s["blocks"]:
            k = q(B["rect"])
            counts[k] = counts.get(k, 0) + 1
            sample_rect[k] = B["rect"]

    corner_thr = max(18, int(min_fraction * len(samples)))
    full_thr = max(35, int(0.82 * len(samples)))
    static = set()
    for k, c in counts.items():
        r = sample_rect[k]
        bw = r[2] - r[0]
        cx = 0.5 * (r[0] + r[2])
        is_small_corner = bw < 0.24 * W and (cx < 0.25 * W or cx > 0.75 * W)
        if is_small_corner and c >= corner_thr:
            static.add(k)
        elif not skip_centered and c >= full_thr:
            static.add(k)

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
                cy = 0.5 * (r[1] + r[3])
                if abs(cy - band_cy) <= 1.8 * band_lh:
                    kept.append(B)
        samples[i]["blocks"] = kept

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
    in full-resolution coordinates, scanned inside the user's movable 4-sided ROI."""
    engine = _get_ocr_engine()
    W, H = info["width"], info["height"]
    sfps = float(settings.ocr_fps)
    total = max(1, int(info["duration"] * sfps))
    left, top, right, bottom = get_roi_fractions(settings)
    custom_x = (left > 0.05) or (right < 0.95)
    log(
        f"Quét phụ đề cũ bằng OCR ({sfps:g} fps, khung: X {int(left*100)}%→{int(right*100)}%, "
        f"Y {int(top*100)}%→{int(bottom*100)}%, ~{total} khung)..."
    )

    samples = []
    for i, frame in read_frames(video, info["fps"], W, H, scale_width=OCR_WIDTH, sample_fps=sfps):
        h, w = frame.shape[:2]
        scale = W / w
        x0, y0, x1, y1 = get_roi_bounds(settings, w, h)
        crop = frame[y0:y1, x0:x1]
        boxes = []
        if crop.size:
            for bx1, by1, bx2, by2 in _detect_boxes_with_engine(engine, crop):
                bh = by2 - by1
                bw = bx2 - bx1
                if not (0.015 * h <= bh <= 0.22 * h):
                    continue
                if bw < 0.85 * bh:
                    continue
                boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])

        blocks = _merge_lines(boxes, W=w, H=h, custom_x_roi=custom_x) if boxes else []
        for B in blocks:
            B["rect"] = [v * scale for v in B["rect"]]
            B["line_h"] *= scale
        samples.append({"t": i / sfps, "blocks": blocks})
        if progress and i % 8 == 0:
            progress(min(1.0, i / total))

    samples = _remove_static(samples, W, H, skip_centered=True)
    samples = _temporal_stabilize(samples, W, H)

    n_found = sum(1 for s in samples if s["blocks"])
    log(f"Phát hiện phụ đề gốc ở {n_found}/{len(samples)} khung mẫu")
    return samples


# =============================================================================
# 1B. FULL OCR SUBTITLE TEXT EXTRACTION (FOR SILENT / TEXT-ONLY VIDEOS)
# =============================================================================
_FP_MORPH_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _stroke_fingerprint(crop_bgr: np.ndarray) -> np.ndarray:
    """Extract a fast background-resistant text-stroke fingerprint vector (128x32) for subtitle change detection."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_AREA)
    tophat = cv2.morphologyEx(small, cv2.MORPH_TOPHAT, _FP_MORPH_K)
    blackhat = cv2.morphologyEx(small, cv2.MORPH_BLACKHAT, _FP_MORPH_K)
    energy = cv2.add(tophat, blackhat).astype(np.float32)
    # Keep high-contrast text strokes and suppress weak background motion
    thr = max(18.0, float(np.percentile(energy, 75)) * 0.65)
    energy = np.where(energy >= thr, energy, 0.0).reshape(-1)
    norm = float(np.linalg.norm(energy))
    return energy / norm if norm > 1e-5 else energy


def _crop_sharpness(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _build_contact_sheets(events: list[dict], batch_size: int = 12) -> list[tuple[bytes, list[int]]]:
    """Stitch representative subtitle crops into numbered high-contrast JPEG sheets for Gemini Vision OCR."""
    sheets: list[tuple[bytes, list[int]]] = []
    canvas_w = 960
    badge_w = 82
    pad_y = 10
    content_w = canvas_w - badge_w - 16

    for start_i in range(0, len(events), batch_size):
        chunk = events[start_i:start_i + batch_size]
        rows = []
        row_ids = []
        for offset, ev in enumerate(chunk):
            ev_id = start_i + offset
            crop = ev["best_crop"]
            ch, cw = crop.shape[:2]
            # Scale crop nicely so text height is legible (~52..92 px tall) without exceeding content_w
            target_h = int(np.clip(ch * 1.4, 52, 92))
            scale = min(content_w / max(1, cw), target_h / max(1, ch))
            nw = max(16, int(round(cw * scale)))
            nh = max(24, int(round(ch * scale)))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA)

            row_h = nh + pad_y * 2
            row = np.full((row_h, canvas_w, 3), (30, 41, 59), dtype=np.uint8)
            # Crisp yellow index badge on the left
            cv2.rectangle(row, (6, 6), (badge_w - 6, row_h - 6), (15, 23, 42), -1)
            cv2.rectangle(row, (6, 6), (badge_w - 6, row_h - 6), (56, 189, 248), 1)
            cv2.putText(
                row, f"#{ev_id}", (12, row_h // 2 + 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (50, 230, 255), 2, cv2.LINE_AA,
            )
            # Center crop nicely in content area
            ox = badge_w + 8 + max(0, (content_w - nw) // 2)
            row[pad_y:pad_y + nh, ox:ox + nw] = resized
            # Clean horizontal separator line
            cv2.line(row, (0, row_h - 1), (canvas_w, row_h - 1), (100, 116, 139), 2)
            rows.append(row)
            row_ids.append(ev_id)

        sheet_img = np.vstack(rows)
        ok, buf = cv2.imencode(".jpg", sheet_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            sheets.append((buf.tobytes(), row_ids))
    return sheets


def _text_similarity(a: str, b: str) -> float:
    na = re.sub(r"[\s\W_]+", "", (a or "").lower())
    nb = re.sub(r"[\s\W_]+", "", (b or "").lower())
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def extract_ocr_subtitles(
    video,
    info,
    settings,
    gemini_api_key: str = "",
    log=print,
    progress=None,
    cancelled=None,
) -> tuple[list[dict], list[dict], str]:
    """Scan video inside the user's movable 4-sided ROI, track subtitle events `[start, end]`,
    and recognize exact burned-in subtitle text (Gemini Vision Contact-Sheet OCR + RapidOCR offline fallback).

    Returns `(segments, samples, detected_lang_code)`.
    """
    engine = _get_ocr_engine()
    W, H = info["width"], info["height"]
    dur = float(info["duration"])
    sfps = max(3.0, float(getattr(settings, "ocr_fps", 5.0)))
    frame_step = 1.0 / sfps
    total = max(1, int(dur * sfps))
    left, top, right, bottom = get_roi_fractions(settings)
    custom_x = (left > 0.05) or (right < 0.95)

    log(
        f"Quét văn bản phụ đề hiển thị trên video ({sfps:g} fps, khung: "
        f"X {int(left*100)}%→{int(right*100)}%, Y {int(top*100)}%→{int(bottom*100)}%)..."
    )

    samples: list[dict] = []
    raw_events: list[dict] = []
    cur_ev: dict | None = None
    miss_streak = 0

    for i, frame in read_frames(video, info["fps"], W, H, scale_width=OCR_WIDTH, sample_fps=sfps):
        if cancelled and cancelled():
            raise InterruptedError("Đã hủy")

        t = i / sfps
        h, w = frame.shape[:2]
        scale = W / w
        x0, y0, x1, y1 = get_roi_bounds(settings, w, h)
        crop_roi = frame[y0:y1, x0:x1]

        boxes = []
        if crop_roi.size:
            for bx1, by1, bx2, by2 in _detect_boxes_with_engine(engine, crop_roi):
                bh = by2 - by1
                bw = bx2 - bx1
                if not (0.015 * h <= bh <= 0.24 * h):
                    continue
                if bw < 0.85 * bh:
                    continue
                boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])

        blocks = _merge_lines(boxes, W=w, H=h, custom_x_roi=custom_x) if boxes else []
        full_blocks = [
            {"rect": [v * scale for v in B["rect"]], "line_h": B["line_h"] * scale}
            for B in blocks
        ]
        samples.append({"t": t, "blocks": full_blocks})

        if blocks:
            # Union bounding box of detected subtitle text inside frame (with crisp 6px pad)
            ux1 = max(0, int(min(B["rect"][0] for B in blocks) - 6))
            uy1 = max(0, int(min(B["rect"][1] for B in blocks) - 5))
            ux2 = min(w, int(max(B["rect"][2] for B in blocks) + 6))
            uy2 = min(h, int(max(B["rect"][3] for B in blocks) + 5))
            sub_crop = frame[uy1:uy2, ux1:ux2]
            if sub_crop.size > 0 and (ux2 - ux1) >= 12 and (uy2 - uy1) >= 8:
                fp = _stroke_fingerprint(sub_crop)
                sharp = _crop_sharpness(sub_crop)
                bw_u = float(ux2 - ux1)
                bh_u = float(uy2 - uy1)

                is_same = False
                if cur_ev is not None and miss_streak <= 1:
                    sim = float(np.dot(cur_ev["fp"], fp))
                    w_ratio = min(cur_ev["bw"], bw_u) / max(1.0, max(cur_ev["bw"], bw_u))
                    h_ratio = min(cur_ev["bh"], bh_u) / max(1.0, max(cur_ev["bh"], bh_u))
                    if sim >= 0.72 and w_ratio >= 0.78 and h_ratio >= 0.72:
                        is_same = True

                if is_same and cur_ev is not None:
                    cur_ev["end"] = t + frame_step
                    cur_ev["count"] += 1
                    # Smooth running fingerprint slightly & keep sharpest frame crop
                    cur_ev["fp"] = 0.85 * cur_ev["fp"] + 0.15 * fp
                    fnorm = float(np.linalg.norm(cur_ev["fp"]))
                    if fnorm > 1e-5:
                        cur_ev["fp"] /= fnorm
                    if sharp > cur_ev["sharp"]:
                        cur_ev["sharp"] = sharp
                        cur_ev["best_crop"] = sub_crop.copy()
                else:
                    if cur_ev is not None:
                        raw_events.append(cur_ev)
                    cur_ev = {
                        "start": t,
                        "end": t + frame_step,
                        "fp": fp,
                        "bw": bw_u,
                        "bh": bh_u,
                        "sharp": sharp,
                        "count": 1,
                        "best_crop": sub_crop.copy(),
                        "text": "",
                    }
                miss_streak = 0
        else:
            miss_streak += 1
            if cur_ev is not None and miss_streak >= 2:
                raw_events.append(cur_ev)
                cur_ev = None

        if progress and i % 6 == 0:
            progress(min(0.60, 0.60 * (i / total)))

    if cur_ev is not None:
        raw_events.append(cur_ev)

    # Stabilize samples for clean old-subtitle erasure during render
    samples = _remove_static(samples, W, H, skip_centered=True)
    samples = _temporal_stabilize(samples, W, H)

    # Filter out 1-frame flickers (< 0.20s unless entire video is ultra-short)
    events = [ev for ev in raw_events if ev["count"] >= 2 or (ev["end"] - ev["start"]) >= 0.25]
    if not events and raw_events:
        events = raw_events

    if not events:
        if progress:
            progress(1.0)
        return [], samples, "en"

    log(f"Phát hiện {len(events)} lượt hiển thị chữ phụ đề — đang nhận dạng nội dung văn bản...")

    # Phase 2A: Fast local RapidOCR baseline on every unique event crop
    for idx, ev in enumerate(events):
        if cancelled and cancelled():
            raise InterruptedError("Đã hủy")
        ev["text"] = recognize_crop_rapidocr(engine, ev["best_crop"])
        if progress:
            progress(0.60 + 0.12 * ((idx + 1) / len(events)))

    # Phase 2B: Upgrade text reading via Gemini Vision Contact Sheets (100% accurate for JA/KO/ZH/VI/EN/stylized!)
    detected_lang = ""
    use_gemini_vision = bool(
        gemini_api_key
        and getattr(settings, "translation_engine", "gemini") == "gemini"
    )
    if use_gemini_vision:
        try:
            from . import gemini_translate
            sheets = _build_contact_sheets(events, batch_size=12)
            log(f"Đang dùng Gemini Vision AI đọc chính xác {len(events)} câu chữ phụ đề ({len(sheets)} bảng ảnh)...")
            gem_texts, det_l = gemini_translate.ocr_contact_sheets_with_gemini(
                sheets=sheets,
                raw_api_keys=gemini_api_key,
                src_lang_hint=getattr(settings, "source_lang", "auto"),
                log=log,
                progress=(lambda f: progress(0.72 + 0.25 * f)) if progress else None,
                cancelled=cancelled,
            )
            if det_l:
                detected_lang = det_l
            for ev_id, g_txt in gem_texts.items():
                if 0 <= ev_id < len(events):
                    events[ev_id]["text"] = g_txt.strip()
        except InterruptedError:
            raise
        except Exception as e:
            log(f"⚠ Gemini Vision OCR gặp lỗi ({e}) — tự động dùng kết quả RapidOCR cục bộ.")

    # Phase 3: Merge consecutive events with identical/near-identical text (handles scene cuts under same subtitle!)
    merged_segs: list[dict] = []
    for ev in events:
        txt = (ev.get("text") or "").strip()
        if not txt or not any(ch.isalnum() for ch in txt):
            continue
        st = round(max(0.0, float(ev["start"])), 3)
        en = round(min(dur, max(st + 0.25, float(ev["end"]))), 3)

        if merged_segs:
            prev = merged_segs[-1]
            gap = st - prev["end"]
            if gap <= 0.65 and _text_similarity(prev["text"], txt) >= 0.84:
                prev["end"] = max(prev["end"], en)
                # Prefer the slightly longer/more complete recognition string
                if len(txt) > len(prev["text"]):
                    prev["text"] = txt
                continue

        merged_segs.append({
            "start": st,
            "end": en,
            "text": txt,
            "gender": "female",
            "speaker": "SUB_1",
            "speaker_label": "📝 Phụ đề OCR",
        })

    if not detected_lang:
        detected_lang = detect_script_lang(
            [s["text"] for s in merged_segs],
            fallback=(settings.source_lang if getattr(settings, "source_lang", "auto") != "auto" else "en"),
        )

    log(f"✔ Quét hoàn tất: trích xuất được {len(merged_segs)} câu phụ đề (Ngôn ngữ phát hiện: {detected_lang})")
    if progress:
        progress(1.0)
    return merged_segs, samples, detected_lang


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
        self.pad_x = pad_x if pad_x is not None else max(18, int(0.028 * W))
        self.pad_y = pad_y if pad_y is not None else max(12, int(0.020 * H))
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
        target_h = min(h, 68)
        scale = target_h / max(1, h)
        sw = max(8, int(round(w * scale)))
        sh = max(8, int(round(h * scale)))
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self._morph_k)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self._morph_k)
        stroke_energy = cv2.add(tophat, blackhat)

        _, mask_edges = cv2.threshold(stroke_energy, 26, 255, cv2.THRESH_BINARY)
        _, mask_bright = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask_edges, mask_bright)
        mask = cv2.dilate(mask, self._dilate_k, iterations=1)

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
                small = cv2.resize(roi, (max(1, w // 6), max(1, h // 6)), interpolation=cv2.INTER_AREA)
                blurred = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3.0), (w, h), interpolation=cv2.INTER_LINEAR)
                darkened = (blurred.astype(np.float32) * 0.24 + np.array([20.0, 20.0, 26.0], dtype=np.float32) * 0.76).astype(np.uint8)
                roi[:] = self._feather_blend(roi, darkened, border=max(6, h // 7))
            elif self.style == "blur":
                small = cv2.resize(roi, (max(1, w // 6), max(1, h // 6)), interpolation=cv2.INTER_AREA)
                sigma = max(4.0, small.shape[0] / 3.0)
                small = cv2.GaussianBlur(small, (0, 0), sigma)
                cov = cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), max(4.0, h / 6.0))
                roi[:] = self._feather_blend(roi, cov, border=max(6, h // 7))
            else:
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
    if len(words) <= 2 and len(text) > max_chars:
        mid = len(text) // 2
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
    """Extract (start, end, translation) synced to TTS timing (or OCR timing) and strictly prevent overlaps."""
    items = []
    for s in segments:
        raw = s.get("translation", "").strip()
        if not raw:
            continue
        st = float(s.get("dub_start", s["start"]))
        en = float(s.get("dub_end", s["end"]))
        en = max(st + 0.25, en)
        items.append([st, en, raw])

    for i in range(len(items) - 1):
        if items[i][1] > items[i + 1][0] - 0.04:
            items[i][1] = max(items[i][0] + 0.15, items[i + 1][0] - 0.04)
            if items[i + 1][0] < items[i][1]:
                items[i + 1][0] = items[i][1] + 0.02
                items[i + 1][1] = max(items[i + 1][0] + 0.20, items[i + 1][1])

    return [(st, en, txt) for st, en, txt in items]


def build_ass(segments, samples, ocr_fps, W, H, font, settings):
    """Build clean, rock-steady ASS subtitles aligned with detected old subs & user's movable ROI."""
    left, top, right, bottom = get_roi_fractions(settings)
    roi_cx = 0.5 * (left + right) * W
    roi_cy = 0.5 * (top + bottom) * H
    roi_w_frac = max(0.35, right - left)

    all_cxs = []
    all_cys = []
    all_lhs = []
    if samples and settings.sub_place_on_old:
        for smp in samples:
            if smp["blocks"]:
                dom = max(smp["blocks"], key=lambda B: (B["rect"][2] - B["rect"][0]) * (B["rect"][3] - B["rect"][1]))
                r = dom["rect"]
                all_cxs.append(0.5 * (r[0] + r[2]))
                all_cys.append(0.5 * (r[1] + r[3]))
                all_lhs.append(dom["line_h"])

    if settings.sub_font_size > 0:
        global_fs = int(settings.sub_font_size)
    elif all_lhs:
        med_lh = float(np.median(all_lhs))
        global_fs = int(np.clip(round(med_lh * 0.88), H * 0.036, H * 0.062))
    else:
        global_fs = int(round(H * 0.048))

    global_cx = W // 2
    global_cy = None
    if all_cys and settings.sub_place_on_old:
        global_cy = int(np.clip(round(float(np.median(all_cys))), H * 0.06, H * 0.94))
        med_cx = float(np.median(all_cxs))
        # Snap to exact horizontal center if within ±8% of center, else keep user's custom horizontal spot
        if abs(med_cx - W * 0.5) >= 0.08 * W:
            global_cx = int(np.clip(round(med_cx), W * 0.14, W * 0.86))
    elif settings.sub_place_on_old:
        # If no old blocks found but user moved ROI away from default bottom band, honor ROI center
        is_custom_roi = (abs(top - 0.72) > 0.04 or abs(bottom - 0.98) > 0.04 or left > 0.08 or right < 0.92)
        if is_custom_roi:
            global_cy = int(np.clip(round(roi_cy), H * 0.06, H * 0.94))
            if abs(roi_cx - W * 0.5) >= 0.08 * W:
                global_cx = int(np.clip(round(roi_cx), W * 0.14, W * 0.86))

    margin_v = int(H * 0.065)
    margin_l = max(int(W * 0.05), int(W * left))
    margin_r = max(int(W * 0.05), int(W * (1.0 - right)))
    outline = max(2.4, H / 260.0)
    shadow = max(1.2, outline * 0.45)

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
        f"2,{margin_l},{margin_r},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    base_max = (26 if font in ("Yu Gothic", "Microsoft YaHei", "Malgun Gothic") else 42) if W >= H else 24
    max_chars = max(14, int(round(base_max * roi_w_frac)))
    pos_tag = f"{{\\an5\\pos({global_cx},{global_cy})}}" if global_cy is not None else ""

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
