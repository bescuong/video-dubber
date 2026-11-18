"""Speaker gender, age-register (Male / Female / Child), and Multi-Character Voice Diarization."""
from __future__ import annotations

import numpy as np
from .config import get_voices_for_lang, encode_voice_spec, LANGUAGES


def _extract_segment_features(y: np.ndarray, sr: int = 16000):
    """Extract pitch (F0), spectral brightness, and MFCC timbre vector for a speech segment."""
    import librosa

    if len(y) < int(sr * 0.28):
        return None

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    if len(rms) < 4 or float(np.max(rms)) < 1e-4:
        return None

    f0 = librosa.yin(y, fmin=65, fmax=420, sr=sr, frame_length=1024, hop_length=256)
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]

    thr = max(float(np.percentile(rms, 45)), float(np.max(rms)) * 0.15)
    voiced = (rms >= thr) & (f0 > 68) & (f0 < 410)
    if int(voiced.sum()) < 4:
        return None

    med_f0 = float(np.median(f0[voiced]))
    iqr_f0 = float(np.percentile(f0[voiced], 75) - np.percentile(f0[voiced], 25))

    # Spectral centroid (vocal brightness) & MFCC timbre fingerprint on active frames
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=256)[0][:n]
    med_cent = float(np.median(cent[voiced]))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=256)[:, :n]
    # Drop C0 (overall loudness), keep C1..C12 vocal-tract shape
    timbre = np.median(mfcc[1:, voiced], axis=1).astype(np.float32)

    return {
        "f0": med_f0,
        "iqr_f0": iqr_f0,
        "centroid": med_cent,
        "timbre": timbre,
    }


def _cluster_within_group(indices: list[int], feats: list[dict], max_speakers: int = 3,
                          dist_threshold: float = 1.15) -> dict[int, int]:
    """Cluster segment indices of the same gender into distinct character IDs (0, 1, 2...)."""
    if len(indices) <= 1:
        return {i: 0 for i in indices}

    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        # Combine normalized F0 + vocal-tract MFCCs
        mat = []
        for idx in indices:
            f = feats[idx]
            vec = np.concatenate([[f["f0"] / 40.0, f["centroid"] / 600.0], f["timbre"] / 18.0])
            mat.append(vec)
        X = np.vstack(mat).astype(np.float64)
        # Standardize features across segments if enough samples
        std = np.std(X, axis=0) + 1e-6
        X_norm = (X - np.mean(X, axis=0)) / np.maximum(std, 0.5)

        dists = pdist(X_norm, metric="cosine")
        if np.all(np.isnan(dists)) or len(dists) == 0:
            return {i: 0 for i in indices}
        dists = np.nan_to_num(dists, nan=0.0)

        Z = linkage(dists, method="average")
        # Distance threshold clustering (cosine distance ~ 0.42 separates distinct speakers well)
        raw_labels = fcluster(Z, t=0.42, criterion="distance")

        # Rank clusters by frequency so main character is always Speaker #1 (index 0)
        unique, counts = np.unique(raw_labels, return_counts=True)
        order = unique[np.argsort(-counts)]
        remap = {}
        for rank, cid in enumerate(order):
            # Merge tiny 1-line outlier clusters into nearest main speaker if too many clusters
            remap[int(cid)] = min(rank, max_speakers - 1)

        return {indices[k]: remap[int(raw_labels[k])] for k in range(len(indices))}
    except Exception:
        return {i: 0 for i in indices}


def analyze_speakers(vocals16k: np.ndarray, segments: list[dict], sr: int = 16000,
                     female_hz: float = 168.0, child_hz: float = 268.0) -> list[dict]:
    """Analyze each segment for gender/age register AND character identity.

    Returns list of dicts with keys:
      - gender: "male" | "female" | "child"
      - speaker: e.g. "NAM_1", "NU_1", "NAM_2", "TRE_1"
      - speaker_label: Vietnamese display label, e.g. "Nam 1", "Nữ 1", "Trẻ em"
      - f0: float (Hz)
    """
    n_segs = len(segments)
    if n_segs == 0:
        return []

    feats = []
    raw_genders = []
    for seg in segments:
        a = max(0, int(seg["start"] * sr))
        b = min(len(vocals16k), int(seg["end"] * sr))
        ft = _extract_segment_features(vocals16k[a:b], sr=sr)
        feats.append(ft)
        if ft is None:
            raw_genders.append(None)
        else:
            f0 = ft["f0"]
            if f0 >= child_hz and ft["centroid"] >= 2100.0:
                raw_genders.append("child")
            elif f0 >= female_hz:
                raw_genders.append("female")
            else:
                raw_genders.append("male")

    # Fill short/unvoiced segments from closest voiced neighbor
    first_valid = next((g for g in raw_genders if g is not None), "male")
    last = first_valid
    genders = []
    for g in raw_genders:
        if g is None:
            genders.append(last)
        else:
            genders.append(g)
            last = g

    # Cluster distinct characters inside male & female groups
    male_valid = [i for i in range(n_segs) if genders[i] == "male" and feats[i] is not None]
    female_valid = [i for i in range(n_segs) if genders[i] == "female" and feats[i] is not None]

    male_map = _cluster_within_group(male_valid, feats, max_speakers=3)
    female_map = _cluster_within_group(female_valid, feats, max_speakers=3)

    results = []
    last_spk_by_gender = {"male": 0, "female": 0, "child": 0}
    for i in range(n_segs):
        g = genders[i]
        f0_val = feats[i]["f0"] if feats[i] is not None else 0.0
        if g == "male":
            c_idx = male_map.get(i, last_spk_by_gender["male"])
            last_spk_by_gender["male"] = c_idx
            spk_code = f"NAM_{c_idx + 1}"
            spk_label = f"Nam {c_idx + 1}" + (" (Trầm)" if 0 < f0_val < 120 else "")
        elif g == "female":
            c_idx = female_map.get(i, last_spk_by_gender["female"])
            last_spk_by_gender["female"] = c_idx
            spk_code = f"NU_{c_idx + 1}"
            spk_label = f"Nữ {c_idx + 1}"
        else:
            spk_code = "TRE_1"
            spk_label = "Trẻ em / Cao"

        results.append({
            "gender": g,
            "speaker": spk_code,
            "speaker_label": spk_label,
            "f0": round(f0_val, 1),
        })

    return results


def guess_genders(vocals16k, segments, sr=16000, threshold_hz=168.0):
    """Backward-compatible wrapper returning simple gender strings."""
    info = analyze_speakers(vocals16k, segments, sr=sr, female_hz=threshold_hz)
    return [x["gender"] for x in info]


def resolve_voice_for_segment(seg: dict, settings) -> str:
    """Pick the best encoded voice spec ('voice_id|pitch') for a segment."""
    # 1. Explicit per-row voice override set by user in table
    row_voice = seg.get("voice_override", "").strip()
    if row_voice and row_voice != "auto":
        return row_voice

    # 2. Global single custom voice mode
    if settings.voice_mode == "custom" and settings.custom_voice:
        return settings.custom_voice

    voices = get_voices_for_lang(settings.target_lang)
    males = [v for v in voices if v.gender == "male"]
    females = [v for v in voices if v.gender == "female"]
    children = [v for v in voices if v.gender == "child"]

    lang_obj = LANGUAGES.get(settings.target_lang, LANGUAGES["vi"])
    fallback_male = encode_voice_spec(males[0]) if males else lang_obj.voice_male
    fallback_female = encode_voice_spec(females[0]) if females else lang_obj.voice_female

    # 3. Forced all-male or all-female mode
    if settings.voice_mode == "male":
        return settings.custom_voice if (settings.custom_voice and "male" in settings.custom_voice.lower()) else fallback_male
    if settings.voice_mode == "female":
        return fallback_female

    seg_gender = seg.get("gender", "male")
    spk = seg.get("speaker", "")

    # 4. Simple auto (Nam / Nữ / Trẻ em) without multi-character differentiation
    if settings.voice_mode == "auto":
        if seg_gender == "child" and children:
            return encode_voice_spec(children[0])
        return fallback_female if seg_gender == "female" else fallback_male

    # 5. Default "auto_character": assign distinct voices to NAM_1, NAM_2, NAM_3, NU_1, NU_2, TRE_1!
    if seg_gender == "child" or spk.startswith("TRE"):
        if children:
            return encode_voice_spec(children[0])
        # Fallback: pitched-up female voice
        base_f = females[0].voice_id if females else lang_obj.voice_female
        return f"{base_f}|+15Hz"

    if seg_gender == "female" or spk.startswith("NU"):
        idx = 0
        if "_" in spk:
            try:
                idx = max(0, int(spk.split("_")[1]) - 1)
            except ValueError:
                idx = 0
        if females:
            chosen = females[idx % len(females)]
            return encode_voice_spec(chosen)
        return fallback_female

    # Male characters (NAM_1, NAM_2, NAM_3...)
    idx = 0
    if "_" in spk:
        try:
            idx = max(0, int(spk.split("_")[1]) - 1)
        except ValueError:
            idx = 0
    if males:
        chosen = males[idx % len(males)]
        return encode_voice_spec(chosen)
    return fallback_male
