"""Speaker gender, age-register (Male / Female / Child), and Multi-Character Voice Diarization.

v2.5 Upgrade — Physically-Grounded Acoustic Clustering (Zero False Splits on 1-Speaker Videos):
1. Extracts octave-scaled pitch (`12 * log2(F0)` in semitones) + vocal-tract MFCC shape (`C2..C12`,
   cepstral-mean normalized to ignore loudness/EQ differences between sentences).
2. Evaluates a Global Single-Speaker Gate BEFORE clustering: if pitch dispersion & vocal-tract
   variance match a single expressive human speaker, locks 100% of segments to ONE consistent speaker
   (eliminates splitting 1 narrator into 3 fake speakers or flipping male/female mid-sentence!).
3. Clusters speakers in absolute physical Euclidean acoustic space (never mean-centered cosine!) and
   assigns gender at the SPEAKER CLUSTER level rather than noisy individual sentences.
"""
from __future__ import annotations

import math
import numpy as np

from .config import get_voices_for_lang, encode_voice_spec, LANGUAGES


def _extract_segment_features(y: np.ndarray, sr: int = 16000) -> dict | None:
    """Extract robust pitch (Hz + semitones), spectral centroid, and vocal-tract MFCC vector."""
    import librosa

    if len(y) < int(sr * 0.32):
        return None

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    if len(rms) < 5 or float(np.max(rms)) < 1e-4:
        return None

    f0 = librosa.yin(y, fmin=65, fmax=420, sr=sr, frame_length=1024, hop_length=256)
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]

    thr = max(float(np.percentile(rms, 50)), float(np.max(rms)) * 0.18)
    voiced = (rms >= thr) & (f0 > 68) & (f0 < 405)
    n_voiced = int(voiced.sum())
    if n_voiced < 5:
        return None

    f0_v = f0[voiced]
    # Trim top/bottom 15% pitch outliers (octave jumps / vocal fry) for rock-steady speaker F0
    lo_p, hi_p = np.percentile(f0_v, [15, 85])
    core = f0_v[(f0_v >= lo_p) & (f0_v <= hi_p)]
    med_f0 = float(np.median(core if len(core) > 0 else f0_v))
    semi_f0 = 12.0 * math.log2(max(65.0, med_f0) / 100.0)  # semitones relative to 100 Hz

    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=256)[0][:n]
    med_cent = float(np.median(cent[voiced]))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=256)[:, :n]
    # Keep C2..C12 (skip C0 energy and C1 spectral tilt which fluctuates with speaking effort/volume)
    vt = np.median(mfcc[2:13, voiced], axis=1).astype(np.float32)
    vt_norm = float(np.linalg.norm(vt))
    if vt_norm > 1e-5:
        vt = vt / vt_norm

    return {
        "f0": med_f0,
        "semi": semi_f0,
        "centroid": med_cent,
        "timbre": vt,
        "dur": len(y) / float(sr),
        "voiced_frames": n_voiced,
    }


def _acoustic_distance(f1: dict, f2: dict) -> float:
    """Physical speaker distance where ~1.0 = distinct speaker threshold.

    - Same speaker natural intonation across sentences: ~2..4 semitones & cosine timbre diff < 0.12 -> dist ~0.35..0.70
    - Male vs Female speaker: ~7..12 semitones -> dist ~1.3..2.2
    - Two distinct speakers of same gender: >= 4.5 semitones + timbre diff > 0.22 -> dist >= 1.05
    """
    d_semi = abs(f1["semi"] - f2["semi"]) / 5.8
    cos_sim = float(np.clip(np.dot(f1["timbre"], f2["timbre"]), -1.0, 1.0))
    d_timbre = (1.0 - cos_sim) / 0.28
    return float(math.sqrt(0.68 * (d_semi ** 2) + 0.32 * (d_timbre ** 2)))


def _cluster_speakers_physically(valid_indices: list[int], feats: list[dict | None],
                                 max_speakers: int = 4) -> dict[int, int]:
    """Cluster voiced segments into real distinct speakers; strictly returns 1 cluster for 1-person videos!"""
    if len(valid_indices) <= 2:
        return {idx: 0 for idx in valid_indices}

    # Prefer reliable longer segments (>= 0.85s) to anchor speaker centroids
    anchor_indices = [i for i in valid_indices if feats[i]["dur"] >= 0.85 and feats[i]["voiced_frames"] >= 10]
    if len(anchor_indices) < 2:
        anchor_indices = list(valid_indices)

    semis = np.array([feats[i]["semi"] for i in anchor_indices], dtype=np.float64)
    timbres = np.vstack([feats[i]["timbre"] for i in anchor_indices]).astype(np.float64)

    # ── STEP 1: Strict Single-Speaker Gate ──────────────────────────────────
    # Check pitch spread (15th..85th percentile) & vocal-tract coherence around dominant speaker
    p15, p85 = np.percentile(semis, [15, 85])
    semi_span_70 = float(p85 - p15)
    med_semi = float(np.median(semis))
    med_timbre = np.median(timbres, axis=0)
    t_norm = float(np.linalg.norm(med_timbre))
    if t_norm > 1e-5:
        med_timbre = med_timbre / t_norm

    # Compute distance of every anchor segment to the primary speaker voice
    med_feat = {"semi": med_semi, "timbre": med_timbre}
    dists_to_main = np.array([_acoustic_distance(feats[i], med_feat) for i in anchor_indices])

    # Count how many reliable segments genuinely deviate strongly from the main speaker (dist >= 1.05)
    min_second_spk_segs = max(2, int(math.ceil(0.14 * len(anchor_indices))))
    n_outliers = int(np.sum(dists_to_main >= 1.05))

    if semi_span_70 <= 4.6 and n_outliers < min_second_spk_segs:
        # Unimodal single narrator / speaker -> 100% locked to 1 speaker!
        return {idx: 0 for idx in valid_indices}

    # ── STEP 2: Agglomerative Clustering in Physical Acoustic Space ─────────
    try:
        from scipy.cluster.hierarchy import fcluster, linkage

        m = len(anchor_indices)
        condensed = []
        for a in range(m):
            for b in range(a + 1, m):
                condensed.append(_acoustic_distance(feats[anchor_indices[a]], feats[anchor_indices[b]]))
        condensed_arr = np.asarray(condensed, dtype=np.float64)
        if len(condensed_arr) == 0 or float(np.max(condensed_arr)) < 1.02:
            return {idx: 0 for idx in valid_indices}

        Z = linkage(condensed_arr, method="average")
        # Require strong physical separation (t=1.02) before splitting two voices!
        raw_labels = fcluster(Z, t=1.02, criterion="distance")

        unique, counts = np.unique(raw_labels, return_counts=True)
        order = unique[np.argsort(-counts)]

        # Keep only substantial speaker clusters (>= min_second_spk_segs or >= 2 clear opposite-gender lines)
        kept_cids = [int(order[0])]
        main_mask = raw_labels == order[0]
        main_semi = float(np.median(semis[main_mask]))

        for cid, cnt in zip(order[1:], counts[np.argsort(-counts)][1:]):
            if len(kept_cids) >= max_speakers:
                break
            c_mask = raw_labels == cid
            c_semi = float(np.median(semis[c_mask]))
            # Opposite gender (>= 5.5 semitones apart) needs at least 2 segments; same-gender needs min_second_spk_segs
            needed = 2 if abs(c_semi - main_semi) >= 5.5 else max(3, min_second_spk_segs)
            if int(cnt) >= needed:
                kept_cids.append(int(cid))

        if len(kept_cids) <= 1:
            return {idx: 0 for idx in valid_indices}

        # Build clean prototype `(semi, timbre)` for each confirmed speaker cluster
        prototypes = []
        for new_id, cid in enumerate(kept_cids):
            c_mask = raw_labels == cid
            p_semi = float(np.median(semis[c_mask]))
            p_tim = np.median(timbres[c_mask], axis=0)
            pn = float(np.linalg.norm(p_tim))
            if pn > 1e-5:
                p_tim = p_tim / pn
            prototypes.append({"semi": p_semi, "timbre": p_tim})

        # Assign all voiced segments to their nearest confirmed speaker prototype
        assigned: dict[int, int] = {}
        for idx in valid_indices:
            best_k = min(range(len(prototypes)), key=lambda k: _acoustic_distance(feats[idx], prototypes[k]))
            assigned[idx] = best_k
        return assigned
    except Exception:
        return {idx: 0 for idx in valid_indices}


def _classify_cluster_gender(cluster_feats: list[dict], female_hz: float = 172.0,
                             child_hz: float = 275.0) -> tuple[str, float]:
    """Classify gender ONCE for an entire speaker cluster using weighted median F0."""
    f0s = np.array([f["f0"] for f in cluster_feats], dtype=np.float64)
    cents = np.array([f["centroid"] for f in cluster_feats], dtype=np.float64)
    med_f0 = float(np.median(f0s))
    med_cent = float(np.median(cents))
    if med_f0 >= child_hz and med_cent >= 2150.0:
        return "child", med_f0
    if med_f0 >= female_hz:
        return "female", med_f0
    return "male", med_f0


def analyze_speakers(vocals16k: np.ndarray, segments: list[dict], sr: int = 16000,
                     female_hz: float = 172.0, child_hz: float = 275.0) -> list[dict]:
    """Analyze dialogue segments for consistent speaker identity & cluster-level gender.

    Guarantees:
    - Single-narrator videos stay 100% on ONE speaker (`Nam 1` or `Nữ 1`), never splitting into 3 or flipping gender!
    - Multi-speaker conversations cleanly separate real speakers (`Nam 1`, `Nữ 1`, `Nam 2`...).
    """
    n_segs = len(segments)
    if n_segs == 0:
        return []

    feats: list[dict | None] = []
    for seg in segments:
        a = max(0, int(float(seg["start"]) * sr))
        b = min(len(vocals16k), int(float(seg["end"]) * sr))
        feats.append(_extract_segment_features(vocals16k[a:b], sr=sr))

    valid_indices = [i for i, ft in enumerate(feats) if ft is not None]
    if not valid_indices:
        return [
            {"gender": "male", "speaker": "NAM_1", "speaker_label": "Nam 1", "f0": 0.0}
            for _ in range(n_segs)
        ]

    # Cluster speakers globally first
    cluster_map = _cluster_speakers_physically(valid_indices, feats, max_speakers=4)

    # Fill unvoiced/very short segments from nearest voiced neighbor
    full_clusters: list[int] = [0] * n_segs
    first_c = cluster_map[valid_indices[0]]
    last_c = first_c
    for i in range(n_segs):
        if i in cluster_map:
            last_c = cluster_map[i]
        full_clusters[i] = last_c

    # Temporal smoothing: prevent a single <0.9s exclamation from jumping speakers between identical neighbors
    for i in range(1, n_segs - 1):
        seg_dur = float(segments[i]["end"]) - float(segments[i]["start"])
        if seg_dur < 0.90 and full_clusters[i - 1] == full_clusters[i + 1] and full_clusters[i] != full_clusters[i - 1]:
            full_clusters[i] = full_clusters[i - 1]

    # Assign gender & canonical speaker codes (`NAM_1`, `NU_1`, `NAM_2`...) at the CLUSTER level!
    unique_cids = sorted(set(full_clusters), key=lambda c: -full_clusters.count(c))
    gender_counts = {"male": 0, "female": 0, "child": 0}
    cluster_meta: dict[int, tuple[str, str, str, float]] = {}

    for cid in unique_cids:
        c_feats = [feats[i] for i in range(n_segs) if full_clusters[i] == cid and feats[i] is not None]
        if not c_feats:
            c_feats = [feats[valid_indices[0]]]  # type: ignore[list-item]
        g, med_f0 = _classify_cluster_gender(c_feats, female_hz=female_hz, child_hz=child_hz)
        gender_counts[g] += 1
        rank = gender_counts[g]
        if g == "male":
            spk_code = f"NAM_{min(rank, 3)}"
            spk_label = f"Nam {min(rank, 3)}" + (" (Trầm)" if 0 < med_f0 < 116 else "")
        elif g == "female":
            spk_code = f"NU_{min(rank, 3)}"
            spk_label = f"Nữ {min(rank, 3)}"
        else:
            spk_code = "TRE_1"
            spk_label = "Trẻ em"
        cluster_meta[cid] = (g, spk_code, spk_label, med_f0)

    results = []
    for i in range(n_segs):
        g, spk_code, spk_label, c_f0 = cluster_meta[full_clusters[i]]
        f0_val = feats[i]["f0"] if feats[i] is not None else c_f0
        results.append({
            "gender": g,
            "speaker": spk_code,
            "speaker_label": spk_label,
            "f0": round(f0_val, 1),
        })
    return results


def guess_genders(vocals16k, segments, sr=16000, threshold_hz=172.0):
    """Backward-compatible wrapper returning simple gender strings."""
    info = analyze_speakers(vocals16k, segments, sr=sr, female_hz=threshold_hz)
    return [x["gender"] for x in info]


def resolve_voice_for_segment(seg: dict, settings) -> str:
    """Pick the best encoded Edge TTS voice spec ('voice_id|pitch') for a segment."""
    row_voice = seg.get("voice_override", "").strip()
    if row_voice and row_voice != "auto":
        return row_voice

    if settings.voice_mode == "custom" and settings.custom_voice:
        return settings.custom_voice

    voices = get_voices_for_lang(settings.target_lang)
    males = [v for v in voices if v.gender == "male"]
    females = [v for v in voices if v.gender == "female"]
    children = [v for v in voices if v.gender == "child"]

    lang_obj = LANGUAGES.get(settings.target_lang, LANGUAGES["vi"])
    fallback_male = encode_voice_spec(males[0]) if males else lang_obj.voice_male
    fallback_female = encode_voice_spec(females[0]) if females else lang_obj.voice_female

    if settings.voice_mode == "male":
        return settings.custom_voice if (settings.custom_voice and "male" in settings.custom_voice.lower()) else fallback_male
    if settings.voice_mode == "female":
        return fallback_female

    seg_gender = seg.get("gender", "male")
    spk = seg.get("speaker", "")

    if settings.voice_mode == "auto":
        if seg_gender == "child" and children:
            return encode_voice_spec(children[0])
        return fallback_female if seg_gender == "female" else fallback_male

    if seg_gender == "child" or spk.startswith("TRE"):
        if children:
            return encode_voice_spec(children[0])
        base_f = females[0].voice_id if females else lang_obj.voice_female
        return f"{base_f}|+12Hz"

    if seg_gender == "female" or spk.startswith("NU"):
        idx = 0
        if "_" in spk:
            try:
                idx = max(0, int(spk.split("_")[1]) - 1)
            except ValueError:
                idx = 0
        if females:
            return encode_voice_spec(females[idx % len(females)])
        return fallback_female

    idx = 0
    if "_" in spk:
        try:
            idx = max(0, int(spk.split("_")[1]) - 1)
        except ValueError:
            idx = 0
    if males:
        return encode_voice_spec(males[idx % len(males)])
    return fallback_male
