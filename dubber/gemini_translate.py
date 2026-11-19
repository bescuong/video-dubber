"""Context-aware Dubbing Script Translation via Google Gemini API with High-to-Low Model Cascade."""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple


# =============================================================================
# 1. EXACT PRIORITY CASCADE (HIGH -> LOW) MATCHING USER'S QUOTA DASHBOARD
# =============================================================================
# Strictly ordered from highest-capability model to lowest-tier model among
# Text-out models that have active usage quota (> 0 RPM/RPD).
GEMINI_MODEL_TIERS: List[Tuple[str, List[str]]] = [
    ("Gemini 4 Hydrogen", [
        "gemini-4-hydrogen",
        "gemini-4-hydrogen-preview",
        "gemini-4.0-hydrogen",
        "gemini-4.0-hydrogen-preview",
    ]),
    ("Gemini 3.8 Flash", [
        "gemini-3.8-flash",
        "gemini-3.8-flash-preview",
    ]),
    ("Gemini 3.6 Flash", [
        "gemini-3.6-flash",
        "gemini-3.6-flash-preview",
    ]),
    ("Gemini 3 Flash", [
        "gemini-3-flash",
        "gemini-3-flash-preview",
        "gemini-3.0-flash",
        "gemini-3.0-flash-preview",
    ]),
    ("Gemini 3.5 Flash Lite", [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite-preview",
    ]),
    ("Gemini 3.1 Flash Lite", [
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite-preview",
    ]),
    ("Gemini 2.5 Flash", [
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview",
    ]),
    ("Gemini 2.5 Flash Lite", [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite-preview",
    ]),
]

# Keywords identifying non-text-script or 0-quota models to exclude during auto-discovery
_EXCLUDED_MODEL_KEYWORDS = (
    "tts", "image", "banana", "transcribe", "embedding", "imagen",
    "veo", "lyria", "aqa", "argon", "carbon", "3.1-pro", "robotics", "computer-use",
)


# =============================================================================
# 2. DUBBING GENRE & REGISTER STYLE GUIDES
# =============================================================================
TRANSLATION_STYLES: Dict[str, Tuple[str, str]] = {
    "auto_natural": (
        "✨ Tự động chuẩn văn nói & bối cảnh (Khuyên dùng)",
        "Tự động phân tích nội dung toàn bộ hội thoại để nhận diện chính xác thể loại video, mối quan hệ nhân vật "
        "và dịch bằng khẩu ngữ tự nhiên, mượt mà, giàu cảm xúc như biên kịch lồng tiếng bản xứ."
    ),
    "movie_drama": (
        "🎬 Phim truyện / Tâm lý / Đời thường",
        "Văn phong thoại phim điện ảnh & truyền hình: câu chữ tự nhiên, giàu cảm xúc, sử dụng linh hoạt từ ngữ khẩu ngữ "
        "đời thực, xưng hô chuẩn xác theo vai vế và tình cảm giữa các nhân vật (anh-em, bố-con, cậu-tớ, mày-tao, ông-tôi...)."
    ),
    "review_recap": (
        "🎙 Review Phim / Tóm tắt cốt truyện",
        "Văn phong thuyết minh Review phim chuyên nghiệp: nhịp câu gãy gọn, súc tích, lôi cuốn, dùng từ đắt giá, "
        "kể chuyện mạch lạc, hấp dẫn người nghe từ đầu đến cuối."
    ),
    "tiktok_vlog": (
        "😂 Hài hước / Vlog / TikTok - Shorts",
        "Văn phong trẻ trung, đời thường, năng động, gần gũi mạng xã hội; dịch thoát ý tự nhiên các câu đùa, "
        "tiếng lóng (slang) và thành ngữ sang cách nói tự nhiên, duyên dáng, không dịch cứng từng chữ."
    ),
    "wuxia_historical": (
        "⚔️ Cổ trang / Tiên hiệp / Kiếm hiệp",
        "Văn phong cổ trang - kiếm hiệp - tu tiên chuẩn mực: dùng từ Hán-Việt hào sảng, trang nhã đúng ngữ cảnh; "
        "xưng hô chuẩn cổ phong (huynh - đệ - muội, tại hạ - các hạ, công tử - cô nương, bệ hạ - thần, sư phụ - đồ nhi, ngươi - ta, bản tọa...)."
    ),
    "anime_cartoon": (
        "🌸 Anime / Hoạt hình / Game",
        "Văn phong lồng tiếng Anime & Hoạt hình sống động: thể hiện rõ cá tính từng nhân vật, câu cảm thán tự nhiên, "
        "nhịp điệu biểu cảm cao, phù hợp lứa tuổi nhân vật."
    ),
    "docu_news": (
        "📚 Khoa học / Tài liệu / Tin tức / Kiến thức",
        "Văn phong thuyết minh tài liệu, giáo dục & bản tin: câu văn mạch lạc, chuẩn xác về thuật ngữ chuyên môn, "
        "lịch sự, rõ ràng, uy tín nhưng vẫn êm tai khi đọc thành tiếng."
    ),
    "sales_marketing": (
        "🛍 Quảng cáo / Review Sản phẩm / Bán hàng",
        "Văn phong giới thiệu sản phẩm cuốn hút, thuyết phục, nhấn mạnh lợi ích tự nhiên, gần gũi và tạo thiện cảm cho người xem."
    ),
}


# Cache resolved model list per API key for the current process session
_RESOLVED_MODELS_CACHE: Dict[str, List[Tuple[str, str]]] = {}


def parse_api_keys(raw_keys: str) -> List[str]:
    """Parse one or multiple Gemini API keys (separated by newline, comma, or semicolon)."""
    if not raw_keys:
        return []
    parts = re.split(r"[\s,;]+", raw_keys.strip())
    out = []
    for p in parts:
        k = p.strip()
        if len(k) >= 20 and k not in out:
            out.append(k)
    return out


def _is_excluded_model(model_id: str, display_name: str) -> bool:
    combined = f"{model_id} {display_name}".lower()
    return any(kw in combined for kw in _EXCLUDED_MODEL_KEYWORDS)


def _matches_tier_slug(model_id: str, slugs: List[str]) -> bool:
    mid = model_id.lower()
    for s in slugs:
        s_low = s.lower()
        if mid == s_low:
            return True
        if mid.startswith(s_low + "-"):
            rest = mid[len(s_low) + 1:]
            # Prevent non-lite tier (e.g. gemini-3-flash) from matching a -lite model!
            if "lite" not in s_low and rest.startswith("lite"):
                continue
            return True
    return False


def resolve_ordered_models(api_key: str, force_refresh: bool = False) -> List[Tuple[str, str]]:
    """Query Google AI Studio `/v1beta/models` to resolve exact live model IDs
    while strictly preserving the 8-tier High-to-Low priority order.

    Returns list of `(model_id, display_label)`.
    """
    if not force_refresh and api_key in _RESOLVED_MODELS_CACHE:
        return list(_RESOLVED_MODELS_CACHE[api_key])

    live_catalog: List[dict] = []
    url = f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=100&key={urllib.parse.quote(api_key)}"
    req = urllib.request.Request(url, headers={"User-Agent": "VideoDubber-Studio"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for m in data.get("models", []):
                methods = m.get("supportedGenerationMethods", []) or []
                if "generateContent" not in methods:
                    continue
                raw_name = m.get("name", "")
                mid = raw_name.split("/", 1)[1] if "/" in raw_name else raw_name
                disp = m.get("displayName", mid)
                if _is_excluded_model(mid, disp):
                    continue
                live_catalog.append({"id": mid, "display": disp})
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        if e.code in (400, 401, 403) and ("API_KEY_INVALID" in err_body or "API key not valid" in err_body):
            raise RuntimeError("Khóa Gemini API không hợp lệ. Vui lòng kiểm tra lại mã Key từ Google AI Studio.")
    except Exception:
        pass

    ordered: List[Tuple[str, str]] = []
    seen_ids = set()

    for tier_label, candidate_slugs in GEMINI_MODEL_TIERS:
        matched_id = None

        if live_catalog:
            # 1. Exact displayName match (e.g. "Gemini 4 Hydrogen", "Gemini 3.8 Flash")
            for m in live_catalog:
                if m["display"].strip().lower() == tier_label.lower():
                    matched_id = m["id"]
                    break

            # 2. Prefix/slug match in live catalog
            if not matched_id:
                for m in live_catalog:
                    if _matches_tier_slug(m["id"], candidate_slugs):
                        matched_id = m["id"]
                        break

        # 3. Fallback to canonical slug if live catalog wasn't available or didn't list preview alias
        if matched_id:
            if matched_id not in seen_ids:
                ordered.append((matched_id, tier_label))
                seen_ids.add(matched_id)
        elif not live_catalog:
            # Offline/unlisted fallback: include primary slug
            primary = candidate_slugs[0]
            if primary not in seen_ids:
                ordered.append((primary, tier_label))
                seen_ids.add(primary)

    # Safety fallback if future Google rename changed all IDs
    if not ordered:
        for tier_label, candidate_slugs in GEMINI_MODEL_TIERS:
            ordered.append((candidate_slugs[0], tier_label))

    _RESOLVED_MODELS_CACHE[api_key] = ordered
    return list(ordered)


def test_gemini_api_key(raw_keys: str) -> Tuple[bool, str, List[str]]:
    """Validate API key(s) against Google AI Studio and return resolved priority model list."""
    keys = parse_api_keys(raw_keys)
    if not keys:
        return False, "Vui lòng nhập ít nhất 1 khóa Gemini API hợp lệ (bắt đầu bằng AIza...).", []

    try:
        models = resolve_ordered_models(keys[0], force_refresh=True)
        # Perform a tiny 5-token live ping on the highest available model (cascading if needed)
        ping_ok = False
        active_first = ""
        last_err = ""
        for mid, label in models:
            ok, _, err = _call_gemini_generate(
                api_key=keys[0],
                model_id=mid,
                system_prompt="Reply OK.",
                user_prompt="Hi",
                json_mode=False,
                timeout=12,
            )
            if ok:
                ping_ok = True
                active_first = f"{label} ({mid})"
                break
            last_err = f"{label}: {err}"

        model_names = [f"{idx + 1}. {lbl}  [{mid}]" for idx, (mid, lbl) in enumerate(models)]
        if ping_ok:
            msg = (
                f"✅ Kết nối thành công ({len(keys)} Key)!\n"
                f"Model ưu tiên cao nhất đang hoạt động: {active_first}"
            )
            return True, msg, model_names
        return False, f"Key hợp lệ nhưng tất cả model đang hết hạn mức hoặc bận ({last_err}).", model_names
    except Exception as e:
        return False, str(e), []


# =============================================================================
# 3. LOW-LEVEL GEMINI GENERATE CONTENT CALL & JSON PARSER
# =============================================================================
def _call_gemini_generate(
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = True,
    timeout: int = 75,
    inline_images: Optional[List[Tuple[str, bytes]]] = None,
) -> Tuple[bool, str, str]:
    """Call Gemini REST `generateContent` (Text or Multimodal Vision). Returns `(success, response_text, error_reason)`."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model_id)}:generateContent?key={urllib.parse.quote(api_key)}"
    )
    gen_cfg: dict = {
        "temperature": 0.2 if inline_images else 0.25,
        "maxOutputTokens": 8192,
    }
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"

    user_parts: List[dict] = [{"text": user_prompt}]
    if inline_images:
        for mime_type, raw_bytes in inline_images:
            user_parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64.b64encode(raw_bytes).decode("ascii"),
                }
            })

    body = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {"role": "user", "parts": user_parts},
        ],
        "generationConfig": gen_cfg,
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "VideoDubber-Studio"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_json = json.loads(resp.read().decode("utf-8"))
            candidates = resp_json.get("candidates", [])
            if not candidates:
                block_reason = resp_json.get("promptFeedback", {}).get("blockReason", "EMPTY_CANDIDATES")
                return False, "", f"Không có phản hồi ({block_reason})"
            parts = candidates[0].get("content", {}).get("parts", [])
            # Skip thought parts if thinking model emits thought=True parts
            text_chunks = [p.get("text", "") for p in parts if not p.get("thought", False) and "text" in p]
            if not text_chunks:
                text_chunks = [p.get("text", "") for p in parts if "text" in p]
            out_txt = "".join(text_chunks).strip()
            if not out_txt:
                fin = candidates[0].get("finishReason", "EMPTY")
                return False, "", f"Phản hồi rỗng ({fin})"
            return True, out_txt, ""
    except urllib.error.HTTPError as e:
        raw_err = e.read().decode("utf-8", errors="ignore")
        try:
            ej = json.loads(raw_err)
            msg = ej.get("error", {}).get("message", raw_err[:160])
            status = ej.get("error", {}).get("status", str(e.code))
            return False, "", f"HTTP {e.code} ({status}): {msg[:140]}"
        except Exception:
            return False, "", f"HTTP {e.code}: {raw_err[:140]}"
    except Exception as e:
        return False, "", str(e)


def _extract_json_payload(raw_text: str) -> Tuple[str, Dict[int, str]]:
    """Extract `(context_memo, {id: translated_text})` from Gemini JSON output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Attempt direct JSON parse, or locate outermost {...} / [...]
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        m_obj = re.search(r"\{[\s\S]*\}", cleaned)
        m_arr = re.search(r"\[[\s\S]*\]", cleaned)
        for cand in (m_obj.group(0) if m_obj else None, m_arr.group(0) if m_arr else None):
            if cand:
                try:
                    parsed = json.loads(cand)
                    break
                except Exception:
                    continue

    if parsed is None:
        raise ValueError("Dữ liệu trả về không đúng chuẩn JSON")

    memo = ""
    items_list = []
    if isinstance(parsed, dict):
        memo = str(parsed.get("context_memo", "") or "").strip()
        items_list = parsed.get("items") or parsed.get("translations") or parsed.get("lines") or []
    elif isinstance(parsed, list):
        items_list = parsed

    out_map: Dict[int, str] = {}
    for it in items_list:
        if isinstance(it, dict) and "id" in it:
            try:
                idx = int(it["id"])
                val = str(it.get("dst") if "dst" in it else it.get("text", "")).strip()
                if val:
                    out_map[idx] = val
            except (ValueError, TypeError):
                continue
    return memo, out_map


# =============================================================================
# 4. HIGH-LEVEL CONTEXTUAL SCRIPT TRANSLATOR WITH MODEL CASCADE
# =============================================================================
def _build_dubbing_prompts(
    chunk_items: List[dict],
    src_lang_name: str,
    tgt_lang_name: str,
    style_code: str,
    custom_notes: str,
    rolling_memo: str,
) -> Tuple[str, str]:
    _, style_desc = TRANSLATION_STYLES.get(style_code, TRANSLATION_STYLES["auto_natural"])

    sys_prompt = (
        f"Bạn là Biên kịch & Đạo diễn Lồng tiếng AI hàng đầu, chuyên chuyển ngữ kịch bản video từ "
        f"{src_lang_name} sang {tgt_lang_name}.\n\n"
        f"🎯 YÊU CẦU VĂN PHONG CHỦ ĐẠO:\n"
        f"- {style_desc}\n"
        f"- Tuyệt đối KHÔNG dịch máy sát từng chữ (word-by-word). Hãy dịch thoát ý sang KHẨU NGỮ TỰ NHIÊN "
        f"mà người bản xứ {tgt_lang_name} thực sự nói ngoài đời hoặc trên phim.\n"
        f"- Đọc toàn bộ đoạn hội thoại và trường 'spk' (nhân vật phát ngôn: Nam 1, Nữ 1, Nam 2, Trẻ em...) "
        f"để xác định chính xác mối quan hệ và KHÓA CHẶT ĐẠI TỪ XƯNG HÔ xuyên suốt 100% các câu, không bao giờ "
        f"để câu trước xưng 'tôi-bạn' câu sau nhảy sang 'anh-em' vô lý.\n"
        f"- KHỚP ĐỘ DÀI LỒNG TIẾNG (Cực kỳ quan trọng): Mỗi câu có trường 'dur' (thời lượng nói tính bằng giây). "
        f"Hãy gọt giũa câu dịch cô đọng, giàu nhịp điệu để đọc ở tốc độ bình thường vừa khít thời lượng 'dur' "
        f"(tránh dịch lê thê dài gấp đôi câu gốc làm giọng đọc bị tua nhanh hoặc dính chữ).\n"
        f"- Giữ nguyên đúng số lượng câu và mã số 'id' của từng câu. Không gộp câu, không bỏ sót bất kỳ 'id' nào.\n"
    )
    if custom_notes.strip():
        sys_prompt += f"\n📌 GHI CHÚ RIÊNG CỦA NGƯỜI DÙNG (Bắt buộc tuân thủ):\n{custom_notes.strip()}\n"

    sys_prompt += (
        "\n📦 ĐỊNH DẠNG JSON ĐẦU RA BẮT BUỘC:\n"
        'Trả về duy nhất 1 JSON object hợp lệ có cấu trúc:\n'
        '{\n'
        '  "context_memo": "Tóm tắt ngắn 1-2 câu về bối cảnh & quy ước xưng hô cố định giữa các nhân vật",\n'
        '  "items": [\n'
        '    {"id": 0, "dst": "Câu dịch tự nhiên khớp thời lượng"},\n'
        '    ...\n'
        '  ]\n'
        '}'
    )

    user_parts = []
    if rolling_memo:
        user_parts.append(
            f"🔗 GHI NHỚ NGỮ CẢNH & XƯNG HÔ TỪ ĐOẠN TRƯỚC (Hãy giữ nhất quán tuyệt đối):\n{rolling_memo}\n"
        )
    user_parts.append(
        "Dịch danh sách câu thoại kịch bản sau sang JSON đúng định dạng yêu cầu:\n"
        + json.dumps(chunk_items, ensure_ascii=False)
    )
    return sys_prompt, "\n".join(user_parts)


def translate_script_with_gemini(
    segments: List[dict],
    src_lang_name: str,
    tgt_lang_name: str,
    raw_api_keys: str,
    style_code: str = "auto_natural",
    custom_notes: str = "",
    chunk_size: int = 75,
    log=print,
    progress=None,
    cancelled=None,
) -> List[str]:
    """Translate entire video script using Gemini API following High-to-Low model cascade.

    Raises RuntimeError if no API keys provided or if all Gemini keys & models fail.
    """
    keys = parse_api_keys(raw_api_keys)
    if not keys:
        raise RuntimeError(
            "Chưa có khóa Gemini API cho tài khoản này! "
            "Vui lòng bấm nút '🔑 Gemini API' trên thanh công cụ để nhập khóa API."
        )

    total = len(segments)
    if total == 0:
        return []

    # Build candidate pool: [(key_idx, api_key, model_id, model_label), ...]
    # Ordered by Model Tier FIRST (High -> Low), rotating across user keys if multiple keys given!
    key_models: List[List[Tuple[str, str]]] = [resolve_ordered_models(k) for k in keys]
    candidates: List[Tuple[int, str, str, str]] = []
    max_tiers = max(len(m) for m in key_models)
    for t_idx in range(max_tiers):
        for k_idx, k_str in enumerate(keys):
            if t_idx < len(key_models[k_idx]):
                mid, mlbl = key_models[k_idx][t_idx]
                candidates.append((k_idx, k_str, mid, mlbl))

    style_title = TRANSLATION_STYLES.get(style_code, TRANSLATION_STYLES["auto_natural"])[0]
    log(f"Đang dịch kịch bản ngữ cảnh bằng Gemini AI — Phong cách: {style_title}")

    depleted: set[Tuple[int, str]] = set()
    final_translations: List[str] = [""] * total
    rolling_memo = ""

    def _translate_batch_items(batch_items: List[dict]) -> Tuple[str, Dict[int, str]]:
        sys_p, usr_p = _build_dubbing_prompts(
            batch_items, src_lang_name, tgt_lang_name, style_code, custom_notes, rolling_memo
        )
        last_err = "Không có model khả dụng"

        for k_idx, k_str, mid, mlbl in candidates:
            if cancelled and cancelled():
                raise InterruptedError("Đã hủy")
            if (k_idx, mid) in depleted:
                continue

            key_tag = f" (Key #{k_idx + 1})" if len(keys) > 1 else ""
            ok, raw_out, err = _call_gemini_generate(k_str, mid, sys_p, usr_p, json_mode=True, timeout=80)
            if not ok:
                last_err = f"{mlbl}: {err}"
                # Mark quota-exhausted (429), missing (404), or forbidden (403) models so next chunks skip immediately
                if any(code in err for code in ("429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND", "403", "400")):
                    depleted.add((k_idx, mid))
                log(f"  ⚠ {mlbl}{key_tag} gặp lỗi/hết lượt ({err[:85]}) ➜ Tự động chuyển xuống model kế tiếp...")
                continue

            try:
                memo, out_map = _extract_json_payload(raw_out)
                # Require at least 85% of items present to accept (any rare stragglers repaired below)
                if len(out_map) < max(1, int(len(batch_items) * 0.85)):
                    raise ValueError(f"Thiếu câu dịch ({len(out_map)}/{len(batch_items)})")
                log(f"  ✔ Dịch thành công {len(batch_items)} câu bằng model: {mlbl}{key_tag} [{mid}]")
                return memo, out_map
            except Exception as parse_e:
                last_err = f"{mlbl}: Lỗi cấu trúc JSON ({parse_e})"
                log(f"  ⚠ {mlbl}{key_tag} trả về thiếu dữ liệu ({parse_e}) ➜ Tự động đổi sang model kế tiếp...")
                continue

        raise RuntimeError(f"Tất cả các model Gemini đều hết lượt hoặc gặp lỗi. Chi tiết cuối: {last_err}")

    for start_i in range(0, total, chunk_size):
        if cancelled and cancelled():
            raise InterruptedError("Đã hủy")

        end_i = min(total, start_i + chunk_size)
        batch_payload = []
        for idx in range(start_i, end_i):
            s = segments[idx]
            dur = round(max(0.3, float(s["end"]) - float(s["start"])), 1)
            spk = s.get("speaker_label") or ("Nữ" if s.get("gender") == "female" else "Nam")
            batch_payload.append({
                "id": idx,
                "spk": spk,
                "dur": dur,
                "src": s["text"].strip(),
            })

        memo, translated_map = _translate_batch_items(batch_payload)
        if memo:
            rolling_memo = memo

        # Check if any rare single ID was missed by the LLM and repair cleanly
        missing_payload = [it for it in batch_payload if it["id"] not in translated_map]
        if missing_payload:
            try:
                _, repair_map = _translate_batch_items(missing_payload)
                translated_map.update(repair_map)
            except Exception:
                pass

        for it in batch_payload:
            idx = it["id"]
            final_translations[idx] = translated_map.get(idx, it["src"]).strip()

        if progress:
            progress(min(1.0, end_i / total))

        # Gentle 1.0s pacing between large chunks to respect 5 RPM on top-tier models
        if end_i < total:
            time.sleep(1.0)

    return final_translations


# =============================================================================
# 5. MULTIMODAL GEMINI VISION CONTACT-SHEET OCR (FOR BURNED-IN VIDEO SUBTITLES)
# =============================================================================
def _extract_ocr_json(raw_text: str) -> Tuple[str, Dict[int, str]]:
    """Parse Gemini Vision OCR response JSON into `(detected_lang, {id: text})`."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        m_obj = re.search(r"\{[\s\S]*\}", cleaned)
        m_arr = re.search(r"\[[\s\S]*\]", cleaned)
        for cand in (m_obj.group(0) if m_obj else None, m_arr.group(0) if m_arr else None):
            if cand:
                try:
                    parsed = json.loads(cand)
                    break
                except Exception:
                    continue

    if parsed is None:
        raise ValueError("Phản hồi OCR không đúng chuẩn JSON")

    det_lang = ""
    items_list = []
    if isinstance(parsed, dict):
        det_lang = str(parsed.get("detected_lang", "") or "").strip().lower()
        items_list = parsed.get("items") or parsed.get("subtitles") or parsed.get("lines") or []
    elif isinstance(parsed, list):
        items_list = parsed

    out_map: Dict[int, str] = {}
    for it in items_list:
        if isinstance(it, dict) and "id" in it:
            try:
                idx = int(it["id"])
                val = str(it.get("text") if "text" in it else it.get("src", "")).strip()
                out_map[idx] = val
            except (ValueError, TypeError):
                continue
    return det_lang, out_map


def ocr_contact_sheets_with_gemini(
    sheets: List[Tuple[bytes, List[int]]],
    raw_api_keys: str,
    src_lang_hint: str = "auto",
    log=print,
    progress=None,
    cancelled=None,
) -> Tuple[Dict[int, str], str]:
    """Read burned-in subtitle crops stitched into numbered contact-sheet JPEGs using Gemini Vision cascade.

    Args:
        sheets: list of `(jpeg_bytes, list_of_row_ids)`
    Returns:
        `(id_to_text_map, detected_iso_lang)`
    """
    keys = parse_api_keys(raw_api_keys)
    if not keys or not sheets:
        return {}, ""

    key_models: List[List[Tuple[str, str]]] = [resolve_ordered_models(k) for k in keys]
    candidates: List[Tuple[int, str, str, str]] = []
    max_tiers = max(len(m) for m in key_models)
    for t_idx in range(max_tiers):
        for k_idx, k_str in enumerate(keys):
            if t_idx < len(key_models[k_idx]):
                mid, mlbl = key_models[k_idx][t_idx]
                candidates.append((k_idx, k_str, mid, mlbl))

    hint_line = (
        f"Ngôn ngữ phụ đề dự kiến: {src_lang_hint}."
        if src_lang_hint and src_lang_hint != "auto"
        else "Tự động nhận diện chính xác ngôn ngữ hiển thị (Trung, Nhật, Hàn, Anh, Việt, Thái, v.v.)."
    )
    sys_prompt = (
        "Bạn là hệ thống OCR đọc phụ đề video chính xác tuyệt đối.\n"
        "Ảnh đính kèm là bảng ghép nhiều hàng ảnh cắt phụ đề từ video. "
        "Bên trái mỗi hàng có nhãn số màu vàng dạng `#0`, `#1`, `#2`...\n"
        f"{hint_line}\n"
        "NHIỆM VỤ:\n"
        "- Đọc NGUYÊN VĂN 100% nội dung chữ phụ đề xuất hiện trên từng hàng (KHÔNG dịch, KHÔNG bịa thêm chữ).\n"
        "- Bỏ qua logo/watermark nhỏ không phải câu phụ đề; nếu hàng nào chỉ là hình ảnh nhiễu không có chữ rõ ràng thì trả text là chuỗi rỗng \"\".\n"
        "- Nếu 1 hàng có 2 dòng chữ phụ đề, nối gọn bằng 1 khoảng trắng (hoặc giữ liền mạch với tiếng Trung/Nhật).\n"
        "- Trả về DUY NHẤT 1 JSON hợp lệ:\n"
        '{\n'
        '  "detected_lang": "mã ISO-639-1 như zh, ja, ko, en, vi, th, fr, ru...",\n'
        '  "items": [{"id": 0, "text": "nguyên văn dòng #0"}, ...]\n'
        '}'
    )

    depleted: set[Tuple[int, str]] = set()
    all_texts: Dict[int, str] = {}
    detected_langs: List[str] = []
    total_sheets = len(sheets)

    for s_idx, (jpeg_bytes, row_ids) in enumerate(sheets):
        if cancelled and cancelled():
            raise InterruptedError("Đã hủy")

        usr_prompt = (
            f"Hãy đọc nguyên văn phụ đề của {len(row_ids)} hàng ảnh có mã số id = {row_ids} trong ảnh đính kèm "
            f"và trả về đúng JSON."
        )
        sheet_ok = False
        last_err = ""

        for k_idx, k_str, mid, mlbl in candidates:
            if cancelled and cancelled():
                raise InterruptedError("Đã hủy")
            if (k_idx, mid) in depleted:
                continue

            key_tag = f" (Key #{k_idx + 1})" if len(keys) > 1 else ""
            ok, raw_out, err = _call_gemini_generate(
                k_str,
                mid,
                sys_prompt,
                usr_prompt,
                json_mode=True,
                timeout=85,
                inline_images=[("image/jpeg", jpeg_bytes)],
            )
            if not ok:
                last_err = f"{mlbl}: {err}"
                if any(code in err for code in ("429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND", "403", "400")):
                    depleted.add((k_idx, mid))
                log(f"  ⚠ OCR Vision ({mlbl}{key_tag}) lỗi/hết lượt ({err[:75]}) ➜ Chuyển model kế tiếp...")
                continue

            try:
                d_lang, out_map = _extract_ocr_json(raw_out)
                if len(out_map) < max(1, int(len(row_ids) * 0.5)):
                    raise ValueError(f"Thiếu hàng OCR ({len(out_map)}/{len(row_ids)})")
                all_texts.update(out_map)
                if d_lang:
                    detected_langs.append(d_lang)
                log(f"  ✔ Đọc chữ phụ đề Vision AI nhóm {s_idx + 1}/{total_sheets} ({len(row_ids)} câu) bằng {mlbl}{key_tag}")
                sheet_ok = True
                break
            except Exception as parse_e:
                last_err = f"{mlbl}: {parse_e}"
                log(f"  ⚠ OCR Vision ({mlbl}{key_tag}) lỗi định dạng ({parse_e}) ➜ Đổi model...")
                continue

        if not sheet_ok:
            log(f"  ⚠ Không thể đọc nhóm ảnh {s_idx + 1}/{total_sheets} bằng Gemini Vision ({last_err}) — dùng kết quả RapidOCR dự phòng.")

        if progress:
            progress(min(1.0, (s_idx + 1) / total_sheets))

        if s_idx + 1 < total_sheets:
            time.sleep(0.8)

    best_lang = ""
    if detected_langs:
        # Most frequent detected language across sheets
        best_lang = max(set(detected_langs), key=detected_langs.count)
    return all_texts, best_lang
