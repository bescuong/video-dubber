"""Language table, server-verified Microsoft Edge Neural Voice catalog, and default settings."""
from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass(frozen=True)
class VoiceOption:
    voice_id: str      # Verified Edge TTS neural voice ShortName
    name: str          # Display label in Vietnamese
    gender: str        # "male" | "female" | "child"
    region: str        # Country / accent tag
    pitch: str = "+0Hz"  # Optional prosody pitch adjustment for character variation
    rate_bias: int = 0   # Optional subtle rate offset (%)


@dataclass(frozen=True)
class Language:
    code: str          # whisper / ISO code (or locale like en-GB)
    whisper_code: str  # base whisper ISO-639-1 code
    name: str          # display name
    nllb: str          # NLLB-200 language code
    voice_male: str    # Primary default male Edge TTS voice
    voice_female: str  # Primary default female Edge TTS voice
    font: str          # Subtitle font able to render this script


# Priority & popular languages with distinct US vs UK English
LANGUAGES: Dict[str, Language] = {
    "vi": Language("vi", "vi", "🇻🇳 Tiếng Việt", "vie_Latn", "vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural", "Arial"),
    "en": Language("en", "en", "🇺🇸 Tiếng Anh - Mỹ (US)", "eng_Latn", "en-US-AndrewMultilingualNeural", "en-US-AvaMultilingualNeural", "Arial"),
    "en-GB": Language("en-GB", "en", "🇬🇧 Tiếng Anh - Anh (UK)", "eng_Latn", "en-GB-RyanNeural", "en-GB-SoniaNeural", "Arial"),
    "ko": Language("ko", "ko", "🇰🇷 Tiếng Hàn (Korean)", "kor_Hang", "ko-KR-InJoonNeural", "ko-KR-SunHiNeural", "Malgun Gothic"),
    "ja": Language("ja", "ja", "🇯🇵 Tiếng Nhật (Japanese)", "jpn_Jpan", "ja-JP-KeitaNeural", "ja-JP-NanamiNeural", "Yu Gothic"),
    "zh": Language("zh", "zh", "🇨🇳 Tiếng Trung (Chinese)", "zho_Hans", "zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural", "Microsoft YaHei"),
    "th": Language("th", "th", "🇹🇭 Tiếng Thái (Thai)", "tha_Thai", "th-TH-NiwatNeural", "th-TH-PremwadeeNeural", "Tahoma"),
    "id": Language("id", "id", "🇮🇩 Tiếng Indonesia", "ind_Latn", "id-ID-ArdiNeural", "id-ID-GadisNeural", "Arial"),
    "fr": Language("fr", "fr", "🇫🇷 Tiếng Pháp (French)", "fra_Latn", "fr-FR-HenriNeural", "fr-FR-DeniseNeural", "Arial"),
    "de": Language("de", "de", "🇩🇪 Tiếng Đức (German)", "deu_Latn", "de-DE-ChristophNeural", "de-DE-TanjaNeural", "Arial"),
    "es": Language("es", "es", "🇪🇸 Tiếng Tây Ban Nha", "spa_Latn", "es-ES-AlvaroNeural", "es-ES-ElviraNeural", "Arial"),
    "pt": Language("pt", "pt", "🇧🇷 Tiếng Bồ Đào Nha", "por_Latn", "pt-BR-AntonioNeural", "pt-BR-FranciscaNeural", "Arial"),
    "it": Language("it", "it", "🇮🇹 Tiếng Ý (Italian)", "ita_Latn", "it-IT-DiegoNeural", "it-IT-ElsaNeural", "Arial"),
    "ru": Language("ru", "ru", "🇷🇺 Tiếng Nga (Russian)", "rus_Cyrl", "ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural", "Arial"),
    "hi": Language("hi", "hi", "🇮🇳 Tiếng Ấn Độ (Hindi)", "hin_Deva", "hi-IN-MadhurNeural", "hi-IN-SwaraNeural", "Nirmala UI"),
}


# 100% Live Server-Verified Neural Voices per language (multi-character ready)
VOICE_CATALOG: Dict[str, List[VoiceOption]] = {
    "vi": [
        VoiceOption("vi-VN-HoaiMyNeural", "Hoài My — Nữ chuẩn, tự nhiên & truyền cảm", "female", "Việt Nam", "+0Hz"),
        VoiceOption("vi-VN-NamMinhNeural", "Nam Minh — Nam chuẩn, ấm áp & rõ ràng", "male", "Việt Nam", "+0Hz"),
        VoiceOption("vi-VN-NamMinhNeural", "Nam Minh (Trầm) — Nam trầm thuyết minh điện ảnh", "male", "Việt Nam", "-7Hz"),
        VoiceOption("vi-VN-HoaiMyNeural", "Hoài My (Trẻ) — Nữ trẻ trung, trong trẻo", "female", "Việt Nam", "+6Hz"),
        VoiceOption("vi-VN-NamMinhNeural", "Nam Minh (Thanh niên) — Nam năng động, vlog", "male", "Việt Nam", "+5Hz"),
        VoiceOption("vi-VN-HoaiMyNeural", "Bé Mai — Giọng cao Thiếu nhi / Hoạt hình", "child", "Việt Nam", "+15Hz"),
    ],
    "en": [
        VoiceOption("en-US-AndrewMultilingualNeural", "Andrew HD — Nam Mỹ tự nhiên nhất, ấm áp (Đề xuất)", "male", "Mỹ (US)"),
        VoiceOption("en-US-AvaMultilingualNeural", "Ava HD — Nữ Mỹ hiện đại, truyền cảm (Đề xuất)", "female", "Mỹ (US)"),
        VoiceOption("en-US-BrianMultilingualNeural", "Brian HD — Nam Mỹ trẻ trung, review/vlog", "male", "Mỹ (US)"),
        VoiceOption("en-US-EmmaMultilingualNeural", "Emma HD — Nữ Mỹ dịu dàng, kể chuyện", "female", "Mỹ (US)"),
        VoiceOption("en-US-ChristopherNeural", "Christopher — Nam Mỹ trầm, thuyết minh phim", "male", "Mỹ (US)"),
        VoiceOption("en-US-RogerNeural", "Roger — Nam Mỹ chững chạc, lịch lãm", "male", "Mỹ (US)"),
        VoiceOption("en-US-EricNeural", "Eric — Nam Mỹ điềm đạm, sắc nét", "male", "Mỹ (US)"),
        VoiceOption("en-US-SteffanNeural", "Steffan — Nam Mỹ tin tức, thuyết trình", "male", "Mỹ (US)"),
        VoiceOption("en-US-GuyNeural", "Guy — Nam Mỹ phát thanh viên chuẩn", "male", "Mỹ (US)"),
        VoiceOption("en-US-AriaNeural", "Aria — Nữ Mỹ giàu cảm xúc, lồng tiếng phim", "female", "Mỹ (US)"),
        VoiceOption("en-US-JennyNeural", "Jenny — Nữ Mỹ thân thiện, tự nhiên", "female", "Mỹ (US)"),
        VoiceOption("en-US-MichelleNeural", "Michelle — Nữ Mỹ thanh lịch, nhẹ nhàng", "female", "Mỹ (US)"),
        VoiceOption("en-US-AnaNeural", "Ana — Giọng bé gái Mỹ (Thiếu nhi)", "child", "Mỹ (US)"),
    ],
    "en-GB": [
        VoiceOption("en-GB-RyanNeural", "Ryan — Nam Anh Quốc lịch lãm, chuẩn Hoàng gia (Đề xuất)", "male", "Anh (UK)"),
        VoiceOption("en-GB-SoniaNeural", "Sonia — Nữ Anh Quốc chuẩn BBC, sang trọng (Đề xuất)", "female", "Anh (UK)"),
        VoiceOption("en-GB-ThomasNeural", "Thomas — Nam Anh trầm ấm, kể chuyện/tài liệu", "male", "Anh (UK)"),
        VoiceOption("en-GB-LibbyNeural", "Libby — Nữ Anh trẻ trung, tự nhiên", "female", "Anh (UK)"),
        VoiceOption("en-AU-WilliamMultilingualNeural", "William HD — Nam Anh-Úc hiện đại, tự nhiên", "male", "Anh-Úc"),
        VoiceOption("en-GB-MaisieNeural", "Maisie — Giọng bé gái Anh Quốc (Thiếu nhi)", "child", "Anh (UK)"),
    ],
    "ko": [
        VoiceOption("ko-KR-SunHiNeural", "Sun-Hi (선희) — Nữ Hàn chuẩn, trong trẻo (Đề xuất)", "female", "Hàn Quốc"),
        VoiceOption("ko-KR-InJoonNeural", "In-Joon (인준) — Nam Hàn ấm áp, tự nhiên (Đề xuất)", "male", "Hàn Quốc"),
        VoiceOption("ko-KR-HyunsuMultilingualNeural", "Hyun-Su HD (현수) — Nam Hàn hiện đại, truyền cảm cao", "male", "Hàn Quốc"),
        VoiceOption("en-US-AvaMultilingualNeural", "Ji-woo HD — Nữ Hàn trẻ trung, tự nhiên đa cảm xúc", "female", "Hàn Quốc"),
        VoiceOption("en-US-EmmaMultilingualNeural", "Soo-jin HD — Nữ Hàn dịu dàng, kể chuyện phim", "female", "Hàn Quốc"),
        VoiceOption("ko-KR-InJoonNeural", "In-Joon (Trầm) — Nam Hàn trầm ổn, điện ảnh", "male", "Hàn Quốc", "-6Hz"),
        VoiceOption("ko-KR-SunHiNeural", "Seo-yeon — Giọng thiếu nữ / Trẻ em Hàn Quốc", "child", "Hàn Quốc", "+13Hz"),
    ],
    "ja": [
        VoiceOption("ja-JP-NanamiNeural", "Nanami (七海) — Nữ Nhật chuẩn, trong sáng (Đề xuất)", "female", "Nhật Bản"),
        VoiceOption("ja-JP-KeitaNeural", "Keita (圭太) — Nam Nhật trẻ trung, rõ ràng (Đề xuất)", "male", "Nhật Bản"),
        VoiceOption("en-US-AndrewMultilingualNeural", "Kenji HD — Nam Nhật trầm ấm, tự nhiên", "male", "Nhật Bản"),
        VoiceOption("en-US-AvaMultilingualNeural", "Yuki HD — Nữ Nhật hiện đại, truyền cảm", "female", "Nhật Bản"),
        VoiceOption("en-US-BrianMultilingualNeural", "Haruto HD — Nam Nhật năng động, đời thường", "male", "Nhật Bản"),
        VoiceOption("en-US-EmmaMultilingualNeural", "Mio HD — Nữ Nhật dịu dàng, nhẹ nhàng", "female", "Nhật Bản"),
        VoiceOption("ja-JP-KeitaNeural", "Daiki (Trầm) — Nam Nhật trầm hùng, thuyết minh", "male", "Nhật Bản", "-6Hz"),
        VoiceOption("ja-JP-NanamiNeural", "Hina — Giọng thiếu nữ / Anime Nhật Bản", "child", "Nhật Bản", "+13Hz"),
    ],
    "zh": [
        VoiceOption("zh-CN-XiaoxiaoNeural", "Xiaoxiao (晓晓) — Nữ Trung tự nhiên, đa cảm xúc (Đề xuất)", "female", "Trung Quốc"),
        VoiceOption("zh-CN-YunxiNeural", "Yunxi (云希) — Nam Trung trẻ, review phim cực chuẩn (Đề xuất)", "male", "Trung Quốc"),
        VoiceOption("zh-CN-YunjianNeural", "Yunjian (云健) — Nam Trung trầm hùng, phim hành động", "male", "Trung Quốc"),
        VoiceOption("zh-CN-YunyangNeural", "Yunyang (云扬) — Nam Trung chuẩn phát thanh viên", "male", "Trung Quốc"),
        VoiceOption("zh-CN-XiaoyiNeural", "Xiaoyi (晓伊) — Nữ Trung hoạt bát, ngọt ngào", "female", "Trung Quốc"),
        VoiceOption("zh-TW-HsiaoChenNeural", "HsiaoChen (曉臻) — Nữ Đài Loan dịu dàng", "female", "Đài Loan"),
        VoiceOption("zh-TW-YunJheNeural", "YunJhe (雲哲) — Nam Đài Loan ấm áp", "male", "Đài Loan"),
        VoiceOption("zh-CN-YunxiaNeural", "Yunxia (云夏) — Giọng thiếu niên / Trẻ em", "child", "Trung Quốc"),
    ],
    "th": [
        VoiceOption("th-TH-PremwadeeNeural", "Premwadee — Nữ Thái chuẩn, tự nhiên", "female", "Thái Lan"),
        VoiceOption("th-TH-NiwatNeural", "Niwat — Nam Thái chuẩn, rõ ràng", "male", "Thái Lan"),
    ],
    "id": [
        VoiceOption("id-ID-GadisNeural", "Gadis — Nữ Indonesia chuẩn", "female", "Indonesia"),
        VoiceOption("id-ID-ArdiNeural", "Ardi — Nam Indonesia chuẩn", "male", "Indonesia"),
    ],
    "fr": [
        VoiceOption("fr-FR-VivienneMultilingualNeural", "Vivienne HD — Nữ Pháp hiện đại, tự nhiên", "female", "Pháp"),
        VoiceOption("fr-FR-RemyMultilingualNeural", "Rémy HD — Nam Pháp ấm áp, lịch lãm", "male", "Pháp"),
        VoiceOption("fr-FR-DeniseNeural", "Denise — Nữ Pháp thanh lịch chuẩn", "female", "Pháp"),
        VoiceOption("fr-FR-HenriNeural", "Henri — Nam Pháp cổ điển", "male", "Pháp"),
        VoiceOption("fr-FR-EloiseNeural", "Eloise — Giọng thiếu nhi Pháp", "child", "Pháp"),
    ],
    "de": [
        VoiceOption("de-DE-TanjaNeural", "Tanja — Nữ Đức chuẩn, tự nhiên", "female", "Đức"),
        VoiceOption("de-DE-ChristophNeural", "Christoph — Nam Đức trầm ấm", "male", "Đức"),
    ],
    "es": [
        VoiceOption("es-ES-ElviraNeural", "Elvira — Nữ Tây Ban Nha chuẩn", "female", "Tây Ban Nha"),
        VoiceOption("es-ES-AlvaroNeural", "Alvaro — Nam Tây Ban Nha trầm ấm", "male", "Tây Ban Nha"),
        VoiceOption("es-ES-XimenaNeural", "Ximena — Nữ Tây Ban Nha trẻ trung", "female", "Tây Ban Nha"),
        VoiceOption("es-MX-DaliaNeural", "Dalia — Nữ Mexico / Mỹ Latin tự nhiên", "female", "Mexico"),
        VoiceOption("es-MX-JorgeNeural", "Jorge — Nam Mexico / Mỹ Latin", "male", "Mexico"),
    ],
    "pt": [
        VoiceOption("pt-BR-ThalitaMultilingualNeural", "Thalita HD — Nữ Brazil truyền cảm", "female", "Brazil"),
        VoiceOption("pt-BR-FranciscaNeural", "Francisca — Nữ Bồ Đào Nha (Brazil) tự nhiên", "female", "Brazil"),
        VoiceOption("pt-BR-AntonioNeural", "Antonio — Nam Bồ Đào Nha (Brazil) chuẩn", "male", "Brazil"),
    ],
    "it": [
        VoiceOption("it-IT-GiuseppeMultilingualNeural", "Giuseppe HD — Nam Ý tự nhiên, truyền cảm", "male", "Ý"),
        VoiceOption("it-IT-ElsaNeural", "Elsa — Nữ Ý trong sáng", "female", "Ý"),
        VoiceOption("it-IT-IsabellaNeural", "Isabella — Nữ Ý thanh lịch", "female", "Ý"),
        VoiceOption("it-IT-DiegoNeural", "Diego — Nam Ý trầm ấm", "male", "Ý"),
    ],
    "ru": [
        VoiceOption("ru-RU-SvetlanaNeural", "Svetlana — Nữ Nga chuẩn, trong sáng", "female", "Nga"),
        VoiceOption("ru-RU-DmitryNeural", "Dmitry — Nam Nga trầm hùng", "male", "Nga"),
    ],
    "hi": [
        VoiceOption("hi-IN-SwaraNeural", "Swara — Nữ Ấn Độ tự nhiên", "female", "Ấn Độ"),
        VoiceOption("hi-IN-MadhurNeural", "Madhur — Nam Ấn Độ chuẩn", "male", "Ấn Độ"),
    ],
}


SAMPLE_PREVIEW_TEXT: Dict[str, str] = {
    "vi": "Xin chào, đây là giọng đọc trí tuệ nhân tạo tự nhiên của ứng dụng Video Dubber.",
    "en": "Hello! Welcome to Video Dubber Studio. This is a natural AI voice preview for your video.",
    "en-GB": "Good day! Welcome to Video Dubber Studio. Here is a preview of the British neural voice.",
    "ko": "안녕하세요, 비디오 더버 스튜디오의 자연스러운 인공지능 음성 미리듣기입니다.",
    "ja": "こんにちは、ビデオダバースタジオの自然なAI音声のサンプルです。",
    "zh": "你好，欢迎使用智能视频配音软件，这是自然语音试听。",
    "th": "สวัสดีครับ นี่คือเสียงพากย์อัจฉริยะที่เป็นธรรมชาติสำหรับวิดีโอของคุณ",
    "id": "Halo, selamat datang di Video Dubber. Ini adalah contoh suara AI yang alami.",
    "fr": "Bonjour et bienvenue sur Video Dubber. Voici un aperçu de la voix naturelle.",
    "de": "Hallo und willkommen bei Video Dubber. Dies ist eine Vorschau der KI-Stimme.",
    "es": "Hola, bienvenido a Video Dubber. Esta es una muestra de voz artificial natural.",
    "pt": "Olá, bem-vindo ao Video Dubber. Esta é uma prévia da nossa voz neural.",
    "it": "Ciao e benvenuto in Video Dubber. Questa è un'anteprima della voce naturale.",
    "ru": "Здравствуйте! Это пример естественного нейронного голоса для озвучки видео.",
    "hi": "नमस्ते, वीडियो डबर में आपका स्वागत है। यह प्राकृतिक एआई आवाज़ का नमूना है।",
}


def get_voices_for_lang(lang_code: str) -> List[VoiceOption]:
    if lang_code in VOICE_CATALOG:
        return VOICE_CATALOG[lang_code]
    lang = LANGUAGES.get(lang_code, LANGUAGES["vi"])
    return [
        VoiceOption(lang.voice_female, f"Giọng Nữ ({lang.name})", "female", lang.name),
        VoiceOption(lang.voice_male, f"Giọng Nam ({lang.name})", "male", lang.name),
    ]


def encode_voice_spec(opt: VoiceOption) -> str:
    """Encode voice_id and optional pitch offset into a compact string 'voice_id|pitch'."""
    if opt.pitch and opt.pitch != "+0Hz":
        return f"{opt.voice_id}|{opt.pitch}"
    return opt.voice_id


def decode_voice_spec(spec: str) -> tuple[str, str]:
    """Return (voice_id, pitch) from encoded spec."""
    if "|" in (spec or ""):
        vid, pitch = spec.split("|", 1)
        return vid.strip(), (pitch.strip() or "+0Hz")
    return (spec or "").strip(), "+0Hz"


WHISPER_MODELS = ["small", "medium", "large-v3-turbo", "large-v3", "base", "tiny"]
NLLB_MODELS = {
    "NLLB 600M (Nhanh, nhẹ VRAM)": "facebook/nllb-200-distilled-600M",
    "NLLB 1.3B (Dịch mượt & chuẩn hơn)": "facebook/nllb-200-distilled-1.3B",
}


@dataclass
class Settings:
    process_mode: str = "dub_audio"      # "dub_audio" | "ocr_sub_only" | "ocr_sub_dub"
    source_lang: str = "auto"            # "auto" or key of LANGUAGES
    target_lang: str = "vi"
    whisper_model: str = "medium"
    translation_engine: str = "gemini"   # "gemini" (AI Context Script) | "nllb" (Offline MT)
    translation_style: str = "auto_natural"  # key of gemini_translate.TRANSLATION_STYLES
    translation_notes: str = ""          # custom pronoun/glossary instructions
    gemini_fallback_nllb: bool = True    # fallback to NLLB only if all 8 Gemini models exhaust quota
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    use_gpu: bool = True
    separate_vocals: bool = True         # Demucs: keep background music/effects
    original_vocals_volume: float = 0.0  # 0..1, keep a bit of original voice
    dub_volume: float = 1.05
    max_speedup: float = 1.50            # max TTS speed-up (lowered from 1.6 to avoid slurring)
    min_gap_ms: int = 150                # minimum breath pause (ms) between sentences
    tts_engine: str = "omnivoice"        # "omnivoice" (AI Voice Clone / Design) | "edge" (Microsoft Edge Neural)
    omnivoice_preset: str = "clone_auto" # key of omnivoice_tts.OMNIVOICE_PRESETS
    omnivoice_ref_audio: str = ""        # custom .wav/.mp3 path when omnivoice_preset == "clone_file"
    omnivoice_steps: int = 16            # OmniVoice diffusion steps: 16 (fast crisp) | 24 (high) | 32 (ultra)
    voice_mode: str = "auto_character"   # auto_character | auto | male | female | custom (Edge TTS)
    custom_voice: str = ""               # encoded voice spec when voice_mode == "custom" (Edge TTS)
    # subtitles
    blur_old_subs: bool = True
    burn_new_subs: bool = True
    sub_region_left: float = 0.02        # horizontal left edge of movable ROI (0..1)
    sub_region_right: float = 0.98       # horizontal right edge of movable ROI (0..1)
    sub_region_top: float = 0.72         # vertical top edge of movable ROI (0..1)
    sub_region_bottom: float = 0.98      # vertical bottom edge of movable ROI (0..1)
    ocr_fps: float = 5.0                 # frames per second sampled for OCR
    blur_style: str = "inpaint_blur"     # inpaint_blur | box | blur | pixelate
    sub_font_size: int = 0               # 0 = auto consistent size
    sub_place_on_old: bool = True        # align unified subtitle Y-band with old subs
    sub_bg_box: bool = False             # semi-transparent dark backdrop behind new subtitle text
    video_crf: int = 20

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
