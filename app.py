"""Desktop GUI application (PySide6) for AI Video Dubbing & Smart Subtitle Replacement.

Version 2.0 Highlights:
- Anti-slur TTS engine with natural breath pauses & 1:1 subtitle synchronization.
- Smart Telea stroke-inpainting + feathered cover for 100% clean old-subtitle removal & steady ASS placement.
- Rich multi-country Neural Voice catalog (VN, US, UK, Korea, Japan, China...) + live voice preview.
- Multi-Character Speaker Diarization (Male / Female / Deep / Child + Character 1, 2, 3).
- GitHub Auto-Updater + 1-Click Admin Release Publisher.
- Real-time Firebase Account Status & 1-Machine Lock enforcement (instant revocation on disable/logout).
- Modern, uncluttered Studio UI with monotonic weighted Step-by-Step Progress & ETA tracker.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Ensure GUI processes with no console (pythonw / PyInstaller) never crash on sys.stdout.write
class _SafeStream:
    def write(self, s): pass
    def flush(self): pass
    def isatty(self): return False

if sys.stdout is None:
    sys.stdout = _SafeStream()
if sys.stderr is None:
    sys.stderr = _SafeStream()

EMBEDDED_APP_VERSION = "2.5.1"


def _boot_hotpatch_if_newer():
    """If a verified newer HotPatch exists in %APPDATA%/VideoDubber/hotpatch, boot into it seamlessly."""
    if os.environ.get("VD_HOTPATCH_BOOTED") == "1":
        return
    try:
        import json as _json
        import re as _re
        import importlib.machinery as _mach

        hp_dir = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "VideoDubber" / "hotpatch"
        manifest_p = hp_dir / "patch_manifest.json"
        hp_app_p = hp_dir / "app.py"
        if not (manifest_p.is_file() and hp_app_p.is_file()):
            return
        mdata = _json.loads(manifest_p.read_text(encoding="utf-8"))
        v_hp = tuple(int(x) for x in _re.findall(r"\d+", str(mdata.get("version", "0")))[:4])
        v_em = tuple(int(x) for x in _re.findall(r"\d+", EMBEDDED_APP_VERSION)[:4])
        if v_hp <= v_em:
            return

        class _HotPatchFinder:
            def __init__(self, root_path: str):
                self.root_path = root_path

            def find_spec(self, fullname, path, target=None):
                if fullname == "dubber" or fullname.startswith("dubber."):
                    search_paths = [self.root_path] if path is None else path
                    return _mach.PathFinder.find_spec(fullname, search_paths, target)
                return None

        os.environ["VD_HOTPATCH_BOOTED"] = "1"
        sys.meta_path.insert(0, _HotPatchFinder(str(hp_dir)))
        sys.path.insert(0, str(hp_dir))
        import runpy
        runpy.run_path(str(hp_app_p), run_name="__main__")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        os.environ.pop("VD_HOTPATCH_BOOTED", None)


_boot_hotpatch_if_newer()

import dubber  # noqa: F401 - initializes Windows DLL directories first
from dubber import auth, gemini_translate, hw_profile, media, omnivoice_tts, pipeline, updater
from dubber.config import (
    LANGUAGES,
    NLLB_MODELS,
    SAMPLE_PREVIEW_TEXT,
    WHISPER_MODELS,
    Settings,
    encode_voice_spec,
    get_voices_for_lang,
)
from dubber.login_dialog import LoginDialog

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".ts"}
APP_VERSION = "2.5.1"


def _app_icon() -> QIcon:
    here = Path(__file__).parent
    for name in ("icon.ico", "icon.png"):
        p = here / name
        if p.exists():
            return QIcon(str(p))
    return QIcon()


def _fmt_mmss(sec: float) -> str:
    s = int(max(0, round(sec)))
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"


# =============================================================================
# BACKGROUND WORKERS
# =============================================================================
class SessionVerifyWorker(QThread):
    """Non-blocking live account status check against Firebase Auth & Firestore."""
    result_sig = Signal(bool, str)

    def run(self):
        try:
            ok, msg = auth.verify_session_online()
            self.result_sig.emit(ok, msg)
        except Exception as e:
            self.result_sig.emit(False, str(e))


class VoicePreviewWorker(QThread):
    """Synthesizes and plays a short voice sample via Windows native audio (OmniVoice or Edge TTS)."""
    done_sig = Signal(bool, str)

    def __init__(
        self,
        engine: str,
        voice_or_preset: str,
        text: str,
        target_lang: str = "vi",
        ref_audio_path: str = "",
        use_gpu: bool = True,
        omni_steps: int = 16,
    ):
        super().__init__()
        self.engine = engine
        self.voice_or_preset = voice_or_preset
        self.text = text
        self.target_lang = target_lang
        self.ref_audio_path = ref_audio_path
        self.use_gpu = use_gpu
        self.omni_steps = omni_steps

    def run(self):
        try:
            out_wav = os.path.join(tempfile.gettempdir(), "videodubber_voice_preview.wav")
            if self.engine == "omnivoice":
                omnivoice_tts.synthesize_preview_wav_omnivoice(
                    preset=self.voice_or_preset,
                    target_lang=self.target_lang,
                    text=self.text,
                    out_wav=out_wav,
                    ref_audio_path=self.ref_audio_path,
                    use_gpu=self.use_gpu,
                    num_step=self.omni_steps,
                )
            else:
                from dubber.tts import synthesize_preview_wav
                synthesize_preview_wav(self.voice_or_preset, self.text, out_wav)
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(out_wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.done_sig.emit(True, "")
        except Exception as e:
            self.done_sig.emit(False, str(e))


class _GeminiTestWorker(QThread):
    done_sig = Signal(bool, str, list)

    def __init__(self, raw_keys: str):
        super().__init__()
        self.raw_keys = raw_keys

    def run(self):
        ok, msg, models = gemini_translate.test_gemini_api_key(self.raw_keys)
        self.done_sig.emit(ok, msg, models)


class GeminiKeyDialog(QDialog):
    """Per-account Gemini API Key manager with High-to-Low 8-Model Cascade inspector."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._test_worker: _GeminiTestWorker | None = None
        self.setWindowTitle("Cấu hình khóa Gemini API (Lưu theo Tài khoản)")
        self.setFixedSize(560, 470)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(10)

        u = auth.get_current_user()
        email_str = u.get("email", "Chưa xác thực") if u else "Chưa xác thực"

        hdr = QLabel("🔑 Khóa Google Gemini API — Dịch Kịch Bản Chuẩn Văn Phong")
        hdr.setFont(QFont("Segoe UI", 13, QFont.Bold))
        hdr.setStyleSheet("color: #111827;")
        lay.addWidget(hdr)

        acc_lbl = QLabel(f"👤 Đang lưu riêng cho tài khoản: {email_str} (tự đồng bộ theo tài khoản của bạn)")
        acc_lbl.setStyleSheet("color: #1d4ed8; font-weight: 600; font-size: 12px;")
        lay.addWidget(acc_lbl)

        cascade_info = QLabel(
            "Thứ tự ưu tiên Model tự động từ Cao ➜ Thấp (tự nhảy sang model kế tiếp nếu hết lượt):\n"
            "1. Gemini 4 Hydrogen  ➔  2. Gemini 3.8 Flash  ➔  3. Gemini 3.6 Flash  ➔  4. Gemini 3 Flash\n"
            "➔  5. Gemini 3.5 Flash Lite  ➔  6. Gemini 3.1 Flash Lite  ➔  7. Gemini 2.5 Flash  ➔  8. Gemini 2.5 Flash Lite"
        )
        cascade_info.setWordWrap(True)
        cascade_info.setStyleSheet(
            "background: #f3f4f6; color: #374151; border: 1px solid #e5e7eb; "
            "border-radius: 6px; padding: 8px 10px; font-size: 11px;"
        )
        lay.addWidget(cascade_info)

        lbl_input = QLabel("Nhập khóa Gemini API (có thể nhập nhiều khóa, mỗi khóa 1 dòng để tăng hạn mức):")
        lbl_input.setStyleSheet("font-weight: bold; font-size: 12px;")
        lay.addWidget(lbl_input)

        self.txt_keys = QPlainTextEdit()
        self.txt_keys.setPlaceholderText("AIzaSy...\n(Mỗi dòng 1 khóa Gemini API lấy từ aistudio.google.com/apikey)")
        self.txt_keys.setPlainText(auth.get_account_gemini_key())
        self.txt_keys.setMaximumHeight(95)
        self.txt_keys.setFont(QFont("Consolas", 10))
        lay.addWidget(self.txt_keys)

        self.txt_status = QPlainTextEdit()
        self.txt_status.setReadOnly(True)
        self.txt_status.setPlaceholderText("Bấm '🧪 Kiểm tra kết nối' để xác nhận khóa API và danh sách Model khả dụng...")
        self.txt_status.setStyleSheet("background: #f9fafb; font-family: Consolas; font-size: 11px;")
        lay.addWidget(self.txt_status, 1)

        btn_row = QHBoxLayout()
        btn_get_key = QPushButton("🌐 Lấy khóa API miễn phí")
        btn_get_key.setCursor(Qt.PointingHandCursor)
        btn_get_key.clicked.connect(self._open_ai_studio)

        self.btn_test = QPushButton("🧪 Kiểm tra kết nối")
        self.btn_test.setCursor(Qt.PointingHandCursor)
        self.btn_test.clicked.connect(self._run_test)

        self.btn_save = QPushButton("💾 Lưu khóa theo tài khoản")
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: white; font-weight: bold; padding: 7px 16px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #15803d; }"
        )
        self.btn_save.clicked.connect(self._save_key)

        btn_close = QPushButton("Đóng")
        btn_close.clicked.connect(self.reject)

        btn_row.addWidget(btn_get_key)
        btn_row.addWidget(self.btn_test)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        btn_row.addWidget(self.btn_save)
        lay.addLayout(btn_row)

    @staticmethod
    def _open_ai_studio():
        import webbrowser
        webbrowser.open("https://aistudio.google.com/apikey")

    def _run_test(self):
        raw = self.txt_keys.toPlainText().strip()
        if not raw:
            QMessageBox.information(self, "Chưa nhập khóa", "Vui lòng dán khóa Gemini API vào ô trước khi kiểm tra.")
            return
        self.btn_test.setEnabled(False)
        self.btn_test.setText("⏳ Đang kiểm tra...")
        self.txt_status.setPlainText("Đang kết nối tới Google AI Studio và kiểm tra thứ tự Model...")

        self._test_worker = _GeminiTestWorker(raw)
        self._test_worker.done_sig.connect(self._on_test_done)
        self._test_worker.start()

    def _on_test_done(self, ok: bool, msg: str, models: list):
        self.btn_test.setEnabled(True)
        self.btn_test.setText("🧪 Kiểm tra kết nối")
        lines = [msg]
        if models:
            lines.append("\nDanh sách Model có lượt sử dụng đã sẵn sàng (Cao ➜ Thấp):")
            lines.extend(f"  {m}" for m in models)
        self.txt_status.setPlainText("\n".join(lines))

    def _save_key(self):
        raw = self.txt_keys.toPlainText().strip()
        if raw and not gemini_translate.parse_api_keys(raw):
            QMessageBox.warning(self, "Khóa không hợp lệ", "Định dạng khóa API quá ngắn hoặc không hợp lệ.")
            return
        self.btn_save.setEnabled(False)
        ok, msg = auth.save_account_gemini_key(raw)
        self.btn_save.setEnabled(True)
        if ok:
            QMessageBox.information(self, "Đã lưu khóa API", msg)
            self.accept()
        else:
            QMessageBox.warning(self, "Lỗi lưu khóa", msg)


class PipelineWorker(QThread):
    log_sig = Signal(str)
    progress_sig = Signal(str, float)
    analyzed_sig = Signal(list, str)
    finished_sig = Signal(str)
    error_sig = Signal(str)

    def __init__(self, mode: str, video: str, settings: Settings, segments=None, output: str = None):
        super().__init__()
        self.mode = mode  # "analyze" | "render" | "both"
        self.video = video
        self.settings = settings
        self.segments = segments or []
        self.output = output
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        # Live account check right before executing AI pipeline
        ok, auth_err = auth.verify_session_online()
        if not ok:
            self.error_sig.emit(f"AUTH_REVOKED::{auth_err}")
            return

        cb = pipeline.Callbacks(
            log=lambda msg: self.log_sig.emit(str(msg)),
            progress=lambda stage, frac: self.progress_sig.emit(stage, frac),
            cancelled=lambda: self._cancel,
        )
        try:
            segs = self.segments
            if self.mode in ("analyze", "both"):
                segs, src_lang = pipeline.analyze(self.video, self.settings, cb)
                self.analyzed_sig.emit(segs, src_lang)
                if self.mode == "analyze":
                    self.finished_sig.emit("")
                    return
            if self.mode in ("render", "both"):
                out = pipeline.render(self.video, self.settings, segs, cb, output=self.output)
                self.finished_sig.emit(out)
        except InterruptedError:
            self.log_sig.emit("⏹ Đã hủy tác vụ theo yêu cầu.")
            self.finished_sig.emit("")
        except Exception as e:
            tb = traceback.format_exc()
            self.log_sig.emit(tb)
            self.error_sig.emit(str(e))


class BatchWorker(QThread):
    item_started = Signal(int, str)
    item_progress = Signal(int, str, float)
    item_finished = Signal(int, str)
    item_error = Signal(int, str)
    batch_finished = Signal(int, int)
    auth_revoked = Signal(str)
    log_sig = Signal(str)

    # Stage weights for accurate per-video 0..100% calculation in batch mode
    _WEIGHTS_AUDIO = {
        pipeline.STAGE_SEPARATE: (0.00, 0.18),
        pipeline.STAGE_ASR: (0.18, 0.18),
        pipeline.STAGE_DIARIZE: (0.36, 0.04),
        pipeline.STAGE_TRANSLATE: (0.40, 0.12),
        pipeline.STAGE_TTS: (0.52, 0.16),
        pipeline.STAGE_MIX: (0.68, 0.02),
        pipeline.STAGE_OCR: (0.70, 0.12),
        pipeline.STAGE_RENDER: (0.82, 0.18),
    }
    _WEIGHTS_OCR_ONLY = {
        pipeline.STAGE_OCR: (0.00, 0.42),
        pipeline.STAGE_TRANSLATE: (0.42, 0.18),
        pipeline.STAGE_RENDER: (0.60, 0.40),
    }
    _WEIGHTS_OCR_DUB = {
        pipeline.STAGE_OCR: (0.00, 0.30),
        pipeline.STAGE_TRANSLATE: (0.30, 0.14),
        pipeline.STAGE_TTS: (0.44, 0.24),
        pipeline.STAGE_MIX: (0.68, 0.03),
        pipeline.STAGE_RENDER: (0.71, 0.29),
    }

    def __init__(self, items: list[dict], settings: Settings, out_dir: str = ""):
        super().__init__()
        self.items = items
        self.settings = settings
        self.out_dir = out_dir
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _map_frac(self, stage: str, frac: float) -> float:
        pmode = getattr(self.settings, "process_mode", "dub_audio")
        wmap = (
            self._WEIGHTS_OCR_ONLY if pmode == "ocr_sub_only"
            else (self._WEIGHTS_OCR_DUB if pmode == "ocr_sub_dub" else self._WEIGHTS_AUDIO)
        )
        base, span = wmap.get(stage, (0.0, 1.0))
        return min(0.99, base + span * float(np.clip(frac, 0.0, 1.0)))

    def run(self):
        success_count = 0
        fail_count = 0
        total = len(self.items)
        self.log_sig.emit(f"=== BẮT ĐẦU CHẠY DANH SÁCH: {total} VIDEO ===")

        for idx, item in enumerate(self.items):
            if self._cancel:
                self.log_sig.emit("⏹ Đã dừng xử lý danh sách.")
                break

            # Re-verify Firebase account status before each video in batch
            ok, auth_err = auth.verify_session_online()
            if not ok:
                self.auth_revoked.emit(auth_err)
                return

            video_path = item["path"]
            fname = Path(video_path).name
            self.item_started.emit(idx, fname)
            self.log_sig.emit(f"\n[{idx + 1}/{total}] Đang xử lý: {fname}")

            suffix = "sub" if getattr(self.settings, "process_mode", "dub_audio") == "ocr_sub_only" else "dub"
            if self.out_dir and os.path.isdir(self.out_dir):
                stem = Path(video_path).stem
                out_path = str(Path(self.out_dir) / f"{stem}_{self.settings.target_lang}_{suffix}.mp4")
            else:
                vp = Path(video_path)
                out_path = str(vp.with_name(f"{vp.stem}_{self.settings.target_lang}_{suffix}.mp4"))

            cb = pipeline.Callbacks(
                log=lambda msg: self.log_sig.emit(f"  {msg}"),
                progress=lambda stg, fr, i=idx: self.item_progress.emit(i, stg, self._map_frac(stg, fr)),
                cancelled=lambda: self._cancel,
            )

            try:
                segs, _src = pipeline.analyze(video_path, self.settings, cb)
                if self._cancel:
                    break
                final_out = pipeline.render(video_path, self.settings, segs, cb, output=out_path)
                self.item_finished.emit(idx, final_out)
                self.log_sig.emit(f"✅ [{idx + 1}/{total}] Hoàn tất: {final_out}")
                success_count += 1
            except InterruptedError:
                self.log_sig.emit(f"⏹ Đã hủy khi đang xử lý {fname}")
                break
            except Exception as e:
                err_str = str(e)
                self.log_sig.emit(f"❌ [{idx + 1}/{total}] Lỗi {fname}: {err_str}")
                self.item_error.emit(idx, err_str)
                fail_count += 1

        self.batch_finished.emit(success_count, fail_count)


# =============================================================================
# STEP-BY-STEP VISUAL PIPELINE TRACKER (ACCURATE MONOTONIC PROGRESS + ETA)
# =============================================================================
class PipelineProgressCard(QFrame):
    """Visual multi-step progress tracker with stage pills, monotonic overall %, and live ETA."""

    MODE_STEPS = {
        "analyze": [
            (pipeline.STAGE_SEPARATE, "1. Tách nhạc", 0.34),
            (pipeline.STAGE_ASR, "2. Nghe thoại", 0.34),
            (pipeline.STAGE_DIARIZE, "3. Phân vai", 0.08),
            (pipeline.STAGE_TRANSLATE, "4. Dịch kịch bản", 0.24),
        ],
        "analyze_ocr": [
            (pipeline.STAGE_OCR, "1. Quét chữ phụ đề (OCR)", 0.65),
            (pipeline.STAGE_TRANSLATE, "2. Dịch phụ đề AI", 0.35),
        ],
        "render": [
            (pipeline.STAGE_TTS, "1. Lồng tiếng", 0.32),
            (pipeline.STAGE_MIX, "2. Trộn nhạc", 0.04),
            (pipeline.STAGE_OCR, "3. Quét phụ đề", 0.26),
            (pipeline.STAGE_RENDER, "4. Xuất Video", 0.38),
        ],
        "render_ocr": [
            (pipeline.STAGE_OCR, "1. Vùng phụ đề cũ", 0.25),
            (pipeline.STAGE_RENDER, "2. Xóa sub cũ & Ghép sub mới", 0.75),
        ],
        "both": [
            (pipeline.STAGE_SEPARATE, "1. Tách nhạc", 0.18),
            (pipeline.STAGE_ASR, "2. Nghe thoại", 0.18),
            (pipeline.STAGE_DIARIZE, "3. Phân vai", 0.04),
            (pipeline.STAGE_TRANSLATE, "4. Dịch kịch bản", 0.12),
            (pipeline.STAGE_TTS, "5. Lồng tiếng", 0.16),
            (pipeline.STAGE_OCR, "6. Phụ đề OCR", 0.12),
            (pipeline.STAGE_RENDER, "7. Xuất Video", 0.20),
        ],
        "both_ocr": [
            (pipeline.STAGE_OCR, "1. Quét chữ sub (OCR)", 0.42),
            (pipeline.STAGE_TRANSLATE, "2. Dịch phụ đề", 0.20),
            (pipeline.STAGE_RENDER, "3. Xóa sub cũ & Ghép sub mới", 0.38),
        ],
        "both_ocr_dub": [
            (pipeline.STAGE_OCR, "1. Quét chữ sub (OCR)", 0.30),
            (pipeline.STAGE_TRANSLATE, "2. Dịch phụ đề", 0.14),
            (pipeline.STAGE_TTS, "3. Lồng tiếng AI", 0.26),
            (pipeline.STAGE_RENDER, "4. Xóa sub cũ & Xuất Video", 0.30),
        ],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ProgressCard")
        self._mode = "both"
        self._steps = self.MODE_STEPS["both"]
        self._pill_labels: list[QLabel] = []
        self._start_ts: float = 0.0
        self._overall_pct: float = 0.0
        self._active: bool = False
        self._cur_stage_name: str = "Sẵn sàng"
        self._cur_stage_pct: int = 0

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick_clock)

        self._build_ui()
        self._rebuild_pills("both")

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)

        # Top row: Stage title + Overall % + Elapsed/ETA
        top = QHBoxLayout()
        self.lbl_title = QLabel("🟢 Sẵn sàng xử lý")
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #111827;")

        self.lbl_timer = QLabel("")
        self.lbl_timer.setStyleSheet("color: #4b5563; font-size: 12px; font-weight: 500;")

        self.lbl_pct = QLabel("0%")
        self.lbl_pct.setStyleSheet("font-weight: bold; font-size: 15px; color: #2563eb;")

        top.addWidget(self.lbl_title)
        top.addStretch(1)
        top.addWidget(self.lbl_timer)
        top.addSpacing(14)
        top.addWidget(self.lbl_pct)
        lay.addLayout(top)

        # Main progress bar (0..1000 for smooth 0.1% increments)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        self.bar.setStyleSheet(
            "QProgressBar { background-color: #e5e7eb; border: none; border-radius: 5px; }"
            "QProgressBar::chunk { background-color: #2563eb; border-radius: 5px; }"
        )
        lay.addWidget(self.bar)

        # Step pills row
        self.pills_layout = QHBoxLayout()
        self.pills_layout.setSpacing(6)
        lay.addLayout(self.pills_layout)

    def _rebuild_pills(self, mode: str):
        self._mode = mode if mode in self.MODE_STEPS else "both"
        self._steps = self.MODE_STEPS[self._mode]
        while self.pills_layout.count():
            item = self.pills_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._pill_labels.clear()

        for _, short_label, _ in self._steps:
            lbl = QLabel(short_label)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(self._pill_style("pending"))
            self.pills_layout.addWidget(lbl, 1)
            self._pill_labels.append(lbl)

    @staticmethod
    def _pill_style(state: str) -> str:
        if state == "done":
            return (
                "background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; "
                "border-radius: 4px; padding: 3px 6px; font-size: 11px; font-weight: bold;"
            )
        if state == "active":
            return (
                "background-color: #dbeafe; color: #1d4ed8; border: 1px solid #60a5fa; "
                "border-radius: 4px; padding: 3px 6px; font-size: 11px; font-weight: bold;"
            )
        return (
            "background-color: #f3f4f6; color: #9ca3af; border: 1px solid #e5e7eb; "
            "border-radius: 4px; padding: 3px 6px; font-size: 11px;"
        )

    def start(self, mode: str):
        self._rebuild_pills(mode)
        self._start_ts = time.time()
        self._overall_pct = 0.0
        self._active = True
        self._cur_stage_name = "Đang khởi tạo mô hình..."
        self._cur_stage_pct = 0
        self.bar.setValue(8)
        self.lbl_pct.setText("1%")
        self.lbl_title.setText("🔵 Đang khởi tạo tiến trình...")
        self.lbl_timer.setText("⏱ Đã chạy: 00:00")
        self._timer.start()

    def update_stage(self, stage: str, frac: float):
        if not self._active:
            return
        frac = float(np.clip(frac, 0.0, 1.0))
        self._cur_stage_name = stage
        self._cur_stage_pct = int(round(frac * 100))

        # Find matching step index (handle STAGE_MIX smoothly inside TTS/OCR transition)
        active_idx = -1
        cum_before = 0.0
        active_weight = 0.0
        for idx, (s_key, _, weight) in enumerate(self._steps):
            if s_key == stage:
                active_idx = idx
                active_weight = weight
                break
            cum_before += weight

        if stage == pipeline.STAGE_MIX and active_idx == -1:
            # Map quick mix step right after TTS
            for idx, (s_key, _, weight) in enumerate(self._steps):
                cum_before += weight
                if s_key == pipeline.STAGE_TTS:
                    active_idx = idx
                    active_weight = 0.0
                    break

        if active_idx >= 0:
            computed = (cum_before + active_weight * frac) * 100.0
            # Strictly monotonic so bar NEVER jumps backward!
            self._overall_pct = max(self._overall_pct, min(99.4, computed))
            for idx, lbl in enumerate(self._pill_labels):
                base_txt = self._steps[idx][1]
                if idx < active_idx:
                    lbl.setText(f"✓ {base_txt.split('. ', 1)[-1]}")
                    lbl.setStyleSheet(self._pill_style("done"))
                elif idx == active_idx:
                    lbl.setText(f"● {base_txt} ({self._cur_stage_pct}%)")
                    lbl.setStyleSheet(self._pill_style("active"))
                else:
                    lbl.setText(base_txt)
                    lbl.setStyleSheet(self._pill_style("pending"))

        self.bar.setValue(int(round(self._overall_pct * 10)))
        self.lbl_pct.setText(f"{int(round(self._overall_pct))}%")
        self.lbl_title.setText(f"🔵 Đang thực hiện: {stage} ({self._cur_stage_pct}%)")
        self._tick_clock()

    def _tick_clock(self):
        if not self._active:
            return
        elapsed = max(0.0, time.time() - self._start_ts)
        if self._overall_pct >= 6.0 and elapsed >= 3.0:
            rem = elapsed * (100.0 - self._overall_pct) / max(1.0, self._overall_pct)
            self.lbl_timer.setText(f"⏱ Đã chạy: {_fmt_mmss(elapsed)}  ·  Ước tính còn: ~{_fmt_mmss(rem)}")
        else:
            self.lbl_timer.setText(f"⏱ Đã chạy: {_fmt_mmss(elapsed)}  ·  Đang tính thời gian...")

    def finish(self, cancelled: bool = False):
        self._active = False
        self._timer.stop()
        elapsed = max(0.0, time.time() - self._start_ts) if self._start_ts else 0.0
        if cancelled:
            self.lbl_title.setText("⏹ Đã dừng tác vụ")
            self.lbl_timer.setText(f"⏱ Thời gian: {_fmt_mmss(elapsed)}")
            return
        self._overall_pct = 100.0
        self.bar.setValue(1000)
        self.lbl_pct.setText("100%")
        self.lbl_title.setText("✅ Hoàn tất xử lý thành công!")
        self.lbl_timer.setText(f"⏱ Tổng thời gian: {_fmt_mmss(elapsed)}")
        for idx, lbl in enumerate(self._pill_labels):
            base_txt = self._steps[idx][1]
            lbl.setText(f"✓ {base_txt.split('. ', 1)[-1]}")
            lbl.setStyleSheet(self._pill_style("done"))

    def fail(self):
        self._active = False
        self._timer.stop()
        self.lbl_title.setText("❌ Gặp lỗi trong quá trình xử lý (Xem nhật ký bên dưới)")


# =============================================================================
# PREVIEW & BATCH TABLE WIDGETS
# =============================================================================
class PreviewLabel(QLabel):
    """Interactive video frame preview with a draggable & 8-handle resizable Subtitle Detection ROI box."""
    file_dropped = Signal(str)
    roi_changed = Signal(float, float, float, float)  # (left, top, right, bottom) in [0..1]

    _MIN_W = 0.08
    _MIN_H = 0.04
    _HANDLE_HIT = 11

    def __init__(self):
        super().__init__("🎬 Kéo thả video vào đây\nhoặc bấm '📂 Chọn Video' ở trên")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(440, 250)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setStyleSheet(
            "QLabel { background-color: #111827; color: #9ca3af; font-size: 13px; "
            "border: 1px solid #374151; border-radius: 8px; }"
        )
        self._pixmap: QPixmap | None = None
        self._scaled_pixmap: QPixmap | None = None
        self._roi: list[float] = [0.02, 0.72, 0.98, 0.98]  # [left, top, right, bottom]
        self._show_roi: bool = True
        self._drag_mode: str | None = None  # None | 'move' | 'draw' | 'tl'|'tr'|'bl'|'br'|'t'|'b'|'l'|'r'
        self._drag_start_norm: tuple[float, float] = (0.0, 0.0)
        self._drag_orig_roi: list[float] = list(self._roi)

    def set_frame(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._rescale()
        self.update()

    def set_roi(self, left: float, top: float, right: float, bottom: float, emit: bool = False):
        l = float(np.clip(left, 0.0, 1.0 - self._MIN_W))
        r = float(np.clip(right, l + self._MIN_W, 1.0))
        t = float(np.clip(top, 0.0, 1.0 - self._MIN_H))
        b = float(np.clip(bottom, t + self._MIN_H, 1.0))
        self._roi = [l, t, r, b]
        self.update()
        if emit:
            self.roi_changed.emit(l, t, r, b)

    def get_roi(self) -> tuple[float, float, float, float]:
        return tuple(self._roi)  # type: ignore[return-value]

    def set_show_roi(self, visible: bool):
        self._show_roi = bool(visible)
        self.update()

    def _rescale(self):
        if self._pixmap is not None:
            self._scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            self._scaled_pixmap = None

    def _image_rect(self) -> QRect:
        if self._scaled_pixmap is None:
            return QRect()
        pw = self._scaled_pixmap.width()
        ph = self._scaled_pixmap.height()
        ox = (self.width() - pw) // 2
        oy = (self.height() - ph) // 2
        return QRect(ox, oy, pw, ph)

    def _roi_widget_rect(self) -> QRect:
        ir = self._image_rect()
        if ir.isNull() or ir.width() < 2 or ir.height() < 2:
            return QRect()
        l, t, r, b = self._roi
        x1 = int(round(ir.left() + l * ir.width()))
        y1 = int(round(ir.top() + t * ir.height()))
        x2 = int(round(ir.left() + r * ir.width()))
        y2 = int(round(ir.top() + b * ir.height()))
        return QRect(x1, y1, max(4, x2 - x1), max(4, y2 - y1))

    def _pos_to_norm(self, px: float, py: float) -> tuple[float, float]:
        ir = self._image_rect()
        if ir.isNull() or ir.width() <= 0 or ir.height() <= 0:
            return 0.0, 0.0
        nx = float(np.clip((px - ir.left()) / ir.width(), 0.0, 1.0))
        ny = float(np.clip((py - ir.top()) / ir.height(), 0.0, 1.0))
        return nx, ny

    def _hit_test(self, px: float, py: float) -> str | None:
        if self._scaled_pixmap is None or not self._show_roi:
            return None
        ir = self._image_rect()
        if not ir.adjusted(-12, -12, 12, 12).contains(int(px), int(py)):
            return None
        rr = self._roi_widget_rect()
        if rr.isNull():
            return None

        ht = self._HANDLE_HIT
        cx = rr.center().x()
        cy = rr.center().y()
        handles = [
            ("tl", rr.left(), rr.top()),
            ("tr", rr.right(), rr.top()),
            ("bl", rr.left(), rr.bottom()),
            ("br", rr.right(), rr.bottom()),
            ("t", cx, rr.top()),
            ("b", cx, rr.bottom()),
            ("l", rr.left(), cy),
            ("r", rr.right(), cy),
        ]
        for name, hx, hy in handles:
            if abs(px - hx) <= ht and abs(py - hy) <= ht:
                return name
        # Also allow grabbing anywhere along the 4 edges cleanly
        if rr.left() <= px <= rr.right():
            if abs(py - rr.top()) <= 7:
                return "t"
            if abs(py - rr.bottom()) <= 7:
                return "b"
        if rr.top() <= py <= rr.bottom():
            if abs(px - rr.left()) <= 7:
                return "l"
            if abs(px - rr.right()) <= 7:
                return "r"
        if rr.contains(int(px), int(py)):
            return "move"
        return "draw"

    def _cursor_for_hit(self, hit: str | None):
        if hit in ("tl", "br"):
            return Qt.SizeFDiagCursor
        if hit in ("tr", "bl"):
            return Qt.SizeBDiagCursor
        if hit in ("t", "b"):
            return Qt.SizeVerCursor
        if hit in ("l", "r"):
            return Qt.SizeHorCursor
        if hit == "move":
            return Qt.OpenHandCursor
        if hit == "draw":
            return Qt.CrossCursor
        return Qt.ArrowCursor

    def paintEvent(self, ev):
        if self._scaled_pixmap is None:
            super().paintEvent(ev)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#111827"))

        ir = self._image_rect()
        painter.drawPixmap(ir.topLeft(), self._scaled_pixmap)

        if self._show_roi and not ir.isNull():
            rr = self._roi_widget_rect()
            # Translucent green highlight inside movable subtitle ROI
            painter.fillRect(rr, QColor(34, 197, 94, 34))

            # Crisp bright green border
            pen = QPen(QColor("#22c55e"), 2)
            painter.setPen(pen)
            painter.drawRect(rr)

            # 8 interactive white-filled resize handles (corners + edge midpoints)
            hs = 4
            cx = rr.center().x()
            cy = rr.center().y()
            pts = [
                (rr.left(), rr.top()), (rr.right(), rr.top()),
                (rr.left(), rr.bottom()), (rr.right(), rr.bottom()),
                (cx, rr.top()), (cx, rr.bottom()),
                (rr.left(), cy), (rr.right(), cy),
            ]
            painter.setBrush(QColor("#ffffff"))
            painter.setPen(QPen(QColor("#15803d"), 1.5))
            for hx, hy in pts:
                painter.drawRect(int(hx - hs), int(hy - hs), hs * 2, hs * 2)

            # Floating hint badge above (or inside top of) the ROI box
            badge_txt = "🎯 Khung quét phụ đề (Kéo di chuyển · Kéo góc co giãn)"
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(badge_txt) + 12
            th = fm.height() + 4
            bx = max(ir.left() + 4, min(rr.left(), ir.right() - tw - 4))
            by = rr.top() - th - 3 if (rr.top() - th - 3 >= ir.top() + 2) else (rr.top() + 4)
            badge_rect = QRect(bx, by, tw, th)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(17, 24, 39, 200))
            painter.drawRoundedRect(badge_rect, 4, 4)
            painter.setPen(QColor("#86efac"))
            painter.drawText(badge_rect, Qt.AlignCenter, badge_txt)

        painter.end()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._scaled_pixmap is not None and self._show_roi:
            pos = ev.position()
            hit = self._hit_test(pos.x(), pos.y())
            if hit is not None:
                self._drag_mode = hit
                self._drag_start_norm = self._pos_to_norm(pos.x(), pos.y())
                self._drag_orig_roi = list(self._roi)
                if hit == "move":
                    self.setCursor(Qt.ClosedHandCursor)
                ev.accept()
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._drag_mode is None:
            hit = self._hit_test(pos.x(), pos.y())
            self.setCursor(self._cursor_for_hit(hit))
            super().mouseMoveEvent(ev)
            return

        nx, ny = self._pos_to_norm(pos.x(), pos.y())
        sx, sy = self._drag_start_norm
        ol, ot, or_, ob = self._drag_orig_roi
        l, t, r, b = ol, ot, or_, ob
        mode = self._drag_mode

        if mode == "move":
            dx = nx - sx
            dy = ny - sy
            bw = or_ - ol
            bh = ob - ot
            l = float(np.clip(ol + dx, 0.0, 1.0 - bw))
            t = float(np.clip(ot + dy, 0.0, 1.0 - bh))
            r = l + bw
            b = t + bh
        elif mode == "draw":
            l = min(sx, nx)
            r = max(sx, nx)
            t = min(sy, ny)
            b = max(sy, ny)
            if r - l < self._MIN_W:
                r = min(1.0, l + self._MIN_W)
                l = max(0.0, r - self._MIN_W)
            if b - t < self._MIN_H:
                b = min(1.0, t + self._MIN_H)
                t = max(0.0, b - self._MIN_H)
        else:
            if "l" in mode:
                l = float(np.clip(nx, 0.0, r - self._MIN_W))
            if "r" in mode:
                r = float(np.clip(nx, l + self._MIN_W, 1.0))
            if "t" in mode:
                t = float(np.clip(ny, 0.0, b - self._MIN_H))
            if "b" in mode:
                b = float(np.clip(ny, t + self._MIN_H, 1.0))

        self._roi = [l, t, r, b]
        self.update()
        self.roi_changed.emit(l, t, r, b)
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if self._drag_mode is not None:
            self._drag_mode = None
            pos = ev.position()
            self.setCursor(self._cursor_for_hit(self._hit_test(pos.x(), pos.y())))
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        if self._scaled_pixmap is not None and self._show_roi:
            # Double-click resets ROI box smoothly to standard bottom subtitle zone
            self.set_roi(0.02, 0.72, 0.98, 0.98, emit=True)
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale()
        self.update()

    def dragEnterEvent(self, ev: QDragEnterEvent):
        if ev.mimeData().hasUrls():
            for u in ev.mimeData().urls():
                if Path(u.toLocalFile()).suffix.lower() in VIDEO_EXTS:
                    ev.acceptProposedAction()
                    return

    def dropEvent(self, ev: QDropEvent):
        for u in ev.mimeData().urls():
            p = u.toLocalFile()
            if Path(p).suffix.lower() in VIDEO_EXTS:
                self.file_dropped.emit(p)
                return


class BatchTableWidget(QTableWidget):
    files_dropped = Signal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, ev: QDragEnterEvent):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev: QDropEvent):
        paths = []
        for u in ev.mimeData().urls():
            p = u.toLocalFile()
            if Path(p).suffix.lower() in VIDEO_EXTS:
                paths.append(p)
            elif os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for f in files:
                        if Path(f).suffix.lower() in VIDEO_EXTS:
                            paths.append(str(Path(root) / f))
        if paths:
            self.files_dropped.emit(paths)


# =============================================================================
# MAIN STUDIO WINDOW
# =============================================================================
class MainWindow(QMainWindow):
    logout_requested = Signal(str)  # emits optional reason message for LoginDialog

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Video Dubber Studio v{APP_VERSION} — Dịch & Lồng tiếng Video AI")
        self.setWindowIcon(_app_icon())
        self.resize(1420, 900)
        self.setMinimumSize(1060, 720)

        self.video_path: str = ""
        self.video_info: dict = {}
        self.last_output: str = ""
        self.worker: PipelineWorker | None = None
        self.batch_worker: BatchWorker | None = None
        self._preview_worker: VoicePreviewWorker | None = None
        self._upd_worker: updater.UpdateCheckWorker | None = None
        self._verify_worker: SessionVerifyWorker | None = None
        self._pending_update_info: dict | None = None
        self._ocr_detector = None
        self._cap = None
        self._tray = None
        self._signing_out = False

        self._build_ui()
        self._apply_theme()
        self._setup_tray()
        self._refresh_voice_combo()
        self._sync_tts_engine_ui()
        self._sync_process_mode_ui()
        self._refresh_gemini_badge()
        prof_init = hw_profile.get_hw_profile()
        self._log(f"🖥 Cấu hình tăng tốc: {prof_init.badge_text}")
        if prof_init.driver_outdated and prof_init.driver_warning:
            self._log(prof_init.driver_warning)

        # Live Firebase account heartbeat (every 2.5 minutes) — kicks immediately if disabled on Firebase
        self._auth_timer = QTimer(self)
        self._auth_timer.setInterval(150_000)
        self._auth_timer.timeout.connect(self._run_heartbeat_check)
        self._auth_timer.start()

        # Silent startup update check after 2 seconds
        QTimer.singleShot(2000, lambda: self._check_updates(silent=True))

    # ------------------------------------------------------------------ UI Construction
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(12, 8, 12, 8)
        main_layout.setSpacing(8)

        # ── 1. Clean Header Bar ─────────────────────────────────────────
        header = QFrame()
        header.setObjectName("HeaderBar")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(14, 8, 14, 8)
        h_lay.setSpacing(10)

        lbl_brand = QLabel(f"🎬 Video Dubber Studio  v{APP_VERSION}")
        lbl_brand.setFont(QFont("Segoe UI", 13, QFont.Bold))
        lbl_brand.setStyleSheet("color: #111827;")
        h_lay.addWidget(lbl_brand)

        lbl_tag = QLabel("Lồng tiếng đa nhân vật · Xóa phụ đề thông minh")
        lbl_tag.setStyleSheet("color: #6b7280; font-size: 12px;")
        h_lay.addWidget(lbl_tag)
        h_lay.addStretch(1)

        self.btn_gemini_key = QPushButton("🔑 Gemini API")
        self.btn_gemini_key.setCursor(Qt.PointingHandCursor)
        self.btn_gemini_key.setToolTip("Nhập hoặc quản lý khóa Gemini API lưu riêng theo tài khoản")
        self.btn_gemini_key.clicked.connect(self._open_gemini_key_dialog)
        h_lay.addWidget(self.btn_gemini_key)

        self.btn_check_upd = QPushButton("🔄 Kiểm tra cập nhật")
        self.btn_check_upd.setCursor(Qt.PointingHandCursor)
        self.btn_check_upd.setStyleSheet(
            "QPushButton { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; padding: 5px 12px; border-radius: 6px; font-weight: 600; font-size: 12px; }"
            "QPushButton:hover { background: #dbeafe; }"
        )
        self.btn_check_upd.clicked.connect(lambda: self._check_updates(silent=False))
        h_lay.addWidget(self.btn_check_upd)

        u = auth.get_current_user()
        email_txt = u.get("email", "Chưa xác thực") if u else "Chưa xác thực"
        self.lbl_user_badge = QLabel(f"👤 {email_txt}")
        self.lbl_user_badge.setStyleSheet(
            "background: #f3f4f6; color: #1f2937; border: 1px solid #e5e7eb; "
            "border-radius: 6px; padding: 5px 10px; font-size: 12px; font-weight: 600;"
        )
        h_lay.addWidget(self.lbl_user_badge)

        btn_logout = QPushButton("🚪 Đăng xuất")
        btn_logout.setCursor(Qt.PointingHandCursor)
        btn_logout.setStyleSheet(
            "QPushButton { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; padding: 5px 12px; border-radius: 6px; font-weight: 600; font-size: 12px; }"
            "QPushButton:hover { background: #fee2e2; }"
        )
        btn_logout.clicked.connect(self._logout_clicked)
        h_lay.addWidget(btn_logout)

        main_layout.addWidget(header)

        # ── 1B. Proactive NVIDIA Driver Outdated Alert Banner (if CUDA failed on an NVIDIA PC) ──
        prof = hw_profile.get_hw_profile()
        if prof.driver_outdated and prof.driver_warning:
            drv_bar = QFrame()
            drv_bar.setStyleSheet(
                "QFrame { background-color: #fff7ed; border: 1.5px solid #fb923c; border-radius: 7px; }"
            )
            d_lay = QHBoxLayout(drv_bar)
            d_lay.setContentsMargins(12, 6, 12, 6)
            d_lay.setSpacing(10)
            lbl_drv = QLabel(prof.driver_warning)
            lbl_drv.setWordWrap(True)
            lbl_drv.setStyleSheet("color: #9a3412; font-weight: bold; font-size: 12px; border: none;")
            btn_dl_drv = QPushButton("🌐 Tải NVIDIA Driver mới nhất")
            btn_dl_drv.setCursor(Qt.PointingHandCursor)
            btn_dl_drv.setStyleSheet(
                "QPushButton { background: #ea580c; color: white; font-weight: bold; padding: 6px 12px; border-radius: 5px; border: none; }"
                "QPushButton:hover { background: #c2410c; }"
            )
            btn_dl_drv.clicked.connect(self._open_nvidia_driver_page)
            d_lay.addWidget(lbl_drv, 1)
            d_lay.addWidget(btn_dl_drv)
            main_layout.addWidget(drv_bar)

        # ── Status Bar & Menu Bar ───────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage(
            f"{prof.badge_text}  ·  Thiết bị: {auth.get_short_machine_id()}"
        )
        self._build_menu()

        # ── 2. Workspace Mode Tabs (Single vs Batch) ────────────────────
        self.tabs = QTabWidget()
        self.tab_single = QWidget()
        self.tab_batch = QWidget()

        self._build_single_tab(self.tab_single)
        self._build_batch_tab(self.tab_batch)

        self.tabs.addTab(self.tab_single, "🎬 Xử lý 1 Video (Xem trước & Chỉnh sửa chi tiết)")
        self.tabs.addTab(self.tab_batch, "📋 Chạy theo danh sách hàng loạt (Batch Queue)")
        main_layout.addWidget(self.tabs, 1)

    def _build_menu(self):
        mb = self.menuBar()

        m_file = mb.addMenu("&Tệp")
        a_open = m_file.addAction("📂 Mở video...")
        a_open.setShortcut("Ctrl+O")
        a_open.triggered.connect(self._browse_video)

        a_batch = m_file.addAction("➕ Thêm nhiều video vào danh sách Batch...")
        a_batch.triggered.connect(self._batch_add_files)

        a_save = m_file.addAction("💾 Lưu bản dịch hiện tại")
        a_save.setShortcut("Ctrl+S")
        a_save.triggered.connect(self._save_edits)

        m_file.addSeparator()
        a_quit = m_file.addAction("Thoát ứng dụng")
        a_quit.setShortcut("Ctrl+Q")
        a_quit.triggered.connect(self.close)

        m_acc = mb.addMenu("&Tài khoản & Cập nhật")
        a_gem = m_acc.addAction("🔑 Cấu hình khóa Gemini API (Lưu theo tài khoản)...")
        a_gem.triggered.connect(self._open_gemini_key_dialog)

        a_mid = m_acc.addAction(f"📋 Sao chép Mã máy ({auth.get_short_machine_id()})")
        a_mid.triggered.connect(self._copy_machine_id)

        a_upd = m_acc.addAction("🔄 Kiểm tra bản cập nhật mới...")
        a_upd.triggered.connect(lambda: self._check_updates(silent=False))

        m_acc.addSeparator()
        a_out = m_acc.addAction("🚪 Đăng xuất tài khoản")
        a_out.triggered.connect(self._logout_clicked)

        m_help = mb.addMenu("&Trợ giúp")
        a_about = m_help.addAction(f"ℹ️ Giới thiệu Video Dubber Studio v{APP_VERSION}")
        a_about.triggered.connect(self._show_about)

    # ------------------------------------------------------------------ Tab 1: Single Video Studio
    def _build_single_tab(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(8)

        # Clean File Selector Row
        file_bar = QHBoxLayout()
        file_bar.setSpacing(8)
        btn_browse = QPushButton("📂 Chọn Video...")
        btn_browse.setCursor(Qt.PointingHandCursor)
        btn_browse.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: white; font-weight: bold; padding: 7px 14px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #1d4ed8; }"
        )
        btn_browse.clicked.connect(self._browse_video)

        self.ed_input = QLineEdit()
        self.ed_input.setPlaceholderText("Đường dẫn video gốc (.mp4, .mkv, .mov...) — có thể kéo thả trực tiếp vào khung xem trước")
        self.ed_input.setReadOnly(True)

        self.ed_output = QLineEdit()
        self.ed_output.setPlaceholderText("File xuất ra (tự động đặt cạnh video gốc)")
        btn_out = QPushButton("Đổi nơi lưu...")
        btn_out.clicked.connect(self._browse_output)

        file_bar.addWidget(btn_browse)
        file_bar.addWidget(self.ed_input, 3)
        file_bar.addWidget(QLabel("Xuất ra:"))
        file_bar.addWidget(self.ed_output, 2)
        file_bar.addWidget(btn_out)
        layout.addLayout(file_bar)

        # Main Horizontal Splitter: Left (Preview + Clean Settings Tabs) | Right (Dialogue Table + Log)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # ==================== LEFT COLUMN ====================
        left_w = QWidget()
        left_layout = QVBoxLayout(left_w)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(8)

        # Video Preview Card
        prev_box = QGroupBox("Xem trước khung hình & Khung quét phụ đề (Kéo thả trực tiếp trên video)")
        prev_v = QVBoxLayout(prev_box)
        prev_v.setSpacing(6)
        self.preview = PreviewLabel()
        self.preview.file_dropped.connect(self._load_video)
        self.preview.roi_changed.connect(self._on_preview_roi_changed)
        prev_v.addWidget(self.preview, 1)

        scrub_row = QHBoxLayout()
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font-family: Consolas; font-weight: bold;")
        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setRange(0, 1000)
        self.slider_time.setEnabled(False)
        self.slider_time.valueChanged.connect(self._on_scrub)

        self.btn_reset_roi = QPushButton("↺ Đặt lại khung")
        self.btn_reset_roi.setToolTip("Đặt lại khung quét phụ đề về vị trí mặc định phía dưới (hoặc nhấp đúp trên video)")
        self.btn_reset_roi.clicked.connect(self._reset_roi_box)

        self.btn_test_blur = QPushButton("🔍 Xem thử xóa phụ đề")
        self.btn_test_blur.setEnabled(False)
        self.btn_test_blur.clicked.connect(self._preview_blur_frame)

        scrub_row.addWidget(self.lbl_time)
        scrub_row.addWidget(self.slider_time, 1)
        scrub_row.addWidget(self.btn_reset_roi)
        scrub_row.addWidget(self.btn_test_blur)
        prev_v.addLayout(scrub_row)

        meta_row = QHBoxLayout()
        self.lbl_meta = QLabel("Chưa tải video")
        self.lbl_meta.setStyleSheet("color: #6b7280; font-size: 12px;")
        self.lbl_roi_info = QLabel("🎯 Khung quét: X 2%→98% · Y 72%→98%")
        self.lbl_roi_info.setStyleSheet("color: #15803d; font-size: 11px; font-weight: 600;")
        meta_row.addWidget(self.lbl_meta, 1)
        meta_row.addWidget(self.lbl_roi_info)
        prev_v.addLayout(meta_row)
        left_layout.addWidget(prev_box, 3)

        # Clean 2-Tab Settings Panel (keeps UI super simple while retaining full power!)
        self.cfg_tabs = QTabWidget()
        tab_quick = QWidget()
        tab_adv = QWidget()
        self._build_quick_settings(tab_quick)
        self._build_advanced_settings(tab_adv)
        self.cfg_tabs.addTab(tab_quick, "⚡ Thiết lập chính (Đơn giản)")
        self.cfg_tabs.addTab(tab_adv, "⚙ Tùy chỉnh nâng cao")
        left_layout.addWidget(self.cfg_tabs, 2)

        splitter.addWidget(left_w)

        # ==================== RIGHT COLUMN ====================
        right_w = QWidget()
        right_layout = QVBoxLayout(right_w)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(8)

        tbl_box = QGroupBox("Bảng lời thoại, Phân vai nhân vật & Bản dịch (Sửa trực tiếp tại đây)")
        tbl_v = QVBoxLayout(tbl_box)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Bắt đầu", "Kết thúc", "Nhân vật / Giọng đọc", "Lời thoại gốc", "Bản dịch lồng tiếng"
        ])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Interactive)
        self.table.setColumnWidth(2, 185)
        hdr.setSectionResizeMode(3, QHeaderView.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.cellClicked.connect(self._jump_to_segment)
        tbl_v.addWidget(self.table, 1)

        tbl_btns = QHBoxLayout()
        btn_add_row = QPushButton("➕ Thêm dòng")
        btn_add_row.clicked.connect(self._add_segment_row)
        btn_del_row = QPushButton("➖ Xóa dòng chọn")
        btn_del_row.clicked.connect(self._del_segment_row)
        btn_all_male = QPushButton("👨 Đưa tất cả về Nam 1")
        btn_all_male.setToolTip("Gộp toàn bộ các câu trong bảng về 1 vai Nam 1 duy nhất")
        btn_all_male.clicked.connect(lambda: self._bulk_set_role("NAM_1"))
        btn_all_female = QPushButton("👩 Đưa tất cả về Nữ 1")
        btn_all_female.setToolTip("Gộp toàn bộ các câu trong bảng về 1 vai Nữ 1 duy nhất")
        btn_all_female.clicked.connect(lambda: self._bulk_set_role("NU_1"))
        btn_save_proj = QPushButton("💾 Lưu chỉnh sửa")
        btn_save_proj.clicked.connect(self._save_edits)
        tbl_btns.addWidget(btn_add_row)
        tbl_btns.addWidget(btn_del_row)
        tbl_btns.addWidget(btn_all_male)
        tbl_btns.addWidget(btn_all_female)
        tbl_btns.addWidget(btn_save_proj)
        tbl_btns.addStretch(1)
        tbl_v.addLayout(tbl_btns)
        right_layout.addWidget(tbl_box, 4)

        log_box = QGroupBox("Nhật ký chi tiết")
        log_v = QVBoxLayout(log_box)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 9))
        self.txt_log.setMaximumBlockCount(1500)
        log_v.addWidget(self.txt_log)
        right_layout.addWidget(log_box, 1)

        splitter.addWidget(right_w)
        splitter.setSizes([580, 780])

        # ── Step-by-Step Visual Progress Tracker ────────────────────────
        self.prog_card = PipelineProgressCard()
        layout.addWidget(self.prog_card)

        # ── Clear Action Buttons Bar ────────────────────────────────────
        act_row = QHBoxLayout()
        act_row.setSpacing(10)

        self.btn_analyze = QPushButton("🔍 Bước 1: Phân tích, Nhận diện vai & Dịch")
        self.btn_analyze.setMinimumHeight(38)
        self.btn_analyze.clicked.connect(lambda: self._start_worker("analyze"))

        self.btn_render = QPushButton("🎙 Bước 2: Lồng tiếng & Xuất Video")
        self.btn_render.setMinimumHeight(38)
        self.btn_render.clicked.connect(lambda: self._start_worker("render"))

        self.btn_both = QPushButton("⚡ CHẠY TỰ ĐỘNG TRỌN GÓI (A → Z)")
        self.btn_both.setMinimumHeight(38)
        self.btn_both.setCursor(Qt.PointingHandCursor)
        self.btn_both.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: white; font-weight: bold; font-size: 13px; padding: 8px 22px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #15803d; }"
            "QPushButton:disabled { background-color: #9ca3af; color: #f3f4f6; }"
        )
        self.btn_both.clicked.connect(lambda: self._start_worker("both"))

        self.btn_cancel = QPushButton("⏹ Dừng lại")
        self.btn_cancel.setMinimumHeight(38)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(
            "QPushButton:enabled { background-color: #fee2e2; color: #dc2626; font-weight: bold; border: 1px solid #fca5a5; }"
        )
        self.btn_cancel.clicked.connect(self._cancel_worker)

        self.btn_open_out = QPushButton("📂 Mở Video kết quả")
        self.btn_open_out.setMinimumHeight(38)
        self.btn_open_out.setEnabled(False)
        self.btn_open_out.clicked.connect(self._open_output_folder)

        act_row.addWidget(self.btn_analyze)
        act_row.addWidget(self.btn_render)
        act_row.addWidget(self.btn_both)
        act_row.addWidget(self.btn_cancel)
        act_row.addStretch(1)
        act_row.addWidget(self.btn_open_out)
        layout.addLayout(act_row)

    def _build_quick_settings(self, parent: QWidget):
        lay = QVBoxLayout(parent)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)

        # Row 0: Processing Mode (Audio Dubbing vs Text-Only Silent Subtitle OCR vs OCR + Dubbing)
        pmode_row = QHBoxLayout()
        self.cb_process_mode = QComboBox()
        self.cb_process_mode.addItem("🎬 Lồng tiếng + Phụ đề (Từ giọng nói gốc)", "dub_audio")
        self.cb_process_mode.addItem("📝 Chỉ dịch phụ đề chữ trên video (Video không tiếng / Chỉ có sub)", "ocr_sub_only")
        self.cb_process_mode.addItem("🎙 Quét chữ sub trên video ➜ Dịch + Lồng tiếng AI + Thay phụ đề", "ocr_sub_dub")
        self.cb_process_mode.currentIndexChanged.connect(self._sync_process_mode_ui)
        pmode_row.addWidget(QLabel("Chế độ xử lý:"))
        pmode_row.addWidget(self.cb_process_mode, 1)
        lay.addLayout(pmode_row)

        self.lbl_ocr_mode_tip = QLabel(
            "💡 Chế độ quét chữ trên video (OCR): Hãy dùng chuột kéo khung màu xanh ở màn hình xem trước "
            "đặt đúng vào vùng chữ phụ đề gốc. Hệ thống sẽ quét chữ ➜ Dịch ➜ Xóa sub cũ ➜ Hiển thị sub mới!"
        )
        self.lbl_ocr_mode_tip.setWordWrap(True)
        self.lbl_ocr_mode_tip.setStyleSheet(
            "background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; "
            "border-radius: 6px; padding: 6px 8px; font-size: 11px;"
        )
        self.lbl_ocr_mode_tip.setVisible(False)
        lay.addWidget(self.lbl_ocr_mode_tip)

        # Row 1: Source -> Target Language
        lang_row = QHBoxLayout()
        self.cb_src = QComboBox()
        self.cb_src.addItem("🌐 Tự động nhận diện (Auto)", "auto")
        for code, lang in LANGUAGES.items():
            self.cb_src.addItem(f"{lang.name}", code)

        self.cb_tgt = QComboBox()
        for code, lang in LANGUAGES.items():
            self.cb_tgt.addItem(f"{lang.name}", code)
        self.cb_tgt.setCurrentIndex(list(LANGUAGES.keys()).index("vi"))
        self.cb_tgt.currentIndexChanged.connect(self._on_target_lang_changed)

        lang_row.addWidget(QLabel("Ngôn ngữ gốc:"))
        lang_row.addWidget(self.cb_src, 1)
        lang_row.addWidget(QLabel("➔  Dịch sang:"))
        lang_row.addWidget(self.cb_tgt, 1)
        lay.addLayout(lang_row)

        # Row 1B: Gemini Script Style & Optional Context/Pronoun Note
        style_row = QHBoxLayout()
        self.cb_trans_style = QComboBox()
        for scode, (stitle, _sdesc) in gemini_translate.TRANSLATION_STYLES.items():
            self.cb_trans_style.addItem(stitle, scode)

        self.ed_trans_notes = QLineEdit()
        self.ed_trans_notes.setPlaceholderText("Ghi chú xưng hô/tên riêng (Tùy chọn, VD: Xưng hô Anh - Em, giữ tên John)")

        style_row.addWidget(QLabel("Văn phong dịch:"))
        style_row.addWidget(self.cb_trans_style, 3)
        style_row.addWidget(self.ed_trans_notes, 2)
        lay.addLayout(style_row)

        # TTS Voice Controls Container (hidden cleanly when in pure Subtitle-Only OCR mode)
        self.w_tts_section = QWidget()
        tts_sec_v = QVBoxLayout(self.w_tts_section)
        tts_sec_v.setContentsMargins(0, 0, 0, 0)
        tts_sec_v.setSpacing(6)

        # Row 2: TTS Engine Selector (OmniVoice AI vs Edge Neural TTS)
        tts_eng_row = QHBoxLayout()
        self.cb_tts_engine = QComboBox()
        self.cb_tts_engine.addItem("🔥 OmniVoice AI — Clone giọng gốc nhân vật & Thiết kế giọng (Đề xuất)", "omnivoice")
        self.cb_tts_engine.addItem("☁️ Edge Neural TTS — 75 giọng Online siêu nhanh", "edge")
        if not omnivoice_tts.is_omnivoice_available():
            self.cb_tts_engine.setCurrentIndex(1)
        self.cb_tts_engine.currentIndexChanged.connect(self._sync_tts_engine_ui)
        tts_eng_row.addWidget(QLabel("Thư viện giọng:"))
        tts_eng_row.addWidget(self.cb_tts_engine, 1)
        tts_sec_v.addLayout(tts_eng_row)

        # Row 3A: OmniVoice Controls (Visible when OmniVoice AI is active)
        self.w_omni_box = QWidget()
        omni_v = QVBoxLayout(self.w_omni_box)
        omni_v.setContentsMargins(0, 0, 0, 0)
        omni_v.setSpacing(6)

        omni_p_row = QHBoxLayout()
        self.cb_omni_preset = QComboBox()
        for pid, plabel, _inst, _gen in omnivoice_tts.OMNIVOICE_PRESETS:
            self.cb_omni_preset.addItem(plabel, pid)
        self.cb_omni_preset.currentIndexChanged.connect(self._sync_tts_engine_ui)

        self.btn_preview_omni = QPushButton("🔊 Nghe thử giọng")
        self.btn_preview_omni.setCursor(Qt.PointingHandCursor)
        self.btn_preview_omni.setStyleSheet(
            "QPushButton { background: #eff6ff; color: #1d4ed8; font-weight: bold; border: 1px solid #bfdbfe; padding: 5px 12px; }"
            "QPushButton:hover { background: #dbeafe; }"
        )
        self.btn_preview_omni.clicked.connect(self._play_voice_preview)

        omni_p_row.addWidget(QLabel("Kiểu giọng AI:"))
        omni_p_row.addWidget(self.cb_omni_preset, 1)
        omni_p_row.addWidget(self.btn_preview_omni)
        omni_v.addLayout(omni_p_row)

        self.w_omni_ref_row = QWidget()
        ref_h = QHBoxLayout(self.w_omni_ref_row)
        ref_h.setContentsMargins(0, 0, 0, 0)
        ref_h.setSpacing(6)
        self.ed_omni_ref = QLineEdit()
        self.ed_omni_ref.setPlaceholderText("Chọn file giọng mẫu rõ ràng dài 3–10 giây (.wav, .mp3, .m4a)...")
        btn_pick_ref = QPushButton("📂 Chọn file giọng mẫu...")
        btn_pick_ref.clicked.connect(self._browse_omni_ref)
        ref_h.addWidget(QLabel("File giọng mẫu:"))
        ref_h.addWidget(self.ed_omni_ref, 1)
        ref_h.addWidget(btn_pick_ref)
        omni_v.addWidget(self.w_omni_ref_row)
        tts_sec_v.addWidget(self.w_omni_box)

        # Row 3B: Edge Neural TTS Controls (Visible when Edge TTS is active)
        self.w_edge_box = QWidget()
        edge_v = QVBoxLayout(self.w_edge_box)
        edge_v.setContentsMargins(0, 0, 0, 0)
        edge_v.setSpacing(6)

        mode_row = QHBoxLayout()
        self.cb_voice_mode = QComboBox()
        self.cb_voice_mode.addItem("🎭 Tự động phân vai Nhân vật + Nam/Nữ/Trẻ em (Khuyên dùng)", "auto_character")
        self.cb_voice_mode.addItem("🎙 Dùng cố định 1 giọng đọc đã chọn bên dưới", "custom")
        self.cb_voice_mode.addItem("👫 Tự động đổi Nam / Nữ cơ bản", "auto")
        self.cb_voice_mode.addItem("👨 Luôn dùng giọng Nam chính", "male")
        self.cb_voice_mode.addItem("👩 Luôn dùng giọng Nữ chính", "female")
        mode_row.addWidget(QLabel("Chế độ phân vai:"))
        mode_row.addWidget(self.cb_voice_mode, 1)
        edge_v.addLayout(mode_row)

        voice_row = QHBoxLayout()
        self.cb_voice_pick = QComboBox()
        self.btn_preview_voice = QPushButton("🔊 Nghe thử giọng")
        self.btn_preview_voice.setCursor(Qt.PointingHandCursor)
        self.btn_preview_voice.setStyleSheet(
            "QPushButton { background: #eff6ff; color: #1d4ed8; font-weight: bold; border: 1px solid #bfdbfe; padding: 5px 12px; }"
            "QPushButton:hover { background: #dbeafe; }"
        )
        self.btn_preview_voice.clicked.connect(self._play_voice_preview)

        voice_row.addWidget(QLabel("Giọng đọc mẫu:"))
        voice_row.addWidget(self.cb_voice_pick, 1)
        voice_row.addWidget(self.btn_preview_voice)
        edge_v.addLayout(voice_row)
        tts_sec_v.addWidget(self.w_edge_box)

        # Row 4: Demucs vocal separator (only applies in audio dubbing mode)
        self.chk_demucs = QCheckBox("🎵 Giữ nguyên nhạc nền & tiếng động gốc (Tách giọng AI Demucs)")
        self.chk_demucs.setChecked(True)
        tts_sec_v.addWidget(self.chk_demucs)

        lay.addWidget(self.w_tts_section)

        sub_quick = QHBoxLayout()
        self.chk_blur = QCheckBox("🧹 Tự động xóa sạch phụ đề gốc")
        self.chk_blur.setChecked(True)
        self.chk_burn = QCheckBox("📝 Chèn phụ đề mới đã dịch")
        self.chk_burn.setChecked(True)
        sub_quick.addWidget(self.chk_blur)
        sub_quick.addWidget(self.chk_burn)
        sub_quick.addStretch(1)
        lay.addLayout(sub_quick)

        lay.addStretch(1)

    def _build_advanced_settings(self, parent: QWidget):
        form = QFormLayout(parent)
        form.setContentsMargins(10, 8, 10, 8)
        form.setSpacing(8)

        # Subtitle Cover & Styling
        cover_row = QHBoxLayout()
        self.cb_blur_style = QComboBox()
        self.cb_blur_style.addItem("✨ Xóa nét chữ thông minh + Làm mờ mịn (Đề xuất)", "inpaint_blur")
        self.cb_blur_style.addItem("🎬 Dải nền tối điện ảnh (Che kín tuyệt đối 100%)", "box")
        self.cb_blur_style.addItem("💧 Làm mờ đa tầng (Heavy Blur)", "blur")
        self.cb_blur_style.addItem("🟦 Làm nhòe ô vuông (Pixelate)", "pixelate")

        self.chk_place_old = QCheckBox("Căn theo vùng sub cũ")
        self.chk_place_old.setChecked(True)
        self.chk_sub_bg = QCheckBox("Khung nền tối dưới chữ mới")
        self.chk_sub_bg.setChecked(False)

        cover_row.addWidget(self.cb_blur_style, 1)
        cover_row.addWidget(self.chk_place_old)
        cover_row.addWidget(self.chk_sub_bg)
        form.addRow("Kiểu che sub cũ:", cover_row)

        # Scan Region (4-sided synced with draggable ROI box) & Font Size
        reg_row = QHBoxLayout()
        self.sp_reg_left = QSpinBox()
        self.sp_reg_left.setRange(0, 90)
        self.sp_reg_left.setSuffix("%")
        self.sp_reg_left.setValue(2)
        self.sp_reg_left.valueChanged.connect(self._refresh_preview_overlay)

        self.sp_reg_right = QSpinBox()
        self.sp_reg_right.setRange(10, 100)
        self.sp_reg_right.setSuffix("%")
        self.sp_reg_right.setValue(98)
        self.sp_reg_right.valueChanged.connect(self._refresh_preview_overlay)

        self.sp_reg_top = QSpinBox()
        self.sp_reg_top.setRange(0, 95)
        self.sp_reg_top.setSuffix("%")
        self.sp_reg_top.setValue(72)
        self.sp_reg_top.valueChanged.connect(self._refresh_preview_overlay)

        self.sp_reg_bot = QSpinBox()
        self.sp_reg_bot.setRange(5, 100)
        self.sp_reg_bot.setSuffix("%")
        self.sp_reg_bot.setValue(98)
        self.sp_reg_bot.valueChanged.connect(self._refresh_preview_overlay)

        self.sp_fontsize = QSpinBox()
        self.sp_fontsize.setRange(0, 140)
        self.sp_fontsize.setValue(0)
        self.sp_fontsize.setSpecialValueText("Tự động")

        self.sp_ocr_fps = QDoubleSpinBox()
        self.sp_ocr_fps.setRange(2.0, 10.0)
        self.sp_ocr_fps.setSingleStep(1.0)
        self.sp_ocr_fps.setValue(4.0)

        reg_row.addWidget(QLabel("X:"))
        reg_row.addWidget(self.sp_reg_left)
        reg_row.addWidget(QLabel("→"))
        reg_row.addWidget(self.sp_reg_right)
        reg_row.addSpacing(4)
        reg_row.addWidget(QLabel("Y:"))
        reg_row.addWidget(self.sp_reg_top)
        reg_row.addWidget(QLabel("→"))
        reg_row.addWidget(self.sp_reg_bot)
        reg_row.addSpacing(6)
        reg_row.addWidget(QLabel("Cỡ chữ:"))
        reg_row.addWidget(self.sp_fontsize)
        reg_row.addSpacing(6)
        reg_row.addWidget(QLabel("OCR fps:"))
        reg_row.addWidget(self.sp_ocr_fps)
        form.addRow("Vùng quét sub:", reg_row)

        # Audio & Anti-Slur Timing
        aud_row = QHBoxLayout()
        self.sp_dub_vol = QDoubleSpinBox()
        self.sp_dub_vol.setRange(0.2, 3.0)
        self.sp_dub_vol.setSingleStep(0.1)
        self.sp_dub_vol.setValue(1.05)

        self.sp_orig_vol = QDoubleSpinBox()
        self.sp_orig_vol.setRange(0.0, 1.0)
        self.sp_orig_vol.setSingleStep(0.05)
        self.sp_orig_vol.setValue(0.0)

        self.sp_gap_ms = QSpinBox()
        self.sp_gap_ms.setRange(100, 600)
        self.sp_gap_ms.setSingleStep(20)
        self.sp_gap_ms.setSuffix(" ms")
        self.sp_gap_ms.setValue(240)
        self.sp_gap_ms.setToolTip("Khoảng nghỉ tự nhiên giữa 2 câu liên tiếp để chống dính chữ (khuyên dùng 240 ms)")

        self.sp_speed = QDoubleSpinBox()
        self.sp_speed.setRange(1.0, 1.8)
        self.sp_speed.setSingleStep(0.04)
        self.sp_speed.setValue(1.28)

        aud_row.addWidget(QLabel("Âm lồng:"))
        aud_row.addWidget(self.sp_dub_vol)
        aud_row.addWidget(QLabel("Giọng gốc:"))
        aud_row.addWidget(self.sp_orig_vol)
        aud_row.addWidget(QLabel("Nghỉ nối câu:"))
        aud_row.addWidget(self.sp_gap_ms)
        aud_row.addWidget(QLabel("Tốc độ max:"))
        aud_row.addWidget(self.sp_speed)
        form.addRow("Nhịp & Âm lượng:", aud_row)

        # AI Models & Translation Engine
        eng_row = QHBoxLayout()
        self.cb_trans_engine = QComboBox()
        self.cb_trans_engine.addItem("✨ Gemini AI (Dịch kịch bản chuẩn văn phong - Đề xuất)", "gemini")
        self.cb_trans_engine.addItem("💻 NLLB-200 Offline (Dịch máy cục bộ không cần API)", "nllb")

        self.chk_gem_fallback = QCheckBox("Dự phòng NLLB nếu hết lượt Gemini")
        self.chk_gem_fallback.setChecked(True)

        eng_row.addWidget(self.cb_trans_engine, 1)
        eng_row.addWidget(self.chk_gem_fallback)
        form.addRow("Bộ máy dịch:", eng_row)

        mdl_row = QHBoxLayout()
        self.cb_whisper = QComboBox()
        for m in WHISPER_MODELS:
            self.cb_whisper.addItem(m, m)
        self.cb_whisper.setCurrentText("medium")

        self.cb_omni_steps = QComboBox()
        self.cb_omni_steps.addItem("16 bước (Nhanh & rõ - Khuyên dùng)", 16)
        self.cb_omni_steps.addItem("24 bước (Chất lượng cao)", 24)
        self.cb_omni_steps.addItem("32 bước (Studio tối đa)", 32)
        self.cb_omni_steps.setToolTip("Số bước khuếch tán của OmniVoice AI (16 bước đã rất tự nhiên và nhanh)")

        self.cb_nllb = QComboBox()
        for label, model_id in NLLB_MODELS.items():
            self.cb_nllb.addItem(label, model_id)

        self.chk_gpu = QCheckBox("Tăng tốc GPU CUDA")
        self.chk_gpu.setChecked(True)

        mdl_row.addWidget(QLabel("Whisper:"))
        mdl_row.addWidget(self.cb_whisper)
        mdl_row.addWidget(QLabel("OmniVoice:"))
        mdl_row.addWidget(self.cb_omni_steps)
        mdl_row.addWidget(QLabel("NLLB:"))
        mdl_row.addWidget(self.cb_nllb, 1)
        mdl_row.addWidget(self.chk_gpu)
        form.addRow("Mô hình Offline:", mdl_row)

    # ------------------------------------------------------------------ Tab 2: Batch Queue
    def _build_batch_tab(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        bar = QHBoxLayout()
        btn_add_files = QPushButton("➕ Thêm nhiều video...")
        btn_add_files.setStyleSheet("font-weight: bold;")
        btn_add_files.clicked.connect(self._batch_add_files)

        btn_add_folder = QPushButton("📁 Quét cả thư mục...")
        btn_add_folder.clicked.connect(self._batch_add_folder)

        btn_del_item = QPushButton("➖ Xóa mục chọn")
        btn_del_item.clicked.connect(self._batch_remove_selected)

        btn_clear = QPushButton("🧹 Xóa danh sách")
        btn_clear.clicked.connect(self._batch_clear)

        bar.addWidget(btn_add_files)
        bar.addWidget(btn_add_folder)
        bar.addWidget(btn_del_item)
        bar.addWidget(btn_clear)
        bar.addSpacing(14)

        bar.addWidget(QLabel("Thư mục lưu kết quả:"))
        self.ed_batch_out = QLineEdit()
        self.ed_batch_out.setPlaceholderText("Để trống = Lưu ngay cạnh từng file video gốc")
        btn_browse_batch_out = QPushButton("Chọn thư mục...")
        btn_browse_batch_out.clicked.connect(self._batch_browse_out_dir)
        bar.addWidget(self.ed_batch_out, 1)
        bar.addWidget(btn_browse_batch_out)
        layout.addLayout(bar)

        self.batch_table = BatchTableWidget(0, 6)
        self.batch_table.setHorizontalHeaderLabels([
            "#", "Tên video", "Thời lượng", "Bước hiện tại", "Tiến trình file", "Đường dẫn gốc"
        ])
        b_hdr = self.batch_table.horizontalHeader()
        b_hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        b_hdr.setSectionResizeMode(1, QHeaderView.Interactive)
        self.batch_table.setColumnWidth(1, 270)
        b_hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        b_hdr.setSectionResizeMode(3, QHeaderView.Interactive)
        self.batch_table.setColumnWidth(3, 190)
        b_hdr.setSectionResizeMode(4, QHeaderView.Stretch)
        b_hdr.setSectionResizeMode(5, QHeaderView.Stretch)
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.files_dropped.connect(self._batch_add_paths)
        layout.addWidget(self.batch_table, 3)

        b_log_box = QGroupBox("Nhật ký chạy hàng loạt")
        b_log_layout = QVBoxLayout(b_log_box)
        self.txt_batch_log = QPlainTextEdit()
        self.txt_batch_log.setReadOnly(True)
        self.txt_batch_log.setFont(QFont("Consolas", 9))
        b_log_layout.addWidget(self.txt_batch_log)
        layout.addWidget(b_log_box, 1)

        # Overall Batch Progress Card
        b_card = QFrame()
        b_card.setObjectName("ProgressCard")
        bc_lay = QVBoxLayout(b_card)
        bc_lay.setContentsMargins(14, 8, 14, 8)
        top_b = QHBoxLayout()
        self.lbl_batch_status = QLabel("Danh sách trống · Thêm hoặc kéo thả nhiều video vào bảng trên")
        self.lbl_batch_status.setStyleSheet("font-weight: bold; color: #111827;")
        self.lbl_batch_pct = QLabel("0%")
        self.lbl_batch_pct.setStyleSheet("font-weight: bold; font-size: 14px; color: #16a34a;")
        top_b.addWidget(self.lbl_batch_status)
        top_b.addStretch(1)
        top_b.addWidget(self.lbl_batch_pct)
        bc_lay.addLayout(top_b)

        self.progress_batch = QProgressBar()
        self.progress_batch.setRange(0, 1000)
        self.progress_batch.setValue(0)
        self.progress_batch.setTextVisible(False)
        self.progress_batch.setFixedHeight(10)
        self.progress_batch.setStyleSheet(
            "QProgressBar { background-color: #e5e7eb; border: none; border-radius: 5px; }"
            "QProgressBar::chunk { background-color: #16a34a; border-radius: 5px; }"
        )
        bc_lay.addWidget(self.progress_batch)
        layout.addWidget(b_card)

        b_act_layout = QHBoxLayout()
        self.btn_batch_start = QPushButton("▶ BẮT ĐẦU CHẠY TOÀN BỘ DANH SÁCH")
        self.btn_batch_start.setCursor(Qt.PointingHandCursor)
        self.btn_batch_start.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: white; font-weight: bold; font-size: 13px; padding: 10px 24px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #15803d; }"
            "QPushButton:disabled { background-color: #9ca3af; }"
        )
        self.btn_batch_start.clicked.connect(self._batch_start)

        self.btn_batch_stop = QPushButton("⏹ Dừng danh sách")
        self.btn_batch_stop.setEnabled(False)
        self.btn_batch_stop.setStyleSheet("padding: 10px 20px; font-weight: bold;")
        self.btn_batch_stop.clicked.connect(self._batch_cancel)

        btn_open_batch_dir = QPushButton("📁 Mở thư mục kết quả")
        btn_open_batch_dir.clicked.connect(self._batch_open_output_dir)

        b_act_layout.addWidget(self.btn_batch_start)
        b_act_layout.addWidget(self.btn_batch_stop)
        b_act_layout.addStretch()
        b_act_layout.addWidget(btn_open_batch_dir)
        layout.addLayout(b_act_layout)

    # ------------------------------------------------------------------ Theme & Styling
    def _apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow { background-color: #f3f4f6; }
            QWidget { font-family: 'Segoe UI', Arial; font-size: 13px; color: #1f2937; }
            QFrame#HeaderBar { background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; }
            QFrame#ProgressCard { background-color: #ffffff; border: 1px solid #dbeafe; border-radius: 8px; }
            QGroupBox {
                background-color: #ffffff;
                font-weight: bold;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #111827; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 5px;
                padding: 4px 8px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1.5px solid #2563eb;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                padding: 6px 12px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #f9fafb; }
            QTabWidget::pane { border: 1px solid #e5e7eb; border-radius: 6px; background: #ffffff; }
            QTabBar::tab {
                background: #e5e7eb; color: #374151;
                padding: 7px 16px; margin-right: 4px;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
                font-weight: 600;
            }
            QTabBar::tab:selected { background: #2563eb; color: #ffffff; }
            QTableWidget { border: 1px solid #e5e7eb; border-radius: 6px; gridline-color: #f3f4f6; }
            QHeaderView::section {
                background-color: #f9fafb; font-weight: bold; color: #374151;
                border: none; border-bottom: 1px solid #e5e7eb; padding: 6px;
            }
            """
        )

    # ------------------------------------------------------------------ Voice Catalog & Live Preview
    def _on_target_lang_changed(self):
        self._refresh_voice_combo()
        self._update_default_output()
        # Refresh per-row voice dropdowns while preserving speaker tags
        segs = self._collect_segments()
        if segs:
            self._populate_table(segs)

    def _refresh_voice_combo(self):
        lang_code = self.cb_tgt.currentData() or "vi"
        voices = get_voices_for_lang(lang_code)
        self.cb_voice_pick.clear()
        for v in voices:
            icon = "👩" if v.gender == "female" else ("🧒" if v.gender == "child" else "👨")
            self.cb_voice_pick.addItem(f"{icon} {v.name}", encode_voice_spec(v))

    def _sync_process_mode_ui(self):
        pmode = self.cb_process_mode.currentData() or "dub_audio"
        is_ocr_only = pmode == "ocr_sub_only"
        is_ocr_any = pmode in ("ocr_sub_only", "ocr_sub_dub")

        self.w_tts_section.setVisible(not is_ocr_only)
        self.chk_demucs.setVisible(pmode == "dub_audio")
        self.lbl_ocr_mode_tip.setVisible(is_ocr_any)

        if is_ocr_only:
            self.chk_blur.setChecked(True)
            self.chk_burn.setChecked(True)
            self.btn_analyze.setText("🔍 Bước 1: Quét chữ phụ đề & Dịch")
            self.btn_render.setText("📝 Bước 2: Xóa sub cũ & Ghép phụ đề mới")
        elif pmode == "ocr_sub_dub":
            self.chk_blur.setChecked(True)
            self.chk_burn.setChecked(True)
            self.btn_analyze.setText("🔍 Bước 1: Quét chữ phụ đề & Dịch")
            self.btn_render.setText("🎙 Bước 2: Lồng tiếng & Xuất Video")
        else:
            self.btn_analyze.setText("🔍 Bước 1: Phân tích, Nhận diện vai & Dịch")
            self.btn_render.setText("🎙 Bước 2: Lồng tiếng & Xuất Video")

        self._update_default_output()

    def _sync_tts_engine_ui(self):
        eng = self.cb_tts_engine.currentData() or "omnivoice"
        is_omni = eng == "omnivoice"
        self.w_omni_box.setVisible(is_omni)
        self.w_edge_box.setVisible(not is_omni)
        if is_omni:
            preset = self.cb_omni_preset.currentData() or "design_auto"
            self.w_omni_ref_row.setVisible(preset == "clone_file")

    def _browse_omni_ref(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file giọng nói mẫu (3–10 giây rõ lời)",
            "",
            "Âm thanh / Video (*.wav *.mp3 *.m4a *.ogg *.flac *.mp4);;Tất cả (*.*)",
        )
        if p:
            self.ed_omni_ref.setText(p)

    def _play_voice_preview(self):
        eng = self.cb_tts_engine.currentData() or "omnivoice"
        lang_code = self.cb_tgt.currentData() or "vi"
        sample_txt = SAMPLE_PREVIEW_TEXT.get(lang_code, SAMPLE_PREVIEW_TEXT["en"])

        if eng == "omnivoice":
            preset = self.cb_omni_preset.currentData() or "design_auto"
            ref_path = self.ed_omni_ref.text().strip()
            if preset == "clone_file" and (not ref_path or not os.path.isfile(ref_path)):
                QMessageBox.information(
                    self,
                    "Chưa chọn file giọng mẫu",
                    "Vui lòng bấm '📂 Chọn file giọng mẫu...' trước khi nghe thử chế độ Clone từ file.",
                )
                return
            self.btn_preview_omni.setEnabled(False)
            self.btn_preview_omni.setText("⏳ Đang tạo mẫu AI...")
            self._preview_worker = VoicePreviewWorker(
                engine="omnivoice",
                voice_or_preset=preset,
                text=sample_txt,
                target_lang=lang_code,
                ref_audio_path=ref_path,
                use_gpu=self.chk_gpu.isChecked(),
                omni_steps=int(self.cb_omni_steps.currentData() or 16),
            )
        else:
            voice_spec = self.cb_voice_pick.currentData()
            if not voice_spec:
                return
            self.btn_preview_voice.setEnabled(False)
            self.btn_preview_voice.setText("⏳ Đang tạo mẫu...")
            self._preview_worker = VoicePreviewWorker(
                engine="edge",
                voice_or_preset=voice_spec,
                text=sample_txt,
                target_lang=lang_code,
            )

        self._preview_worker.done_sig.connect(self._on_voice_preview_done)
        self._preview_worker.start()

    def _on_voice_preview_done(self, ok: bool, err: str):
        self.btn_preview_voice.setEnabled(True)
        self.btn_preview_voice.setText("🔊 Nghe thử giọng")
        self.btn_preview_omni.setEnabled(True)
        self.btn_preview_omni.setText("🔊 Nghe thử giọng")
        if not ok:
            QMessageBox.warning(self, "Lỗi nghe thử giọng", f"Không tạo được giọng mẫu:\n{err}")

    # ------------------------------------------------------------------ Account Security & Auto-Updater
    def _run_heartbeat_check(self):
        if self._signing_out:
            return
        if self._verify_worker and self._verify_worker.isRunning():
            return
        self._verify_worker = SessionVerifyWorker()
        self._verify_worker.result_sig.connect(self._on_heartbeat_result)
        self._verify_worker.start()

    def _on_heartbeat_result(self, ok: bool, msg: str):
        if self._signing_out or ok:
            return
        # If account was disabled/locked/deleted on Firebase -> immediately revoke & kick out!
        if "khóa" in msg or "vô hiệu hóa" in msg or "tồn tại" in msg or "hết hiệu lực" in msg or "MÁY KHÁC" in msg:
            self._force_kick_to_login(msg)

    def _force_kick_to_login(self, reason_msg: str):
        if self._signing_out:
            return
        self._signing_out = True
        self._auth_timer.stop()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1500)
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
            self.batch_worker.wait(1500)
        auth.logout()
        if self._tray is not None:
            self._tray.dispose()
            self._tray = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.hide()
        self.logout_requested.emit(reason_msg)

    def _logout_clicked(self):
        ret = QMessageBox.question(
            self,
            "Xác nhận đăng xuất",
            "Bạn có chắc chắn muốn đăng xuất khỏi tài khoản hiện tại?\n"
            "(Sau khi đăng xuất, bạn sẽ phải đăng nhập lại để sử dụng ứng dụng).",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret == QMessageBox.Yes:
            self._force_kick_to_login("Đã đăng xuất tài khoản thành công.")

    @staticmethod
    def _open_nvidia_driver_page():
        import webbrowser
        webbrowser.open("https://www.nvidia.com/Download/index.aspx")

    def _copy_machine_id(self):
        mid = auth.get_short_machine_id()
        QApplication.clipboard().setText(mid)
        QMessageBox.information(self, "Mã thiết bị", f"Đã sao chép Mã thiết bị vào bộ nhớ tạm:\n\n{mid}")

    def _refresh_gemini_badge(self):
        has_key = bool(gemini_translate.parse_api_keys(auth.get_account_gemini_key()))
        if has_key:
            self.btn_gemini_key.setText("🔑 Gemini API: Đã lưu")
            self.btn_gemini_key.setStyleSheet(
                "QPushButton { background: #dcfce7; color: #15803d; border: 1px solid #86efac; "
                "padding: 5px 12px; border-radius: 6px; font-weight: bold; font-size: 12px; }"
                "QPushButton:hover { background: #bbf7d0; }"
            )
        else:
            self.btn_gemini_key.setText("🔑 Nhập Gemini API")
            self.btn_gemini_key.setStyleSheet(
                "QPushButton { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; "
                "padding: 5px 12px; border-radius: 6px; font-weight: bold; font-size: 12px; }"
                "QPushButton:hover { background: #fef3c7; }"
            )

    def _open_gemini_key_dialog(self) -> bool:
        dlg = GeminiKeyDialog(self)
        res = dlg.exec() == QDialog.Accepted
        self._refresh_gemini_badge()
        return res

    def _ensure_gemini_key_if_needed(self, settings: Settings, will_translate: bool) -> bool:
        if not will_translate or settings.translation_engine != "gemini":
            return True
        if gemini_translate.parse_api_keys(auth.get_account_gemini_key()):
            return True
        QMessageBox.information(
            self,
            "Cần nhập khóa Gemini API",
            "Bạn đang chọn dịch kịch bản chuẩn văn phong bằng Gemini AI.\n"
            "Vui lòng nhập khóa Gemini API của bạn (khóa sẽ được tự động lưu theo tài khoản).",
        )
        self._open_gemini_key_dialog()
        return bool(gemini_translate.parse_api_keys(auth.get_account_gemini_key()))

    def _check_updates(self, silent: bool = False):
        if self._pending_update_info and self._pending_update_info.get("has_update") and not silent:
            dlg = updater.UpdateDialog(APP_VERSION, self._pending_update_info, self)
            dlg.exec()
            return

        if not silent:
            self.btn_check_upd.setEnabled(False)
            self.btn_check_upd.setText("⏳ Đang kiểm tra...")

        self._upd_worker = updater.UpdateCheckWorker(APP_VERSION)
        self._upd_worker.finished_sig.connect(lambda res, s=silent: self._on_update_checked(res, s))
        self._upd_worker.start()

    def _on_update_checked(self, res: dict, silent: bool):
        self.btn_check_upd.setEnabled(True)
        err = res.get("error", "")

        if err == "NO_REPO_CONFIGURED":
            self.btn_check_upd.setText("🔄 Kiểm tra cập nhật")
            if not silent:
                QMessageBox.information(
                    self,
                    "Chưa cấu hình kho cập nhật",
                    "Chưa thiết lập kho GitHub nhận bản cập nhật.",
                )
            return

        if err:
            self.btn_check_upd.setText("🔄 Kiểm tra cập nhật")
            if not silent:
                QMessageBox.warning(self, "Kiểm tra cập nhật", err)
            return

        if res.get("has_update"):
            self._pending_update_info = res
            latest = res.get("latest_version", "")
            self.btn_check_upd.setText(f"🚀 Cập nhật v{latest}!")
            self.btn_check_upd.setStyleSheet(
                "QPushButton { background: #dcfce7; color: #15803d; border: 1px solid #86efac; "
                "padding: 5px 12px; border-radius: 6px; font-weight: bold; font-size: 12px; }"
                "QPushButton:hover { background: #bbf7d0; }"
            )
            dlg = updater.UpdateDialog(APP_VERSION, res, self)
            dlg.exec()
        else:
            self.btn_check_upd.setText("🔄 Kiểm tra cập nhật")
            if not silent:
                QMessageBox.information(
                    self,
                    "Đã là bản mới nhất",
                    f"✅ Bạn đang sử dụng phiên bản mới nhất (v{APP_VERSION})!",
                )

    # ------------------------------------------------------------------ Video & Preview
    def _browse_video(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Chọn video đầu vào", "",
            "Video (*.mp4 *.mkv *.mov *.avi *.webm *.flv *.m4v *.ts);;Tất cả (*.*)"
        )
        if p:
            self._load_video(p)

    def _browse_output(self):
        p, _ = QFileDialog.getSaveFileName(self, "Lưu video đầu ra", self.ed_output.text(), "MP4 Video (*.mp4)")
        if p:
            self.ed_output.setText(p)

    def _load_video(self, path: str):
        try:
            info = media.probe(path)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi đọc video", str(e))
            return
        self.video_path = str(Path(path).resolve())
        self.video_info = info
        self.ed_input.setText(self.video_path)

        dur = info["duration"]
        self.lbl_meta.setText(
            f"📐 {info['width']}×{info['height']}  ·  🎞 {info['fps']:.2f} fps  ·  "
            f"⏱ {_fmt_mmss(dur)} ({dur:.1f}s)  ·  🔊 {'Có âm thanh' if info['has_audio'] else 'Không có âm thanh (Chỉ hình)'}"
        )
        if not info["has_audio"] and (self.cb_process_mode.currentData() or "dub_audio") == "dub_audio":
            idx_ocr = self.cb_process_mode.findData("ocr_sub_only")
            if idx_ocr >= 0:
                self.cb_process_mode.setCurrentIndex(idx_ocr)
                self._log("ℹ Video không có âm thanh — đã tự động chuyển sang chế độ 'Chỉ dịch phụ đề chữ trên video (OCR)'.")
        self._update_default_output()

        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.video_path)
        self.slider_time.setEnabled(True)
        self.btn_test_blur.setEnabled(True)
        self.slider_time.setValue(min(200, 1000))
        self._on_scrub(self.slider_time.value())

        proj = pipeline.load_project(self.video_path)
        if proj:
            ps = proj.get("settings") or {}
            if "sub_region_top" in ps and "sub_region_bottom" in ps:
                self.preview.set_roi(
                    float(ps.get("sub_region_left", 0.02)),
                    float(ps.get("sub_region_top", 0.72)),
                    float(ps.get("sub_region_right", 0.98)),
                    float(ps.get("sub_region_bottom", 0.98)),
                    emit=True,
                )
            if proj.get("segments"):
                self._populate_table(proj["segments"])
                self._log(f"📂 Đã tải lại bản dịch lưu sẵn ({len(proj['segments'])} câu).")
            else:
                self.table.setRowCount(0)
        else:
            self.table.setRowCount(0)

    def _update_default_output(self):
        if not self.video_path:
            return
        tgt = self.cb_tgt.currentData() or "vi"
        pmode = self.cb_process_mode.currentData() or "dub_audio"
        suffix = "sub" if pmode == "ocr_sub_only" else "dub"
        vp = Path(self.video_path)
        self.ed_output.setText(str(vp.with_name(f"{vp.stem}_{tgt}_{suffix}.mp4")))

    def _read_current_frame(self):
        if self._cap is None or not self.video_info:
            return None, 0.0
        dur = max(0.01, self.video_info["duration"])
        t = (self.slider_time.value() / 1000.0) * dur
        self._cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None, t
        return frame, t

    def _on_scrub(self, _val):
        frame, t = self._read_current_frame()
        dur = self.video_info.get("duration", 0.0)
        self.lbl_time.setText(f"{_fmt_mmss(t)} / {_fmt_mmss(dur)}")
        if frame is not None:
            self._draw_scan_region_and_show(frame)

    def _on_preview_roi_changed(self, left: float, top: float, right: float, bottom: float):
        li = int(round(left * 100))
        ti = int(round(top * 100))
        ri = int(round(right * 100))
        bi = int(round(bottom * 100))
        for sp, val in (
            (self.sp_reg_left, li),
            (self.sp_reg_right, ri),
            (self.sp_reg_top, ti),
            (self.sp_reg_bot, bi),
        ):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        self.lbl_roi_info.setText(f"🎯 Khung quét: X {li}%→{ri}% · Y {ti}%→{bi}%")

    def _reset_roi_box(self):
        self.preview.set_roi(0.02, 0.72, 0.98, 0.98, emit=True)

    def _refresh_preview_overlay(self):
        if self.sp_reg_left.value() >= self.sp_reg_right.value() - 7:
            self.sp_reg_right.blockSignals(True)
            self.sp_reg_right.setValue(min(100, self.sp_reg_left.value() + 8))
            self.sp_reg_right.blockSignals(False)
        if self.sp_reg_top.value() >= self.sp_reg_bot.value() - 4:
            self.sp_reg_bot.blockSignals(True)
            self.sp_reg_bot.setValue(min(100, self.sp_reg_top.value() + 5))
            self.sp_reg_bot.blockSignals(False)
        l = self.sp_reg_left.value() / 100.0
        r = self.sp_reg_right.value() / 100.0
        t = self.sp_reg_top.value() / 100.0
        b = self.sp_reg_bot.value() / 100.0
        self.preview.set_roi(l, t, r, b, emit=False)
        self.lbl_roi_info.setText(
            f"🎯 Khung quét: X {self.sp_reg_left.value()}%→{self.sp_reg_right.value()}% · "
            f"Y {self.sp_reg_top.value()}%→{self.sp_reg_bot.value()}%"
        )

    def _draw_scan_region_and_show(self, frame: np.ndarray):
        self.preview.set_frame(frame)

    def _preview_blur_frame(self):
        frame, t = self._read_current_frame()
        if frame is None:
            return
        try:
            from dubber.subtitles import (
                Blurrer,
                _detect_boxes_with_engine,
                _get_ocr_engine,
                _merge_lines,
                get_roi_bounds,
                recognize_crop_rapidocr,
            )

            if self._ocr_detector is None:
                self._ocr_detector = _get_ocr_engine()
            engine = self._ocr_detector
            settings = self._collect_settings()

            h, w = frame.shape[:2]
            scale = 1.0
            small = frame
            if w > 960:
                nw = 960
                nh = int(round(h * nw / w))
                small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                scale = w / nw
            sh, sw = small.shape[:2]
            x0, y0, x1, y1 = get_roi_bounds(settings, sw, sh)
            custom_x = (settings.sub_region_left > 0.05) or (settings.sub_region_right < 0.95)
            crop = small[y0:y1, x0:x1]
            boxes = []
            if crop.size:
                for bx1, by1, bx2, by2 in _detect_boxes_with_engine(engine, crop):
                    bh = by2 - by1
                    bw = bx2 - bx1
                    if 0.015 * sh <= bh <= 0.24 * sh and bw >= 0.85 * bh:
                        boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])

            blocks = _merge_lines(boxes, W=sw, H=sh, custom_x_roi=custom_x) if boxes else []
            ocr_preview_txt = recognize_crop_rapidocr(engine, crop) if blocks else ""
            for B in blocks:
                B["rect"] = [v * scale for v in B["rect"]]
                B["line_h"] *= scale
            samples = [{"t": 0.0, "blocks": blocks}]
            blurrer = Blurrer(samples, 5.0, w, h, style=self.cb_blur_style.currentData())
            out = blurrer.apply(frame, 0.0)
            for rx1, ry1, rx2, ry2 in blurrer.rects_at(0.0):
                cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 210, 255), max(2, h // 360))
            self.preview.set_frame(out)
            txt_hint = f' — Chữ OCR sơ bộ: "{ocr_preview_txt}"' if ocr_preview_txt else ""
            self._log(f"🔍 Xem thử xóa phụ đề tại {t:.2f}s: đã xử lý sạch {len(blocks)} cụm chữ{txt_hint}.")
        except Exception as e:
            QMessageBox.warning(self, "Lỗi xem thử OCR", str(e))

    # ------------------------------------------------------------------ Dialogue & Character Table
    def _populate_table(self, segments: list[dict]):
        self.table.setRowCount(0)
        for seg in segments:
            self._insert_row(seg)

    def _insert_row(self, seg: dict):
        r = self.table.rowCount()
        self.table.insertRow(r)
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        gender = seg.get("gender", "male")
        spk = seg.get("speaker", "NAM_1" if gender == "male" else ("TRE_1" if gender == "child" else "NU_1"))
        spk_lbl = seg.get("speaker_label", "Nam 1" if gender == "male" else ("Trẻ em" if gender == "child" else "Nữ 1"))
        voice_ov = seg.get("voice_override", "auto")

        self.table.setItem(r, 0, QTableWidgetItem(f"{start:.2f}"))
        self.table.setItem(r, 1, QTableWidgetItem(f"{end:.2f}"))

        cb = QComboBox()
        # Character role presets
        role_options = [
            ("📝 Phụ đề OCR", ("female", "SUB_1", "📝 Phụ đề OCR", "auto")),
            ("👨 Nam 1 (Chính)", ("male", "NAM_1", "Nam 1", "auto")),
            ("👨 Nam 2 (Phụ/Trầm)", ("male", "NAM_2", "Nam 2", "auto")),
            ("👨 Nam 3", ("male", "NAM_3", "Nam 3", "auto")),
            ("👩 Nữ 1 (Chính)", ("female", "NU_1", "Nữ 1", "auto")),
            ("👩 Nữ 2 (Phụ/Trẻ)", ("female", "NU_2", "Nữ 2", "auto")),
            ("🧒 Trẻ em / Giọng cao", ("child", "TRE_1", "Trẻ em", "auto")),
        ]
        selected_idx = 1  # Default Nam 1 when unspecified
        for idx, (label, meta) in enumerate(role_options):
            cb.addItem(label, meta)
            if voice_ov in ("", "auto") and meta[1] == spk:
                selected_idx = idx

        # Also allow picking a specific neural voice directly on any individual line!
        lang_code = self.cb_tgt.currentData() or "vi"
        for v in get_voices_for_lang(lang_code):
            spec = encode_voice_spec(v)
            short_nm = v.name.split("—")[0].strip()
            cb.addItem(f"🎙 {short_nm}", (v.gender, spk, spk_lbl, spec))
            if voice_ov == spec:
                selected_idx = cb.count() - 1

        cb.setCurrentIndex(selected_idx)
        self.table.setCellWidget(r, 2, cb)

        it_src = QTableWidgetItem(seg.get("text", ""))
        it_src.setBackground(QColor("#f9fafb"))
        self.table.setItem(r, 3, it_src)
        self.table.setItem(r, 4, QTableWidgetItem(seg.get("translation", "")))

    def _collect_segments(self) -> list[dict]:
        segs = []
        for r in range(self.table.rowCount()):
            try:
                start = float(self.table.item(r, 0).text())
                end = float(self.table.item(r, 1).text())
            except (ValueError, AttributeError):
                continue
            cb = self.table.cellWidget(r, 2)
            meta = cb.currentData() if isinstance(cb, QComboBox) else ("male", "NAM_1", "Nam 1", "auto")
            gender, spk, spk_lbl, voice_ov = meta
            src_txt = self.table.item(r, 3).text().strip() if self.table.item(r, 3) else ""
            trn_txt = self.table.item(r, 4).text().strip() if self.table.item(r, 4) else ""
            if end > start:
                segs.append({
                    "start": start,
                    "end": end,
                    "gender": gender,
                    "speaker": spk,
                    "speaker_label": spk_lbl,
                    "voice_override": voice_ov,
                    "text": src_txt,
                    "translation": trn_txt,
                })
        segs.sort(key=lambda s: s["start"])
        return segs

    def _add_segment_row(self):
        last_end = 0.0
        if self.table.rowCount() > 0:
            try:
                last_end = float(self.table.item(self.table.rowCount() - 1, 1).text())
            except Exception:
                pass
        self._insert_row({"start": last_end + 0.15, "end": last_end + 2.65, "gender": "male", "speaker": "NAM_1"})

    def _del_segment_row(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _bulk_set_role(self, spk_code: str):
        count = 0
        for r in range(self.table.rowCount()):
            cb = self.table.cellWidget(r, 2)
            if isinstance(cb, QComboBox):
                for i in range(cb.count()):
                    meta = cb.itemData(i)
                    if isinstance(meta, tuple) and len(meta) >= 4 and meta[1] == spk_code and meta[3] == "auto":
                        cb.setCurrentIndex(i)
                        count += 1
                        break
        if count:
            lbl = "Nam 1" if spk_code == "NAM_1" else "Nữ 1"
            self._log(f"🔁 Đã đồng bộ toàn bộ {count} câu thoại về cùng 1 vai [{lbl}].")

    def _save_edits(self):
        if not self.video_path:
            return
        segs = self._collect_segments()
        s = self._collect_settings()
        proj = pipeline.load_project(self.video_path) or {}
        pipeline.save_project(self.video_path, s, segs, proj.get("source_lang", s.source_lang))
        self._log(f"💾 Đã lưu {len(segs)} câu thoại & phân vai vào project.json.")

    def _jump_to_segment(self, row: int, _col: int):
        if not self.video_info:
            return
        try:
            t = float(self.table.item(row, 0).text())
            dur = max(0.01, self.video_info["duration"])
            self.slider_time.setValue(int(np.clip(t / dur * 1000, 0, 1000)))
        except Exception:
            pass

    # ------------------------------------------------------------------ Single Video Execution
    def _collect_settings(self) -> Settings:
        l, t, r, b = self.preview.get_roi()
        return Settings(
            process_mode=self.cb_process_mode.currentData() or "dub_audio",
            source_lang=self.cb_src.currentData(),
            target_lang=self.cb_tgt.currentData(),
            whisper_model=self.cb_whisper.currentData(),
            translation_engine=self.cb_trans_engine.currentData(),
            translation_style=self.cb_trans_style.currentData(),
            translation_notes=self.ed_trans_notes.text().strip(),
            gemini_fallback_nllb=self.chk_gem_fallback.isChecked(),
            nllb_model=self.cb_nllb.currentData(),
            use_gpu=self.chk_gpu.isChecked(),
            separate_vocals=self.chk_demucs.isChecked(),
            original_vocals_volume=float(self.sp_orig_vol.value()),
            dub_volume=float(self.sp_dub_vol.value()),
            max_speedup=float(self.sp_speed.value()),
            min_gap_ms=int(self.sp_gap_ms.value()),
            tts_engine=self.cb_tts_engine.currentData() or "omnivoice",
            omnivoice_preset=self.cb_omni_preset.currentData() or "design_auto",
            omnivoice_ref_audio=self.ed_omni_ref.text().strip(),
            omnivoice_steps=int(self.cb_omni_steps.currentData() or 16),
            voice_mode=self.cb_voice_mode.currentData(),
            custom_voice=self.cb_voice_pick.currentData() or "",
            blur_old_subs=self.chk_blur.isChecked(),
            burn_new_subs=self.chk_burn.isChecked(),
            sub_region_left=float(l),
            sub_region_right=float(r),
            sub_region_top=float(t),
            sub_region_bottom=float(b),
            ocr_fps=float(self.sp_ocr_fps.value()),
            blur_style=self.cb_blur_style.currentData(),
            sub_font_size=int(self.sp_fontsize.value()),
            sub_place_on_old=self.chk_place_old.isChecked(),
            sub_bg_box=self.chk_sub_bg.isChecked(),
        )

    def _validate_tts_settings(self, settings: Settings) -> bool:
        if getattr(settings, "process_mode", "dub_audio") == "ocr_sub_only":
            return True
        if settings.tts_engine == "omnivoice" and settings.omnivoice_preset == "clone_file":
            if not settings.omnivoice_ref_audio or not os.path.isfile(settings.omnivoice_ref_audio):
                QMessageBox.warning(
                    self,
                    "Chưa chọn file giọng mẫu",
                    "Bạn đang chọn chế độ OmniVoice 'Clone từ file giọng mẫu' nhưng chưa chọn file hợp lệ.\n"
                    "Vui lòng chọn file .wav / .mp3 (3–10 giây) hoặc đổi sang 'Tự động Clone giọng gốc nhân vật'.",
                )
                return False
        return True

    def _set_running_ui(self, running: bool):
        self.btn_analyze.setEnabled(not running)
        self.btn_render.setEnabled(not running)
        self.btn_both.setEnabled(not running)
        self.btn_cancel.setEnabled(running)

    def _start_worker(self, mode: str):
        if not self.video_path:
            QMessageBox.information(self, "Chưa chọn video", "Vui lòng chọn một file video đầu vào trước.")
            return
        segs = []
        if mode == "render":
            segs = self._collect_segments()
            if not segs:
                QMessageBox.information(
                    self,
                    "Chưa có bản dịch",
                    "Bảng lời thoại đang trống. Hãy bấm 'Bước 1: Phân tích & Dịch' hoặc 'CHẠY TỰ ĐỘNG TRỌN GÓI' trước.",
                )
                return

        settings = self._collect_settings()
        if not self._ensure_gemini_key_if_needed(settings, will_translate=(mode in ("analyze", "both"))):
            return
        if mode in ("render", "both") and not self._validate_tts_settings(settings):
            return
        out_path = self.ed_output.text().strip() or None
        self._set_running_ui(True)
        self._was_cancelled = False
        pmode = settings.process_mode
        if pmode == "ocr_sub_only":
            prog_mode = {"analyze": "analyze_ocr", "render": "render_ocr", "both": "both_ocr"}.get(mode, "both_ocr")
        elif pmode == "ocr_sub_dub":
            prog_mode = {"analyze": "analyze_ocr", "render": "render", "both": "both_ocr_dub"}.get(mode, "both_ocr_dub")
        else:
            prog_mode = mode
        self.prog_card.start(prog_mode)

        self.worker = PipelineWorker(mode, self.video_path, settings, segs, out_path)
        self.worker.log_sig.connect(self._log)
        self.worker.progress_sig.connect(self.prog_card.update_stage)
        self.worker.analyzed_sig.connect(self._on_analyzed)
        self.worker.finished_sig.connect(self._on_finished)
        self.worker.error_sig.connect(self._on_error)
        self.worker.start()

    def _cancel_worker(self):
        if self.worker is not None:
            self._was_cancelled = True
            self.worker.cancel()

    def _log(self, msg: str):
        self.txt_log.appendPlainText(msg)

    def _on_analyzed(self, segments: list, _src_lang: str):
        self._populate_table(segments)

    def _on_finished(self, out_path: str):
        self._set_running_ui(False)
        cancelled = getattr(self, "_was_cancelled", False) and not out_path
        self.prog_card.finish(cancelled=cancelled)
        if out_path:
            self.last_output = out_path
            self.btn_open_out.setEnabled(True)
            QMessageBox.information(self, "Xuất video hoàn tất", f"🎉 Video đã được xuất thành công tại:\n{out_path}")

    def _on_error(self, err: str):
        self._set_running_ui(False)
        if err.startswith("AUTH_REVOKED::"):
            reason = err.split("AUTH_REVOKED::", 1)[1]
            self._force_kick_to_login(reason)
            return
        self.prog_card.fail()
        QMessageBox.critical(self, "Lỗi xử lý video", err)

    def _open_output_folder(self):
        target = self.last_output or self.ed_output.text().strip()
        if target and os.path.exists(target):
            subprocess.Popen(["explorer", "/select,", str(Path(target).resolve())])
        elif self.video_path:
            subprocess.Popen(["explorer", str(Path(self.video_path).parent)])

    # ------------------------------------------------------------------ Batch Operations
    def _batch_add_paths(self, paths: list[str]):
        existing = {
            self.batch_table.item(r, 5).text()
            for r in range(self.batch_table.rowCount())
            if self.batch_table.item(r, 5)
        }
        for p in paths:
            p_str = str(Path(p).resolve())
            if p_str in existing or not os.path.exists(p_str):
                continue
            r = self.batch_table.rowCount()
            self.batch_table.insertRow(r)

            dur_str = "..."
            try:
                info = media.probe(p_str)
                dur_str = _fmt_mmss(info.get("duration", 0.0))
            except Exception:
                pass

            self.batch_table.setItem(r, 0, QTableWidgetItem(str(r + 1)))
            self.batch_table.setItem(r, 1, QTableWidgetItem(Path(p_str).name))
            self.batch_table.setItem(r, 2, QTableWidgetItem(dur_str))
            it_status = QTableWidgetItem("⏳ Chờ xử lý")
            it_status.setForeground(QColor("#6b7280"))
            self.batch_table.setItem(r, 3, it_status)

            prog = QProgressBar()
            prog.setRange(0, 100)
            prog.setValue(0)
            prog.setTextVisible(True)
            self.batch_table.setCellWidget(r, 4, prog)

            self.batch_table.setItem(r, 5, QTableWidgetItem(p_str))
            existing.add(p_str)

        self.lbl_batch_status.setText(f"📋 Danh sách sẵn sàng: {self.batch_table.rowCount()} video")

    def _batch_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Chọn danh sách video", "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm *.flv *.m4v *.ts);;Tất cả (*.*)"
        )
        if files:
            self.tabs.setCurrentWidget(self.tab_batch)
            self._batch_add_paths(files)

    def _batch_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa video")
        if folder:
            paths = []
            for root, _, files in os.walk(folder):
                for f in files:
                    if Path(f).suffix.lower() in VIDEO_EXTS:
                        paths.append(str(Path(root) / f))
            if paths:
                self._batch_add_paths(paths)

    def _batch_remove_selected(self):
        rows = sorted({idx.row() for idx in self.batch_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.batch_table.removeRow(r)
        for r in range(self.batch_table.rowCount()):
            self.batch_table.setItem(r, 0, QTableWidgetItem(str(r + 1)))
        self.lbl_batch_status.setText(f"📋 Danh sách sẵn sàng: {self.batch_table.rowCount()} video")

    def _batch_clear(self):
        self.batch_table.setRowCount(0)
        self.lbl_batch_status.setText("Danh sách trống")
        self.progress_batch.setValue(0)
        self.lbl_batch_pct.setText("0%")

    def _batch_browse_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu video kết quả")
        if d:
            self.ed_batch_out.setText(d)

    def _batch_open_output_dir(self):
        d = self.ed_batch_out.text().strip()
        if d and os.path.exists(d):
            subprocess.Popen(["explorer", str(Path(d).resolve())])
        elif self.batch_table.rowCount() > 0:
            first_path = self.batch_table.item(0, 5).text()
            subprocess.Popen(["explorer", str(Path(first_path).parent)])

    def _batch_start(self):
        total = self.batch_table.rowCount()
        if total == 0:
            QMessageBox.information(self, "Danh sách trống", "Vui lòng thêm ít nhất 1 video vào danh sách.")
            return

        items = []
        for r in range(total):
            p = self.batch_table.item(r, 5).text()
            items.append({"path": p, "row": r})
            it_status = self.batch_table.item(r, 3)
            if it_status:
                it_status.setText("⏳ Chờ xử lý")
                it_status.setForeground(QColor("#6b7280"))
            prog = self.batch_table.cellWidget(r, 4)
            if isinstance(prog, QProgressBar):
                prog.setValue(0)

        settings = self._collect_settings()
        if not self._ensure_gemini_key_if_needed(settings, will_translate=True):
            return
        if not self._validate_tts_settings(settings):
            return
        out_dir = self.ed_batch_out.text().strip()

        self.btn_batch_start.setEnabled(False)
        self.btn_batch_stop.setEnabled(True)
        self.progress_batch.setValue(0)
        self.lbl_batch_pct.setText("0%")
        self.txt_batch_log.clear()

        self.batch_worker = BatchWorker(items, settings, out_dir=out_dir)
        self.batch_worker.item_started.connect(self._on_batch_item_started)
        self.batch_worker.item_progress.connect(self._on_batch_item_progress)
        self.batch_worker.item_finished.connect(self._on_batch_item_finished)
        self.batch_worker.item_error.connect(self._on_batch_item_error)
        self.batch_worker.batch_finished.connect(self._on_batch_finished)
        self.batch_worker.auth_revoked.connect(self._force_kick_to_login)
        self.batch_worker.log_sig.connect(self.txt_batch_log.appendPlainText)
        self.batch_worker.start()

    def _batch_cancel(self):
        if self.batch_worker:
            self.batch_worker.cancel()
            self.lbl_batch_status.setText("⏹ Đang dừng danh sách...")

    def _on_batch_item_started(self, idx: int, fname: str):
        total = self.batch_table.rowCount()
        self.lbl_batch_status.setText(f"🔵 Đang xử lý video [{idx + 1}/{total}]: {fname}")
        it_status = self.batch_table.item(idx, 3)
        if it_status:
            it_status.setText("Khởi tạo...")
            it_status.setForeground(QColor("#2563eb"))

    def _on_batch_item_progress(self, idx: int, stage: str, mapped_frac: float):
        it_status = self.batch_table.item(idx, 3)
        if it_status:
            it_status.setText(stage)
        prog = self.batch_table.cellWidget(idx, 4)
        if isinstance(prog, QProgressBar):
            prog.setValue(int(np.clip(round(mapped_frac * 100), 0, 99)))

        total = max(1, self.batch_table.rowCount())
        overall_pct = ((idx + mapped_frac) / total) * 100.0
        self.progress_batch.setValue(int(np.clip(round(overall_pct * 10), 0, 995)))
        self.lbl_batch_pct.setText(f"{int(round(overall_pct))}%")

    def _on_batch_item_finished(self, idx: int, _out_path: str):
        it_status = self.batch_table.item(idx, 3)
        if it_status:
            it_status.setText("✅ Hoàn thành")
            it_status.setForeground(QColor("#16a34a"))
        prog = self.batch_table.cellWidget(idx, 4)
        if isinstance(prog, QProgressBar):
            prog.setValue(100)
        total = max(1, self.batch_table.rowCount())
        pct = int(round(((idx + 1) / total) * 100))
        self.progress_batch.setValue(pct * 10)
        self.lbl_batch_pct.setText(f"{pct}%")

    def _on_batch_item_error(self, idx: int, _err: str):
        it_status = self.batch_table.item(idx, 3)
        if it_status:
            it_status.setText("❌ Gặp lỗi")
            it_status.setForeground(QColor("#dc2626"))

    def _on_batch_finished(self, success_cnt: int, fail_cnt: int):
        self.btn_batch_start.setEnabled(True)
        self.btn_batch_stop.setEnabled(False)
        self.progress_batch.setValue(1000)
        self.lbl_batch_pct.setText("100%")
        self.lbl_batch_status.setText(f"✅ Hoàn tất danh sách: {success_cnt} thành công · {fail_cnt} lỗi")
        QMessageBox.information(
            self,
            "Hoàn tất danh sách",
            f"Đã xử lý xong toàn bộ danh sách video!\n\n"
            f"• Thành công: {success_cnt} video\n"
            f"• Lỗi: {fail_cnt} video",
        )

    # ------------------------------------------------------------------ Tray & Window Lifecycle
    def _setup_tray(self):
        try:
            from dubber.tray import TrayIcon
            self._tray = TrayIcon(self)
            self._tray.show()
        except Exception:
            self._tray = None

    def _show_about(self):
        QMessageBox.about(
            self,
            f"Về Video Dubber Studio v{APP_VERSION}",
            f"<h3>🎬 Video Dubber Studio v{APP_VERSION}</h3>"
            "<p>Giải pháp dịch thuật chuẩn văn phong, Clone giọng nhân vật bằng OmniVoice AI và thay thế phụ đề video chuyên nghiệp trên Windows.</p>"
            "<ul>"
            "<li><b>Nhận diện & Tách câu thông minh:</b> Faster-Whisper (CUDA) chống cắt ngang ý</li>"
            "<li><b>Dịch kịch bản chuẩn văn phong:</b> Google Gemini AI (tự xoay vòng 8 Model Cao ➜ Thấp) & dự phòng NLLB-200 Offline</li>"
            "<li><b>Giọng đọc siêu thực OmniVoice AI:</b> Tự động Clone giọng gốc từng nhân vật, Clone từ file mẫu & Thiết kế giọng đa ngôn ngữ + Edge Neural TTS dự phòng</li>"
            "<li><b>Chống dính chữ:</b> Cân bằng nhịp Elastic Timeline + Khoảng nghỉ tự nhiên</li>"
            "<li><b>Quét chữ phụ đề trên video không tiếng (OCR + Vision AI):</b> Trích xuất chữ sub hiển thị ➜ Dịch ➜ Xóa sạch sub cũ ➜ Ghép sub mới (tùy chọn giữ nguyên âm thanh hoặc lồng tiếng)</li>"
            "<li><b>Khung quét phụ đề linh hoạt:</b> Kéo thả di chuyển và co giãn trực tiếp khung xanh trên màn hình xem trước vào đúng vị trí phụ đề</li>"
            "<li><b>Xóa sạch phụ đề cũ:</b> RapidOCR + Telea Stroke Inpainting & Cố định vị trí sub mới</li>"
            "<li><b>Bảo mật & Tự cập nhật:</b> Khóa thiết bị Firebase thời gian thực + GitHub Auto-Update</li>"
            "</ul>",
        )

    def closeEvent(self, ev):
        if not self._signing_out:
            if (self.worker and self.worker.isRunning()) or (self.batch_worker and self.batch_worker.isRunning()):
                ret = QMessageBox.question(
                    self,
                    "Đang xử lý video",
                    "Tiến trình đang chạy. Bạn có chắc chắn muốn hủy và đóng ứng dụng không?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if ret != QMessageBox.Yes:
                    ev.ignore()
                    return
                if self.worker:
                    self.worker.cancel()
                    self.worker.wait(1500)
                if self.batch_worker:
                    self.batch_worker.cancel()
                    self.batch_worker.wait(1500)

        self._auth_timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._tray is not None:
            self._tray.dispose()
            self._tray = None
        super().closeEvent(ev)


# =============================================================================
# SESSION CONTROLLER (GUARANTEES STRICT LOGIN GATE & CLEAN LOGOUT RE-ENTRY)
# =============================================================================
class AppSessionController:
    def __init__(self, app: QApplication):
        self.app = app
        self.win: MainWindow | None = None

    def start(self):
        # 1. Verify saved session ONLINE against Firebase Auth (`disabled` check) & Firestore
        auto_ok, email_or_err, _uid = auth.auto_login()
        if auto_ok:
            self._open_main_window()
        else:
            # Show error reason if account was disabled/revoked while app was closed
            self._prompt_login(initial_message=email_or_err if email_or_err else "")

    def _prompt_login(self, initial_message: str = ""):
        if self.win is not None:
            self.win.close()
            self.win.deleteLater()
            self.win = None

        dlg = LoginDialog(initial_message=initial_message)
        if dlg.exec() == QDialog.Accepted and auth.get_current_user() is not None:
            self._open_main_window()
        else:
            # User cancelled login or failed verification -> terminate process completely!
            auth.logout()
            self.app.quit()
            sys.exit(0)

    def _open_main_window(self):
        self.win = MainWindow()
        self.win.logout_requested.connect(self._on_logout_requested)
        self.win.show()
        self.win.raise_()
        self.win.activateWindow()

    def _on_logout_requested(self, reason_msg: str):
        # Defer to next event loop tick so MainWindow finishes closing cleanly
        QTimer.singleShot(50, lambda m=reason_msg: self._prompt_login(initial_message=m))


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    app.setWindowIcon(_app_icon())

    controller = AppSessionController(app)
    controller.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
