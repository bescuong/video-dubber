"""GitHub Auto-Updater supporting both Instant Hot-Patches (.zip) and Full Installers (.exe + .bin)."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

# Locked official GitHub repository for update checks
HARDCODED_GITHUB_REPO = "bescuong/video-dubber"

_APP_DATA = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "VideoDubber"
HOTPATCH_DIR = _APP_DATA / "hotpatch"


# =============================================================================
# VERSION & HOTPATCH HELPERS
# =============================================================================
def parse_version(v_str: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", str(v_str or "0"))
    return tuple(int(x) for x in nums[:4]) if nums else (0, 0, 0)


def is_newer_version(latest: str, current: str) -> bool:
    return parse_version(latest) > parse_version(current)


def get_update_repo() -> str:
    return HARDCODED_GITHUB_REPO


# =============================================================================
# GITHUB RELEASE CHECKER
# =============================================================================
def check_for_updates(current_version: str) -> dict:
    """Query GitHub Releases (and version.json fallback) for updates.

    Prefers lightweight HotPatch zip (`VideoDubber_HotPatch_*.zip`) for instant 2-second
    in-place updates unless release notes/metadata specify `[FULL_SETUP]`, otherwise falls
    back to full Setup `.exe` (+ any `.bin` slices).
    """
    repo = get_update_repo()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "VideoDubber-Updater",
    }

    rel_url = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        req = urllib.request.Request(rel_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tag = data.get("tag_name", "").lstrip("vV") or current_version
            notes = data.get("body", "") or "Cải thiện chất lượng lồng tiếng, phụ đề và hiệu năng."
            html_url = data.get("html_url", f"https://github.com/{repo}/releases")
            assets = data.get("assets", [])

            hotpatch_asset = None
            exe_asset = None
            companion_bins = []
            force_full = "[FULL_SETUP]" in notes

            for a in assets:
                nm = a.get("name", "")
                low = nm.lower()
                url = a.get("browser_download_url", "")
                if not url:
                    continue
                if "hotpatch" in low and low.endswith(".zip"):
                    hotpatch_asset = {"name": nm, "url": url, "size": a.get("size", 0)}
                elif low.endswith(".exe"):
                    exe_asset = {"name": nm, "url": url, "size": a.get("size", 0)}
                elif low.endswith(".bin"):
                    companion_bins.append({"name": nm, "url": url, "size": a.get("size", 0)})

            # Prefer fast HotPatch if available and not forced full installer
            if hotpatch_asset and not force_full:
                dl_files = [hotpatch_asset]
                mode = "hotpatch"
            elif exe_asset:
                dl_files = [exe_asset] + companion_bins
                mode = "installer"
            else:
                dl_files = []
                mode = "browser"

            return {
                "has_update": is_newer_version(tag, current_version),
                "latest_version": tag,
                "notes": notes.replace("[FULL_SETUP]", "").strip(),
                "mode": mode,
                "files": dl_files,
                "html_url": html_url,
                "error": "",
            }
    except urllib.error.HTTPError as e:
        if e.code != 404:
            return {
                "has_update": False,
                "latest_version": current_version,
                "notes": "",
                "mode": "browser",
                "files": [],
                "html_url": f"https://github.com/{repo}/releases",
                "error": f"Không thể kiểm tra GitHub ({e.code}): {e.reason}",
            }
    except Exception as e:
        return {
            "has_update": False,
            "latest_version": current_version,
            "notes": "",
            "mode": "browser",
            "files": [],
            "html_url": f"https://github.com/{repo}/releases",
            "error": f"Lỗi kết nối khi kiểm tra cập nhật: {e}",
        }

    # Fallback: raw version.json on main branch
    v_url = f"https://raw.githubusercontent.com/{repo}/main/version.json"
    try:
        req = urllib.request.Request(v_url, headers={"User-Agent": "VideoDubber-Updater"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            vdata = json.loads(resp.read().decode("utf-8"))
            ver = str(vdata.get("version", current_version)).lstrip("vV")
            dl = vdata.get("hotpatch_url") or vdata.get("download_url", "")
            files = []
            mode = "browser"
            if dl:
                nm = Path(urllib.parse.urlparse(dl).path).name or "update.zip"
                files = [{"name": nm, "url": dl, "size": 0}]
                mode = "hotpatch" if nm.lower().endswith(".zip") else "installer"
            return {
                "has_update": is_newer_version(ver, current_version),
                "latest_version": ver,
                "notes": vdata.get("notes", ""),
                "mode": mode,
                "files": files,
                "html_url": f"https://github.com/{repo}/releases",
                "error": "",
            }
    except Exception:
        pass

    return {
        "has_update": False,
        "latest_version": current_version,
        "notes": "Bạn đang dùng phiên bản mới nhất.",
        "mode": "browser",
        "files": [],
        "html_url": f"https://github.com/{repo}/releases",
        "error": "",
    }


class UpdateCheckWorker(QThread):
    finished_sig = Signal(dict)

    def __init__(self, current_version: str):
        super().__init__()
        self.current_version = current_version

    def run(self):
        res = check_for_updates(self.current_version)
        self.finished_sig.emit(res)


class _DownloadAndApplyWorker(QThread):
    progress_sig = Signal(float, str)  # (0..100, status_text)
    done_sig = Signal(bool, str, str)  # (ok, mode, path_or_err)

    def __init__(self, mode: str, files: list[dict], target_version: str):
        super().__init__()
        self.mode = mode
        self.files = files
        self.target_version = target_version
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            stage_dir = Path(tempfile.gettempdir()) / f"VideoDubber_Update_{self.target_version}"
            stage_dir.mkdir(parents=True, exist_ok=True)

            total_expected = sum(int(f.get("size", 0) or 0) for f in self.files)
            downloaded_all = 0
            t0 = time.time()
            saved_paths = []

            for f_info in self.files:
                if self._cancel:
                    self.done_sig.emit(False, self.mode, "Đã hủy tải bản cập nhật.")
                    return
                dest = stage_dir / f_info["name"]
                req = urllib.request.Request(f_info["url"], headers={"User-Agent": "VideoDubber-Updater"})
                with urllib.request.urlopen(req, timeout=45) as resp:
                    clen = int(resp.headers.get("Content-Length", "0") or 0)
                    if total_expected == 0:
                        total_expected = clen
                    chunk_sz = 256 * 1024
                    with open(dest, "wb") as out_f:
                        while True:
                            if self._cancel:
                                self.done_sig.emit(False, self.mode, "Đã hủy tải bản cập nhật.")
                                return
                            buf = resp.read(chunk_sz)
                            if not buf:
                                break
                            out_f.write(buf)
                            downloaded_all += len(buf)
                            elapsed = max(0.1, time.time() - t0)
                            speed_mb = (downloaded_all / (1024 * 1024)) / elapsed
                            if total_expected > 0:
                                pct = min(98.0, downloaded_all * 100.0 / total_expected)
                                txt = (
                                    f"{downloaded_all / (1024*1024):.2f} / "
                                    f"{total_expected / (1024*1024):.2f} MB ({speed_mb:.1f} MB/s)"
                                )
                            else:
                                pct = 60.0
                                txt = f"Đã tải {downloaded_all / 1024:.0f} KB ({speed_mb:.1f} MB/s)"
                            self.progress_sig.emit(pct, txt)
                saved_paths.append(dest)

            if self.mode == "hotpatch" and saved_paths:
                self.progress_sig.emit(99.0, "Đang áp dụng bản vá nhanh...")
                tmp_extract = _APP_DATA / "hotpatch_staging"
                if tmp_extract.exists():
                    shutil.rmtree(tmp_extract, ignore_errors=True)
                tmp_extract.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(saved_paths[0], "r") as zf:
                    zf.extractall(tmp_extract)
                (tmp_extract / "patch_manifest.json").write_text(
                    json.dumps({"version": self.target_version, "applied_at": time.time()}, indent=2),
                    encoding="utf-8",
                )
                if HOTPATCH_DIR.exists():
                    shutil.rmtree(HOTPATCH_DIR, ignore_errors=True)
                tmp_extract.rename(HOTPATCH_DIR)
                self.progress_sig.emit(100.0, "Hoàn tất!")
                self.done_sig.emit(True, "hotpatch", str(HOTPATCH_DIR))
                return

            exe_path = next((str(p) for p in saved_paths if p.suffix.lower() == ".exe"), str(saved_paths[0]))
            self.progress_sig.emit(100.0, "Hoàn tất!")
            self.done_sig.emit(True, "installer", exe_path)
        except Exception as e:
            self.done_sig.emit(False, self.mode, f"Lỗi khi tải bản cập nhật: {e}")


class UpdateDialog(QDialog):
    """User-facing dialog to view release notes and auto-download/install updates."""

    def __init__(self, current_version: str, update_info: dict, parent=None):
        super().__init__(parent)
        self.current_version = current_version
        self.info = update_info
        self._dl_worker: _DownloadAndApplyWorker | None = None
        self.setWindowTitle("Cập nhật phần mềm Video Dubber")
        self.setFixedSize(520, 380)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        latest = self.info.get("latest_version", self.current_version)
        title = QLabel(f"🚀 Đã có phiên bản mới v{latest}!")
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title.setStyleSheet("color: #1d4ed8;")
        layout.addWidget(title)

        sub = QLabel(f"Phiên bản hiện tại: v{self.current_version}  ➔  Phiên bản mới: v{latest}")
        sub.setStyleSheet("color: #4b5563; font-size: 12px;")
        layout.addWidget(sub)

        notes_lbl = QLabel("Nội dung cập nhật:")
        notes_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(notes_lbl)

        self.txt_notes = QPlainTextEdit()
        self.txt_notes.setReadOnly(True)
        self.txt_notes.setPlainText(self.info.get("notes") or "Cải thiện chất lượng lồng tiếng, phụ đề và hiệu năng.")
        layout.addWidget(self.txt_notes, 1)

        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        self.prog.setVisible(False)
        layout.addWidget(self.prog)

        self.lbl_dl = QLabel("")
        self.lbl_dl.setStyleSheet("color: #2563eb; font-size: 12px;")
        layout.addWidget(self.lbl_dl)

        btn_row = QHBoxLayout()
        self.btn_later = QPushButton("Để sau")
        self.btn_later.clicked.connect(self._on_cancel)

        self.btn_update = QPushButton("⬇ Tải về & Tự động cập nhật ngay")
        self.btn_update.setCursor(Qt.PointingHandCursor)
        self.btn_update.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: white; font-weight: bold; padding: 8px 18px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #15803d; }"
        )
        self.btn_update.clicked.connect(self._start_update)

        btn_row.addStretch()
        btn_row.addWidget(self.btn_later)
        btn_row.addWidget(self.btn_update)
        layout.addLayout(btn_row)

    def _on_cancel(self):
        if self._dl_worker and self._dl_worker.isRunning():
            self._dl_worker.cancel()
            self._dl_worker.wait(1500)
        self.reject()

    def _start_update(self):
        files = self.info.get("files", [])
        mode = self.info.get("mode", "browser")
        html_url = self.info.get("html_url", "")
        if not files or mode == "browser":
            if html_url:
                import webbrowser
                webbrowser.open(html_url)
            self.accept()
            return

        self.btn_update.setEnabled(False)
        self.prog.setVisible(True)
        self.lbl_dl.setText("Đang kết nối tải bản cập nhật...")
        self._dl_worker = _DownloadAndApplyWorker(
            mode=mode,
            files=files,
            target_version=self.info.get("latest_version", "latest"),
        )
        self._dl_worker.progress_sig.connect(self._on_prog)
        self._dl_worker.done_sig.connect(self._on_done)
        self._dl_worker.start()

    def _on_prog(self, pct: float, txt: str):
        self.prog.setValue(int(pct))
        self.lbl_dl.setText(txt)

    def _on_done(self, ok: bool, mode: str, path_or_err: str):
        if not ok:
            self.btn_update.setEnabled(True)
            self.lbl_dl.setText(path_or_err)
            QMessageBox.warning(self, "Không thể cập nhật", path_or_err)
            return

        if mode == "hotpatch":
            QMessageBox.information(
                self,
                "Cập nhật hoàn tất!",
                "✅ Đã cập nhật lên phiên bản mới thành công!\nỨng dụng sẽ tự động khởi động lại ngay bây giờ.",
            )
            # Relaunch current executable cleanly with fresh environment
            env = os.environ.copy()
            env.pop("VD_HOTPATCH_BOOTED", None)
            if getattr(sys, "frozen", False):
                subprocess.Popen([sys.executable], env=env)
            else:
                subprocess.Popen([sys.executable, str(Path(__file__).parent.parent / "app.py")], env=env)
            QApplication.quit()
            sys.exit(0)

        p = Path(path_or_err)
        if p.suffix.lower() == ".exe":
            QMessageBox.information(
                self,
                "Tải hoàn tất",
                "Đã tải xong bộ cài nâng cấp mới!\nỨng dụng sẽ đóng và tự động chạy trình cài đặt.",
            )
            subprocess.Popen([str(p)])
            QApplication.quit()
            sys.exit(0)
        else:
            subprocess.Popen(["explorer", "/select,", str(p)])
            self.accept()
