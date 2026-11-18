"""Modern Login & Machine Activation Dialog for Video Dubber."""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from dubber import auth


class _LoginWorker(QThread):
    done = Signal(bool, str)

    def __init__(self, email: str, password: str):
        super().__init__()
        self.email = email
        self.password = password

    def run(self):
        try:
            ok, msg = auth.login(self.email, self.password)
            self.done.emit(ok, msg)
        except Exception as e:
            self.done.emit(False, str(e))


class FirebaseConfigDialog(QDialog):
    """Admin-only dialog to configure Firebase API Key and Project ID."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cấu hình kết nối Firebase (Admin)")
        self.setFixedSize(480, 260)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        info = QLabel(
            "Nhập thông tin dự án Firebase (Firebase Console > Project settings > Web API Key & Project ID):"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #4b5563; font-size: 12px;")
        layout.addWidget(info)

        form = QFormLayout()
        form.setSpacing(10)

        cfg = auth.get_firebase_config() or {}
        self.ed_key = QLineEdit(cfg.get("apiKey") or cfg.get("api_key", ""))
        self.ed_key.setPlaceholderText("AIzaSy...")

        self.ed_pid = QLineEdit(cfg.get("projectId") or cfg.get("project_id", ""))
        self.ed_pid.setPlaceholderText("my-video-dubber")

        form.addRow("Web API Key:", self.ed_key)
        form.addRow("Project ID:", self.ed_pid)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_browse = QPushButton("📁 Chọn file .json...")
        btn_browse.clicked.connect(self._browse_json)

        btn_save = QPushButton("💾 Lưu cấu hình")
        btn_save.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 6px 14px;")
        btn_save.clicked.connect(self._save)

        btn_cancel = QPushButton("Hủy")
        btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(btn_browse)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

    def _browse_json(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn file cấu hình Firebase JSON", "", "JSON (*.json)")
        if p:
            try:
                import json
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                k = data.get("apiKey") or data.get("api_key") or data.get("private_key_id", "")
                pid = data.get("projectId") or data.get("project_id", "")
                if k:
                    self.ed_key.setText(k)
                if pid:
                    self.ed_pid.setText(pid)
            except Exception as e:
                QMessageBox.warning(self, "Lỗi đọc file", str(e))

    def _save(self):
        k = self.ed_key.text().strip()
        pid = self.ed_pid.text().strip()
        if not k:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập Web API Key.")
            return
        auth.save_firebase_config(k, pid)
        QMessageBox.information(self, "Thành công", "Đã lưu cấu hình Firebase thành công!")
        self.accept()


class LoginDialog(QDialog):
    """Modal login dialog shown before MainWindow and after Logout / Session Revocation."""

    def __init__(self, parent=None, initial_message: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Video Dubber — Đăng nhập bản quyền")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedSize(460, 440)
        self._worker: _LoginWorker | None = None
        self._build_ui()
        self._apply_theme()
        if initial_message:
            self.lbl_status.setText(initial_message)
        else:
            self._check_config_status()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 20)
        layout.setSpacing(10)

        title = QLabel("🎬 Video Dubber Studio")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet("color: #111827;")
        layout.addWidget(title)

        sub = QLabel("Đăng nhập tài khoản để kích hoạt sử dụng trên thiết bị này")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #6b7280; font-size: 12px;")
        layout.addWidget(sub)

        # Machine ID card
        mid_card = QFrame()
        mid_card.setStyleSheet("background-color: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 8px; padding: 4px;")
        mid_layout = QHBoxLayout(mid_card)
        mid_layout.setContentsMargins(12, 6, 12, 6)

        lbl_mid_title = QLabel("Mã thiết bị:")
        lbl_mid_title.setStyleSheet("font-size: 11px; color: #4b5563; font-weight: bold; border: none;")
        self.lbl_mid = QLabel(auth.get_short_machine_id())
        self.lbl_mid.setStyleSheet("font-size: 12px; font-weight: bold; color: #1d4ed8; font-family: Consolas; border: none;")

        btn_copy = QPushButton("Sao chép")
        btn_copy.setCursor(Qt.PointingHandCursor)
        btn_copy.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 3px 10px; background: white; border: 1px solid #d1d5db; border-radius: 4px; color: #374151; }"
            "QPushButton:hover { background: #e5e7eb; }"
        )
        btn_copy.clicked.connect(self._copy_machine_id)

        mid_layout.addWidget(lbl_mid_title)
        mid_layout.addWidget(self.lbl_mid)
        mid_layout.addStretch()
        mid_layout.addWidget(btn_copy)
        layout.addWidget(mid_card)

        # Credentials form
        form = QFormLayout()
        form.setSpacing(10)

        self.ed_email = QLineEdit()
        self.ed_email.setPlaceholderText("nhap-email@gmail.com")
        self.ed_email.setMinimumHeight(36)

        self.ed_pass = QLineEdit()
        self.ed_pass.setEchoMode(QLineEdit.Password)
        self.ed_pass.setPlaceholderText("••••••••")
        self.ed_pass.setMinimumHeight(36)
        self.ed_pass.returnPressed.connect(self._do_login)

        form.addRow("Email:", self.ed_email)
        form.addRow("Mật khẩu:", self.ed_pass)
        layout.addLayout(form)

        opts_row = QHBoxLayout()
        self.chk_show = QCheckBox("Hiện mật khẩu")
        self.chk_show.toggled.connect(
            lambda v: self.ed_pass.setEchoMode(QLineEdit.Normal if v else QLineEdit.Password)
        )
        opts_row.addWidget(self.chk_show)
        opts_row.addStretch()
        layout.addLayout(opts_row)

        layout.addSpacing(4)

        self.btn_login = QPushButton("🔐 Đăng nhập & Xác thực tài khoản")
        self.btn_login.setMinimumHeight(40)
        self.btn_login.setCursor(Qt.PointingHandCursor)
        self.btn_login.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: white; font-weight: bold; border-radius: 6px; font-size: 13px; }"
            "QPushButton:hover { background-color: #1d4ed8; }"
            "QPushButton:disabled { background-color: #93c5fd; }"
        )
        self.btn_login.clicked.connect(self._do_login)
        layout.addWidget(self.btn_login)

        self.btn_offline = QPushButton("🔒 Mở khóa Quản trị viên (Admin PIN)")
        self.btn_offline.setCursor(Qt.PointingHandCursor)
        self.btn_offline.setStyleSheet(
            "QPushButton { background-color: #f9fafb; color: #4b5563; font-weight: 500; border-radius: 6px; padding: 6px; font-size: 11px; border: 1px dashed #d1d5db; }"
            "QPushButton:hover { background-color: #f3f4f6; }"
        )
        self.btn_offline.clicked.connect(self._do_admin_unlock)
        layout.addWidget(self.btn_offline)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #dc2626; font-size: 12px; font-weight: 500;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

    def _apply_theme(self):
        self.setStyleSheet(
            "QDialog { background-color: #ffffff; }"
            "QLabel { font-family: 'Segoe UI'; }"
            "QLineEdit { border: 1px solid #d1d5db; border-radius: 6px; padding: 4px 10px; font-size: 13px; background: #ffffff; color: #111827; }"
            "QLineEdit:focus { border: 1.5px solid #2563eb; }"
        )

    def _check_config_status(self):
        if not auth.is_firebase_configured():
            self.lbl_status.setText("ℹ Chưa cấu hình Firebase. Bấm '⚙ Cấu hình hệ thống' để cài đặt.")

    def _copy_machine_id(self):
        QApplication.clipboard().setText(auth.get_short_machine_id())
        self.lbl_status.setStyleSheet("color: #16a34a; font-size: 12px; font-weight: 500;")
        self.lbl_status.setText("✅ Đã sao chép Mã thiết bị vào bộ nhớ tạm.")

    def _open_config(self):
        pin, ok = QInputDialog.getText(
            self,
            "Xác thực Quản trị viên (Admin)",
            "Nhập mã PIN Quản trị viên để thay đổi cấu hình Firebase:",
            QLineEdit.Password,
        )
        if not ok or not pin:
            return
        if not auth.verify_admin_pin(pin):
            QMessageBox.critical(
                self,
                "Từ chối quyền truy cập",
                "⛔ Mã PIN Quản trị viên không chính xác!\nChỉ Admin mới có quyền thay đổi thông số hệ thống.",
            )
            return
        dlg = FirebaseConfigDialog(self)
        if dlg.exec() == QDialog.Accepted:
            self.lbl_status.setStyleSheet("color: #16a34a; font-size: 12px;")
            self.lbl_status.setText("✅ Đã cập nhật cấu hình Firebase.")

    def _do_admin_unlock(self):
        pin, ok = QInputDialog.getText(
            self,
            "Mở khóa Quản trị viên",
            "Nhập mã PIN Quản trị viên (phiên tạm thời, tự khóa khi đăng xuất/đóng app):",
            QLineEdit.Password,
        )
        if ok and auth.verify_admin_pin(pin):
            auth.activate_admin_session()
            self.accept()
        elif ok:
            QMessageBox.warning(self, "Từ chối", "⛔ Mã PIN Quản trị viên không chính xác!")

    def _do_login(self):
        email = self.ed_email.text().strip()
        password = self.ed_pass.text()
        if not email or not password:
            self.lbl_status.setStyleSheet("color: #dc2626; font-size: 12px; font-weight: 500;")
            self.lbl_status.setText("Vui lòng nhập đầy đủ Email và Mật khẩu.")
            return

        self.btn_login.setEnabled(False)
        self.btn_login.setText("Đang kiểm tra tài khoản trực tuyến...")
        self.lbl_status.setText("")

        self._worker = _LoginWorker(email, password)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, success: bool, msg: str):
        self.btn_login.setEnabled(True)
        self.btn_login.setText("🔐 Đăng nhập & Xác thực tài khoản")
        if success:
            self.accept()
        else:
            self.lbl_status.setStyleSheet("color: #dc2626; font-size: 12px; font-weight: 500;")
            self.lbl_status.setText(msg)
            if "MÁY KHÁC" in msg or "vô hiệu hóa" in msg or "khóa" in msg:
                QMessageBox.critical(self, "Tài khoản bị chặn", msg)
