"""Splash screen và cửa sổ chờ trong khi import các thư viện nặng."""
import os
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap, QLinearGradient, QGradient
from PySide6.QtWidgets import QApplication, QLabel, QSplashScreen


def _make_pixmap(w=520, h=280):
    px = QPixmap(w, h)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)

    # Gradient background
    grad = QLinearGradient(0, 0, w, h)
    grad.setColorAt(0.0, QColor(15, 23, 42))
    grad.setColorAt(1.0, QColor(30, 58, 138))
    p.setBrush(grad)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(0, 0, w, h, 18, 18)

    # Title
    p.setPen(QColor(255, 255, 255))
    f = QFont("Segoe UI", 28, QFont.Bold)
    p.setFont(f)
    p.drawText(0, 30, w, 60, Qt.AlignHCenter | Qt.AlignVCenter, "🎬  Video Dubber")

    # Subtitle
    p.setPen(QColor(148, 163, 184))
    f2 = QFont("Segoe UI", 12)
    p.setFont(f2)
    p.drawText(0, 90, w, 36, Qt.AlignHCenter | Qt.AlignVCenter,
               "Lồng tiếng & Thay phụ đề tự động  •  Offline AI")

    # Divider
    p.setPen(QColor(59, 130, 246, 160))
    p.drawLine(60, 142, w - 60, 142)

    # Features
    p.setPen(QColor(203, 213, 225))
    f3 = QFont("Segoe UI", 10)
    p.setFont(f3)
    feats = ["🎙 Whisper ASR", "🌐 NLLB-200 Dịch thuật", "🔊 Edge TTS", "🎵 Demucs", "🔍 OCR"]
    col_w = (w - 60) // len(feats)
    for i, feat in enumerate(feats):
        p.drawText(30 + i * col_w, 158, col_w, 28, Qt.AlignHCenter | Qt.AlignVCenter, feat)

    # Loading text
    p.setPen(QColor(100, 149, 237))
    f4 = QFont("Segoe UI", 10, QFont.Bold)
    p.setFont(f4)
    p.drawText(0, 210, w, 28, Qt.AlignHCenter | Qt.AlignVCenter, "Đang khởi động...")

    # Version
    p.setPen(QColor(71, 85, 105))
    f5 = QFont("Segoe UI", 9)
    p.setFont(f5)
    p.drawText(0, h - 32, w, 24, Qt.AlignHCenter | Qt.AlignVCenter, "v1.0  •  GTX 1060 CUDA 12.6")

    p.end()
    return px


class SplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__(_make_pixmap())
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self._dots = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(400)

    def _tick(self):
        self._dots = (self._dots + 1) % 4
        dots = "." * self._dots + "   "[self._dots:]
        self.showMessage(f"Đang tải thư viện AI{dots}",
                         Qt.AlignBottom | Qt.AlignHCenter,
                         QColor(99, 179, 237))

    def finish_and_show(self, window):
        self._timer.stop()
        self.finish(window)
