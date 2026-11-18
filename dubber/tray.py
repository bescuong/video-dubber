"""System tray icon với kiểm tra phiên đăng nhập an toàn."""
import os
import subprocess
from pathlib import Path

from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from . import auth


def _get_icon():
    here = Path(__file__).parent.parent
    p = here / "icon.ico"
    if p.exists():
        return QIcon(str(p))
    return QIcon()


class TrayIcon(QSystemTrayIcon):
    def __init__(self, main_window, parent=None):
        super().__init__(_get_icon(), parent)
        self.main_window = main_window
        self._disposed = False
        self.setToolTip("Video Dubber Studio — Lồng tiếng AI")
        self._build_menu()
        self.activated.connect(self._on_activated)

    def _build_menu(self):
        menu = QMenu()

        act_show = QAction("🪟 Mở cửa sổ chính", menu)
        act_show.triggered.connect(self._show_main)
        menu.addAction(act_show)

        menu.addSeparator()

        act_open = QAction("📁 Mở thư mục làm việc", menu)
        act_open.triggered.connect(self._open_folder)
        menu.addAction(act_open)

        menu.addSeparator()

        act_quit = QAction("❌ Thoát hoàn toàn", menu)
        act_quit.triggered.connect(self._quit)
        menu.addAction(act_quit)

        self.setContextMenu(menu)

    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_main()

    def _show_main(self):
        if self._disposed or self.main_window is None:
            return
        # Never reopen main window if logged out
        if auth.get_current_user() is None:
            return
        self.main_window.showNormal()
        self.main_window.raise_()
        self.main_window.activateWindow()

    def _open_folder(self):
        if self._disposed or self.main_window is None:
            return
        video = self.main_window.video_path
        if video and os.path.exists(video):
            subprocess.Popen(["explorer", str(Path(video).parent)])
        else:
            subprocess.Popen(["explorer", str(Path.home() / "Videos")])

    def dispose(self):
        self._disposed = True
        self.hide()
        self.setContextMenu(None)
        self.main_window = None
        self.deleteLater()

    def _quit(self):
        self.dispose()
        QApplication.quit()
