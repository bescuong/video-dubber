"""Offline video dubbing toolkit."""
import os
import site
import sys

_DLL_HANDLES = []

# Suppress noisy HuggingFace symlink warning on Windows (safe to ignore)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Ensure GUI apps without console don't crash on print / sys.stdout writes
class _SafeStream:
    def write(self, s): pass
    def flush(self): pass
    def isatty(self): return False

if sys.stdout is None:
    sys.stdout = _SafeStream()
if sys.stderr is None:
    sys.stderr = _SafeStream()


def _setup_windows_dlls():
    if sys.platform != "win32":
        return
    dirs = [
        os.path.dirname(sys.executable),
        os.path.join(os.path.dirname(sys.executable), "bin"),
    ]
    if hasattr(sys, "_MEIPASS"):
        dirs.append(sys._MEIPASS)
        dirs.append(os.path.join(sys._MEIPASS, "bin"))

    # Also check local bin directory next to script/executable
    exe_dir = os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv else sys.executable))
    dirs.append(exe_dir)
    dirs.append(os.path.join(exe_dir, "bin"))

    if hasattr(site, "getsitepackages"):
        try:
            for sp in site.getsitepackages():
                dirs.append(os.path.join(sp, "torch", "lib"))
                dirs.append(os.path.join(sp, "PySide6"))
                dirs.append(os.path.join(sp, "shiboken6"))
        except Exception:
            pass

    for d in dirs:
        if d and os.path.isdir(d):
            try:
                _DLL_HANDLES.append(os.add_dll_directory(d))
            except (OSError, AttributeError):
                pass
            if d not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


_setup_windows_dlls()
