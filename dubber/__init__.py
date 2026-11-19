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
    py_dir = os.path.dirname(sys.executable)
    exe_dir = os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv else sys.executable))

    dirs = [
        py_dir,
        os.path.join(py_dir, "bin"),
        os.path.join(py_dir, "_internal"),
        os.path.join(py_dir, "_internal", "bin"),
        os.path.join(py_dir, "_internal", "torch", "lib"),
        os.path.join(py_dir, "_internal", "ctranslate2"),
        exe_dir,
        os.path.join(exe_dir, "bin"),
        os.path.join(exe_dir, "_internal", "torch", "lib"),
        os.path.join(exe_dir, "_internal", "ctranslate2"),
    ]
    if hasattr(sys, "_MEIPASS"):
        dirs.extend([
            sys._MEIPASS,
            os.path.join(sys._MEIPASS, "bin"),
            os.path.join(sys._MEIPASS, "torch", "lib"),
            os.path.join(sys._MEIPASS, "ctranslate2"),
        ])

    # Virtualenv / site-packages locations
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_sp = os.path.join(repo_root, ".venv", "Lib", "site-packages")
    dirs.extend([
        os.path.join(repo_root, ".venv", "Scripts"),
        os.path.join(venv_sp, "torch", "lib"),
        os.path.join(venv_sp, "ctranslate2"),
    ])

    if hasattr(site, "getsitepackages"):
        try:
            for sp in site.getsitepackages():
                dirs.append(os.path.join(sp, "torch", "lib"))
                dirs.append(os.path.join(sp, "ctranslate2"))
                dirs.append(os.path.join(sp, "PySide6"))
                dirs.append(os.path.join(sp, "shiboken6"))
        except Exception:
            pass

    seen = set()
    cur_path_parts = set(os.environ.get("PATH", "").split(os.pathsep))
    prepend_parts = []
    for d in dirs:
        if not d:
            continue
        nd = os.path.normcase(os.path.abspath(d))
        if nd in seen or not os.path.isdir(d):
            continue
        seen.add(nd)
        try:
            _DLL_HANDLES.append(os.add_dll_directory(d))
        except (OSError, AttributeError):
            pass
        if d not in cur_path_parts:
            prepend_parts.append(d)

    if prepend_parts:
        os.environ["PATH"] = os.pathsep.join(prepend_parts) + os.pathsep + os.environ.get("PATH", "")


_setup_windows_dlls()
