# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
    copy_metadata
)

block_cipher = None
base_dir = os.path.abspath(SPECPATH)
venv_scripts = os.path.join(base_dir, '.venv', 'Scripts')

binaries = []

# Collect dynamic libs
binaries += collect_dynamic_libs('torch')
binaries += collect_dynamic_libs('torchaudio')
binaries += collect_dynamic_libs('ctranslate2')
binaries += collect_dynamic_libs('onnxruntime')

# Bundle all VC++ runtime DLLs from .venv/Scripts to application root '.'
if os.path.isdir(venv_scripts):
    for f in os.listdir(venv_scripts):
        if f.lower().endswith('.dll'):
            binaries.append((os.path.join(venv_scripts, f), '.'))

# Bundle ffmpeg & ffprobe from bin/
bin_dir = os.path.join(base_dir, 'bin')
if os.path.isdir(bin_dir):
    for f in os.listdir(bin_dir):
        if f.lower().endswith('.exe'):
            binaries.append((os.path.join(bin_dir, f), 'bin'))

datas = [
    (os.path.join(base_dir, 'icon.ico'), '.'),
]

models_dir = os.path.join(base_dir, 'models')
if os.path.isdir(models_dir):
    datas.append((models_dir, 'models'))

# Package data files
datas += collect_data_files('rapidocr_onnxruntime')
datas += collect_data_files('demucs')
datas += collect_data_files('transformers')
datas += collect_data_files('edge_tts')
datas += collect_data_files('faster_whisper')

# Explicitly bundle torch._dynamo.polyfills to prevent missing dynamic polyfill modules (copy.py, heapq.py, etc.)
torch_polyfills = os.path.join(base_dir, '.venv', 'Lib', 'site-packages', 'torch', '_dynamo', 'polyfills')
if os.path.isdir(torch_polyfills):
    datas.append((torch_polyfills, os.path.join('torch', '_dynamo', 'polyfills')))

# Package metadata required by huggingface_hub / transformers / demucs
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('huggingface_hub')
datas += copy_metadata('transformers')
datas += copy_metadata('faster_whisper')
datas += copy_metadata('demucs')
datas += copy_metadata('edge_tts')

# Hidden imports
hiddenimports = [
    'dubber',
    'torch._dynamo',
    'torch._dynamo.polyfills',
    'torch._dynamo.polyfills.copy',
    'torch._dynamo.polyfills.heapq',
    'torch._dynamo.polyfills.fx',
    'torch._dynamo.polyfills.tensor',
    'scipy.signal',
    'scipy.special',
    'scipy.sparse.csgraph',
    'sklearn',
    'rapidocr_onnxruntime',
    'onnxruntime',
    'ctranslate2',
    'faster_whisper',
    'demucs',
    'demucs.apply',
    'demucs.pretrained',
    'demucs.htdemucs',
    'transformers',
    'transformers.models.nllb',
    'transformers.models.m2m_100',
    'transformers.models.m2m_100.modeling_m2m_100',
    'sentencepiece',
    'edge_tts',
    'librosa',
    'soundfile',
    'cv2',
    'winsound',
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
]
hiddenimports += collect_submodules('dubber')
hiddenimports += collect_submodules('demucs')
hiddenimports += collect_submodules('faster_whisper')

a = Analysis(
    ['app.py'],
    pathex=[base_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'notebook', 'IPython', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VideoDubber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(base_dir, 'icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='VideoDubber',
)
