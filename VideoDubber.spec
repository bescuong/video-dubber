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
venv_site = os.path.join(base_dir, '.venv', 'Lib', 'site-packages')

# Ensure PyInstaller child processes (collect_submodules) can load torch/c10.dll & VC++ DLLs cleanly
_extra_dll_dirs = [
    venv_scripts,
    os.path.join(venv_site, 'torch', 'lib'),
    os.path.join(venv_site, 'ctranslate2'),
]
_extra_dll_dirs = [d for d in _extra_dll_dirs if os.path.isdir(d)]
if _extra_dll_dirs:
    os.environ['PATH'] = os.pathsep.join(_extra_dll_dirs) + os.pathsep + os.environ.get('PATH', '')
    for _d in _extra_dll_dirs:
        try:
            os.add_dll_directory(_d)
        except Exception:
            pass

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
datas += collect_data_files('omnivoice')
datas += collect_data_files('accelerate')
datas += collect_data_files('certifi')

# Explicitly bundle torch._dynamo.polyfills to prevent missing dynamic polyfill modules (copy.py, heapq.py, etc.)
torch_polyfills = os.path.join(base_dir, '.venv', 'Lib', 'site-packages', 'torch', '_dynamo', 'polyfills')
if os.path.isdir(torch_polyfills):
    datas.append((torch_polyfills, os.path.join('torch', '_dynamo', 'polyfills')))

# Package metadata required by huggingface_hub / transformers / demucs / omnivoice / accelerate
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
datas += copy_metadata('omnivoice')
datas += copy_metadata('accelerate')
datas += copy_metadata('safetensors')

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
    'transformers.models.higgs_audio_v2_tokenizer',
    'transformers.models.higgs_audio_v2_tokenizer.configuration_higgs_audio_v2_tokenizer',
    'transformers.models.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer',
    'transformers.models.dac',
    'transformers.models.dac.configuration_dac',
    'transformers.models.dac.feature_extraction_dac',
    'transformers.models.dac.modeling_dac',
    'transformers.models.hubert',
    'transformers.models.hubert.configuration_hubert',
    'transformers.models.hubert.modeling_hubert',
    'transformers.models.qwen2',
    'transformers.models.qwen2.tokenization_qwen2',
    'transformers.models.qwen2.tokenization_qwen2_fast',
    'transformers.models.qwen3',
    'transformers.models.qwen3.configuration_qwen3',
    'transformers.models.qwen3.modeling_qwen3',
    'transformers.models.audioflamingo3',
    'transformers.models.encoder_decoder',
    'transformers.models.encoder_decoder.configuration_encoder_decoder',
    'omnivoice',
    'accelerate',
    'pydub',
    'pydub.silence',
    'audioop',
    'safetensors',
    'certifi',
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
# Static filesystem submodule collector (fast & immune to Windows DLL search order in subprocesses)
def _walk_pkg(pkg_name, skip_prefixes=()):
    pkg_path = os.path.join(venv_site, *pkg_name.split('.'))
    mods = [pkg_name]
    if not os.path.isdir(pkg_path):
        return mods
    for root, dirs, files in os.walk(pkg_path):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if f.endswith('.py'):
                rel = os.path.relpath(os.path.join(root, f), venv_site)
                mod = rel[:-3].replace(os.sep, '.')
                if mod.endswith('.__init__'):
                    mod = mod[:-9]
                if any(mod == p or mod.startswith(p + '.') for p in skip_prefixes):
                    continue
                if mod not in mods:
                    mods.append(mod)
    return mods


hiddenimports += collect_submodules('dubber')
hiddenimports += _walk_pkg('demucs')
hiddenimports += _walk_pkg('faster_whisper')
hiddenimports += _walk_pkg('omnivoice', skip_prefixes=('omnivoice.training', 'omnivoice.eval', 'omnivoice.scripts', 'omnivoice.cli', 'omnivoice.data'))
hiddenimports += _walk_pkg('accelerate', skip_prefixes=('accelerate.test_utils', 'accelerate.commands'))
hiddenimports += _walk_pkg('pydub')
hiddenimports += _walk_pkg('transformers.models.higgs_audio_v2_tokenizer')
hiddenimports += _walk_pkg('transformers.models.dac')
hiddenimports += _walk_pkg('transformers.models.hubert')
hiddenimports += _walk_pkg('transformers.models.qwen2')
hiddenimports += _walk_pkg('transformers.models.qwen3')

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
