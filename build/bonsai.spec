# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置。

产物是一个单文件 exe：界面、附带的方向适配器和业务代码都在里面，
模型和推理引擎在第一次运行时自己下载。
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("BONSAI_SRC", Path(SPECPATH).parent)).resolve()
CONSOLE = os.environ.get("BONSAI_CONSOLE", "0") == "1"

datas = [
    (str(ROOT / "bonsai" / "ui"), "ui"),
]
# 附带的方向适配器（8 MB）。没有它就不是这个产品了，所以直接打进包里。
assets = ROOT / "bonsai" / "assets"
if assets.is_dir():
    datas.append((str(assets), "assets"))

# 引擎清单必须打进包里：首次运行时应用要立刻知道该下哪个引擎，
# 不能依赖一次网络往返（更不能依赖发布地址恰好是对的）。
manifest = ROOT / "engines.json"
if manifest.exists():
    datas.append((str(manifest), "assets"))

hidden = [
    "webview", "webview.platforms.edgechromium",
    "qrcode", "qrcode.image.svg",
    "tkinter", "tkinter.filedialog",
]

excludes = [
    "numpy", "pandas", "matplotlib", "scipy", "PIL", "PyQt5", "PySide2",
    "PySide6", "IPython", "pytest", "setuptools", "pip", "notebook",
    "sqlite3", "unittest", "pydoc", "doctest",
]

a = Analysis(
    [str(ROOT / "build" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="BonsaiLocal",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=CONSOLE,
    disable_windowed_traceback=False,
    icon=str(ROOT / "build" / "app.ico") if (ROOT / "build" / "app.ico").exists() else None,
)
