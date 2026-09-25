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
    # numpy 是硬依赖，不能排除：
    #   lora_scale.py 顶层就 import 它（给预设调权重时用）
    #   knowledge.py 用它存向量矩阵
    # 曾经它在 excludes 里，症状是"打包版一选非 1.0 权重的预设就崩" ——
    # 源码跑得好好的，只有冻结后才现形。
    "numpy",
    # PDF 是普通人最常往里丢的东西，pypdf 是纯 Python、没有原生依赖，
    # 换来的是"拖个 PDF 进去就能问"。
    "pypdf",
]

excludes = [
    # ─────────────────────────────────────────────────────────────────────
    # 为什么这一串这么长：PyInstaller 会跟踪**函数体内**的 import。
    # gguf/vocab.py 第 404 行有个函数里写着 `from transformers import
    # AutoTokenizer`（我们从不调用它），而 lora_scale.py 需要 gguf。
    # 于是依赖图变成：
    #     lora_scale → gguf → vocab → transformers
    #         → hook-transformers 把整个 site-packages 收进来
    #           （torch / tensorflow / timm / sklearn / diffusers / yt_dlp …）
    # 结果 CArchive 条目数溢出，构建直接崩（struct.error: argument out of
    # range）。所以这些名字不是顺手排掉，是必须挡住的那条路径。
    # ─────────────────────────────────────────────────────────────────────
    "transformers", "sentencepiece", "tokenizers", "huggingface_hub",
    "safetensors", "peft", "datasets", "accelerate", "bitsandbytes", "unsloth",
    "torch", "torchvision", "torchaudio", "tensorflow", "keras",
    "diffusers", "gradio", "timm", "sklearn", "scipy", "pandas", "matplotlib",
    "PIL", "cv2", "librosa", "soundfile", "sounddevice", "numba", "llvmlite",
    "av", "imageio", "pyarrow", "faiss", "triton", "onnxruntime", "yt_dlp",
    "uvicorn", "fastapi", "starlette", "sqlalchemy", "grpc", "google",
    "sympy", "networkx", "h5py", "tifffile", "numexpr", "sentence_transformers",
    # 下面这些本来就在，保留
    "PyQt5", "PySide2", "PySide6", "IPython", "pytest", "setuptools", "pip",
    "notebook", "sqlite3", "unittest", "pydoc", "doctest",
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
