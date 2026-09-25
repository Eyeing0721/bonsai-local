"""Settings the user is allowed to see, and everything they are not.

The rule for this app: anything a person cannot make a meaningful decision about
is derived automatically (port, thread count, GPU layers, batch size, engine
flags). What stays configurable is written in the product's own vocabulary --
"记忆容量" instead of n_ctx, "存储位置" instead of model path.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import threading
from pathlib import Path

APP_NAME = "BonsaiLocal"
APP_TITLE = "Bonsai 本地助手"
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------- memory tiers
# A user picks a feeling, not a number. n_ctx is an implementation detail.
MEMORY_TIERS: dict[str, dict] = {
    "short":    {"label": "简短对话", "ctx": 8192,  "hint": "占用最少，适合日常问答"},
    "standard": {"label": "标准",     "ctx": 16384, "hint": "推荐，记忆与速度平衡"},
    "long":     {"label": "长文档",   "ctx": 32768, "hint": "能读长材料，占用更多显存"},
}
DEFAULT_TIER = "standard"

DEFAULTS: dict = {
    "memory_tier": DEFAULT_TIER,
    "data_dir": "",              # empty => default_data_dir()
    "remote_enabled": False,
    "api_token": "",
    "engine_variant": "auto",    # auto | cuda | cpu
    "first_run_done": False,
    "last_model": "",            # reserved: only one model ships today
    "enabled_loras": [],         # 助手风格包（不含始终生效的去拒答适配器）
    "preset": "default",         # 预设：挑好的组合 + 调好的权重
    "kb_enabled": True,          # 有资料时自动参考；关掉就是纯聊天
    "kb_dense": False,           # 向量检索：要额外下 610 MB 的向量模型
}


def default_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / APP_NAME


def app_root() -> Path:
    """Folder the app lives in (works both frozen by PyInstaller and from source)."""
    import sys
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def resource_dir() -> Path:
    """Where bundled read-only resources (the UI) live."""
    import sys
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


class Settings:
    """JSON settings with atomic writes and a lock, so two windows cannot clobber it."""

    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._path = path
        self._data: dict = dict(DEFAULTS)
        if path is not None:
            self.load()
        else:
            # normalise even before a file exists, otherwise a fresh install has
            # an empty API token and every /v1 request is unauthenticated-but-open
            self._normalise()

    # -- persistence ------------------------------------------------------
    @property
    def path(self) -> Path:
        if self._path is None:
            self._path = self.data_dir / "settings.json"
        return self._path

    def load(self) -> None:
        with self._lock:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = {**DEFAULTS, **raw}
            except FileNotFoundError:
                pass
            except Exception:                       # corrupt file: keep defaults
                pass
            self._normalise()

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, self.path)          # atomic on Windows and POSIX

    def _normalise(self) -> None:
        if self._data.get("memory_tier") not in MEMORY_TIERS:
            self._data["memory_tier"] = DEFAULT_TIER
        if not self._data.get("api_token"):
            self._data["api_token"] = new_token()
        if not isinstance(self._data.get("enabled_loras"), list):
            self._data["enabled_loras"] = []

    # -- accessors --------------------------------------------------------
    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, DEFAULTS.get(key, default))

    def set(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = value

    def update(self, **kw) -> None:
        with self._lock:
            self._data.update(kw)

    def as_dict(self) -> dict:
        with self._lock:
            return dict(self._data)

    # -- derived paths ----------------------------------------------------
    @property
    def data_dir(self) -> Path:
        raw = self.get("data_dir") or ""
        return Path(raw) if raw else default_data_dir()

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def engines_dir(self) -> Path:
        return self.data_dir / "engines"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def knowledge_dir(self) -> Path:
        return self.data_dir / "knowledge"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir, self.engines_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- product-facing values -------------------------------------------
    @property
    def context_size(self) -> int:
        return MEMORY_TIERS[self.get("memory_tier")]["ctx"]

    def rotate_token(self) -> str:
        tok = new_token()
        self.set("api_token", tok)
        self.save()
        return tok


def new_token() -> str:
    """Short enough to type from a phone, long enough not to be guessed."""
    return "sk-" + secrets.token_hex(16)


# ------------------------------------------------------------------ hardware
# 我们为哪些计算能力编了 CUDA 内核（见 build/make_release.py 的架构列表）。
# 低于这个值就不下载 GPU 引擎：下了也用不了，还会白等几分钟。
MIN_CUDA_COMPUTE_CAP = 75          # sm_75 = Turing（RTX 20 系）


def _nvidia_smi(exe: str, query: str) -> str | None:
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except Exception:                                           # noqa: BLE001
        pass
    return None


def detect_gpu() -> dict | None:
    """Best-effort NVIDIA query. Absence is not an error: we fall back to CPU."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None

    base = _nvidia_smi(exe, "name,memory.total")
    if not base:
        return None
    name, _, vram = base.rpartition(",")

    # compute_cap 需要较新的驱动；拿不到就当作"未知"，仍然尝试 GPU
    cap_raw = _nvidia_smi(exe, "compute_cap")
    cap: int | None = None
    if cap_raw:
        try:
            cap = int(round(float(cap_raw) * 10))       # "8.9" -> 89
        except ValueError:
            cap = None

    return {
        "name": name.strip(),
        "vram_mb": int(vram.strip() or 0),
        "compute_cap": cap,
        "cuda_ok": cap is None or cap >= MIN_CUDA_COMPUTE_CAP,
    }


def gpu_summary(gpu: dict | None) -> str:
    """一句话描述运行方式，给界面用。"""
    if not gpu:
        return "CPU（没有检测到 NVIDIA 显卡）"
    cap = gpu.get("compute_cap")
    cap_txt = f" · 计算能力 {cap / 10:.1f}" if cap else ""
    if not gpu.get("cuda_ok", True):
        return f"CPU（{gpu['name']} 的计算能力 {cap / 10:.1f} 低于 {MIN_CUDA_COMPUTE_CAP / 10:.1f}，用不了 GPU 加速）"
    return f"GPU · {gpu['name']}{cap_txt}"


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def local_ip() -> str:
    """LAN address, only used to show the user a friendly fallback URL."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
