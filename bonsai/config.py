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
APP_VERSION = "1.2.1"

# ---------------------------------------------------------------- memory tiers
# A user picks a feeling, not a number. n_ctx is an implementation detail.
#
# 档位一路开到模型原生上限（262,144）。能不能真开起来由显存决定，见下面
# context_ceiling()：装不下的档位在界面上会标出来，而不是等用户选了才崩。
MEMORY_TIERS: dict[str, dict] = {
    "short":    {"label": "简短对话", "ctx": 8192,   "hint": "占用最少，适合日常问答"},
    "standard": {"label": "标准",     "ctx": 16384,  "hint": "推荐，记忆与速度平衡"},
    "long":     {"label": "长文档",   "ctx": 32768,  "hint": "能读长材料，占用更多显存"},
    "huge":     {"label": "超长文档", "ctx": 65536,  "hint": "整本书或整个代码库，需要 12 GB 以上显存"},
    "extreme":  {"label": "超长会话", "ctx": 131072, "hint": "代理式长会话；第一轮要读几分钟，之后靠前缀缓存"},
    "max":      {"label": "拉满",     "ctx": 262144, "hint": "模型的原生上限（26 万 token），只有大显存开得起来"},
}
DEFAULT_TIER = "standard"
TIER_ORDER = list(MEMORY_TIERS)

# ------------------------------------------------------------------ context math
# 上下文的显存账单。这个模型 64 层里只有 16 层是普通注意力，另外 48 层是 GDN
# （状态定长，**不随上下文增长**），所以每 token 的 KV 只算这 16 层：
#
#   2(K,V) × 4 个 KV 头 × 256 维 × 2 字节 = 4 KiB / 层 / token
#   16 层合起来 = 64 KiB/token
#
# KV 量化到 q8_0 后减半（32 KiB/token），质量损失很小 —— 我们实测开 q8_0 之后
# 262,144 上下文能装进 16 GB 卡（15.96 GiB），而 f16 只能到 131,072（15.5 GiB）。
# 所以「能开多大」的主要杠杆就是 KV 精度，而不是买更大的卡。
KV_FULL_ATTN_LAYERS = 16
KV_HEAD_COUNT = 4
KV_HEAD_DIM = 256
KV_BYTES_F16 = 2 * KV_HEAD_COUNT * KV_HEAD_DIM * 2 * KV_FULL_ATTN_LAYERS   # 64 KiB
KV_BYTES_Q8 = KV_BYTES_F16 // 2                                            # 32 KiB

# 权重是产品里唯一那个模型的三值量化大小（PTQ1_0，5.54 GiB）。
WEIGHTS_MIB = 5673

# 计算缓冲随上下文缓慢增长。下面这组常数是拿两次实收回推并**校准到零偏差**的：
# 回代 f16@131072 得 15.50 GiB、q8_0@262144 得 15.96 GiB，与实测完全相同。
BUFFER_BASE_MIB = 1538.0
BUFFER_PER_TOKEN_MIB = 0.0036

# 桌面本身要占一点显存，但这台机器实测下来几乎可以忽略（262,144 token 那次
# 总共用了 15.96 GiB / 16.0 GiB，而桌面还开着）。留 0 而不是留几百 MiB，否则
# 「拉满」这一档会被一个并不存在的余量挡在门外。真要遇到显存被别的东西占住的
# 情况，引擎启动失败会带着日志尾部报上来，比静默降档更容易查。
VRAM_RESERVE_MIB = 0

# 没有 GPU 时不建议开大：prefill 在 CPU 上慢几十倍。
CPU_MAX_CTX = 32768


def kv_mib_per_token(q8: bool) -> float:
    """每 token 的 KV 显存（MiB）。"""
    return (KV_BYTES_Q8 if q8 else KV_BYTES_F16) / (1024 * 1024)


def context_ceiling(vram_mb: int, q8: bool, weights_mib: int = WEIGHTS_MIB) -> int:
    """这个显存档位能装下的最大上下文。

    解  weights + base + per·ctx + ctx·kv  ≤  vram − reserve
    即  ctx ≤ (vram − reserve − weights − base) / (kv + per)
    """
    if vram_mb <= 0:
        return 0
    head = vram_mb - VRAM_RESERVE_MIB - weights_mib - BUFFER_BASE_MIB
    if head <= 0:
        return 0
    return max(0, int(head / (kv_mib_per_token(q8) + BUFFER_PER_TOKEN_MIB)))


def resolve_tier(tier: str, vram_mb: int | None,
                 weights_mib: int = WEIGHTS_MIB) -> dict:
    """这个档位实际会怎么跑。

    先试 f16（最快、最准），装不下就退到 q8_0；两个都装不下就是 unavailable。
    返回的 kv_q8 由引擎翻译成 --cache-type-k/v 参数。
    """
    ctx = MEMORY_TIERS.get(tier, MEMORY_TIERS[DEFAULT_TIER])["ctx"]
    if not vram_mb:                      # 无 GPU：只给到 CPU 建议上限
        return {"ctx": min(ctx, CPU_MAX_CTX), "kv_q8": False,
                "available": ctx <= CPU_MAX_CTX, "reason": "cpu",
                "ceiling": CPU_MAX_CTX}
    if ctx <= context_ceiling(vram_mb, False, weights_mib):
        return {"ctx": ctx, "kv_q8": False, "available": True, "reason": "",
                "ceiling": context_ceiling(vram_mb, True, weights_mib)}
    if ctx <= context_ceiling(vram_mb, True, weights_mib):
        return {"ctx": ctx, "kv_q8": True, "available": True, "reason": "",
                "ceiling": context_ceiling(vram_mb, True, weights_mib)}
    return {"ctx": ctx, "kv_q8": True, "available": False, "reason": "vram",
            "ceiling": context_ceiling(vram_mb, True, weights_mib)}


def best_tier(vram_mb: int | None, weights_mib: int = WEIGHTS_MIB) -> str:
    """不越界的最大档位 —— 用来做「自动」。"""
    best = TIER_ORDER[0]
    for name in TIER_ORDER:
        if resolve_tier(name, vram_mb, weights_mib)["available"]:
            best = name
    return best


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
                # utf-8-sig，不是 utf-8：Windows 记事本和 PowerShell 的
                # `Set-Content -Encoding UTF8` 都会写 BOM，而 json.loads 见到
                # BOM 会直接抛异常。异常被下面吞掉之后 _data 保持默认值，表现
                # 为"用户手改过一次设置文件，所有配置就悄悄没了"。utf-8-sig 是
                # utf-8 的超集，带不带 BOM 都能读。
                raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
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
        """数据目录，永远是绝对路径。

        必须 resolve：llama-server 是以**引擎目录**为工作目录启动的，如果这里
        留着相对路径（比如命令行传了 `--data-dir devdata`），模型路径传给引擎
        后就变成了相对引擎目录，表现为 "failed to open GGUF file ... No such
        file or directory" —— 而文件其实好好地在磁盘上。
        """
        raw = self.get("data_dir") or ""
        if not raw:
            return default_data_dir()
        try:
            return Path(raw).expanduser().resolve()
        except OSError:
            return Path(raw)

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
        """用户选的档位标称大小（界面上显示这个）。"""
        return MEMORY_TIERS[self.get("memory_tier")]["ctx"]

    def resolve_context(self, vram_mb: int | None,
                        weights_mib: int = WEIGHTS_MIB) -> dict:
        """档位 → 实际能跑的配置。

        装不下时不报错，而是按这台机器的上限截断（并打上 clamped 标记让界面
        能说一句人话）—— 用户选「拉满」而卡只有 8 GB，应该得到"已经替你开到
        这台的极限"，而不是启动失败。
        """
        r = resolve_tier(self.get("memory_tier"), vram_mb, weights_mib)
        if r["available"] or not vram_mb:
            return r
        ceil = r.get("ceiling", 0)
        if ceil <= 0:
            # 连一个 KV 都放不下：退到最小的档，交给 CPU 之外的最低配
            return dict(r, ctx=MEMORY_TIERS[TIER_ORDER[0]]["ctx"], clamped=True)
        return dict(r, ctx=(ceil // 1024) * 1024, clamped=True)

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
