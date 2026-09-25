"""First-run acquisition of the two big things: the model and the GPU engine.

Rules this file follows:
  * never ask the user for a URL, a folder or a version;
  * resume rather than restart -- 5.95 GB over a flaky connection is normal;
  * if a mirror is faster or reachable and the primary is not, use it silently;
  * report progress in bytes and a human label, because a progress bar is the
    only thing standing between "it is working" and "it is broken".
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings

# --------------------------------------------------------------------- manifest
MODEL_FILE = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
MODEL_SIZE = 5_946_648_928
MODEL_SHA256 = "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
MODEL_SOURCES = [
    "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/" + MODEL_FILE,
    "https://hf-mirror.com/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/" + MODEL_FILE,
]

# The rank-1 abliteration adapter is 8 MB, so it ships inside the app rather than
# being downloaded. It is applied at a fixed strength: the product has no knob.
LORA_FILE = "abl-lora-t.gguf"

# Engine builds. A zip containing llama-server plus its DLLs. `cuda-ada` targets
# RTX 40-series (sm_89); `cpu` runs anywhere and is the honest fallback.
#
# This URL is only a *refresh* path: the same file is also shipped inside the app
# (see load_engine_manifest). A fresh install must never need a network round-trip
# merely to discover what it should download.
ENGINE_ASSETS = "https://github.com/Eyeing0721/bonsai-local/releases/latest/download/engines.json"

USER_AGENT = "BonsaiLocal/1.0 (+https://github.com/Eyeing0721/bonsai-local)"
TIMEOUT = 30
STALL_SECONDS = 45          # no bytes for this long => try the next source


# ---------------------------------------------------------------------- progress
@dataclass
class Progress:
    """Single source of truth the UI polls. Cheap to read from another thread."""

    stage: str = "idle"           # idle|checking|downloading|verifying|extracting|starting|ready|error
    label: str = ""
    done: int = 0
    total: int = 0
    detail: str = ""
    error: str = ""
    source: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n

    def snapshot(self) -> dict:
        with self._lock:
            pct = (self.done / self.total * 100.0) if self.total else 0.0
            return {
                "stage": self.stage, "label": self.label, "detail": self.detail,
                "done": self.done, "total": self.total,
                "pct": round(min(pct, 100.0), 2), "error": self.error,
                "source": self.source,
                "mb_done": round(self.done / 2**20, 1),
                "mb_total": round(self.total / 2**20, 1),
            }


PROGRESS = Progress()


# ---------------------------------------------------------------------- helpers
def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def _reachable(url: str, timeout: float = 6.0) -> bool:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 400
    except Exception:
        return False


def _remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def sha256_of(path: Path, progress: Progress | None = None,
              label: str = "校验文件") -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as fh:
        while chunk := fh.read(8 << 20):
            h.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress.set(stage="verifying", label=label, done=done, total=total)
    return h.hexdigest()


# ---------------------------------------------------------------------- download
def download(url: str, dest: Path, progress: Progress, label: str,
             expect_size: int = 0) -> None:
    """Stream to `dest.part` and rename on success. Resumes from whatever is there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0

    total = expect_size or _remote_size(url)
    if have and total and have >= total:
        os.replace(part, dest)
        return

    headers = {"User-Agent": USER_AGENT}
    mode = "wb"
    if have:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"

    req = urllib.request.Request(url, headers=headers)
    progress.set(stage="downloading", label=label, done=have, total=total,
                 source=url.split("/")[2])
    started = time.time()
    last_tick = started
    last_bytes = have

    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        if have and r.status == 200:
            # server ignored Range; start over rather than corrupt the file
            have = 0
            mode = "wb"
            progress.set(done=0)
        if not total:
            cl = int(r.headers.get("Content-Length") or 0)
            total = cl + have
        progress.set(total=total)
        with part.open(mode) as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                progress.add(len(chunk))
                now = time.time()
                if now - last_tick >= 1.0:
                    rate = (progress.done - last_bytes) / (now - last_tick)
                    eta = (total - progress.done) / rate if rate > 0 else 0
                    progress.set(detail=f"{human(int(rate))}/s · 约剩 {int(eta)} 秒")
                    last_tick, last_bytes = now, progress.done
                if now - last_tick > STALL_SECONDS and progress.done == last_bytes:
                    raise TimeoutError("download stalled")

    if total and part.stat().st_size < total:
        raise IOError(f"incomplete: {part.stat().st_size} of {total}")
    os.replace(part, dest)


def fetch_with_fallback(sources: list[str], dest: Path, progress: Progress,
                        label: str, expect_size: int = 0,
                        sha256: str = "") -> None:
    """Try each source in turn. A source that fails mid-way still leaves the .part
    behind, so the next attempt continues instead of starting from zero."""
    last_err: Exception | None = None
    for i, url in enumerate(sources):
        try:
            download(url, dest, progress, label, expect_size)
        except Exception as e:                              # noqa: BLE001
            last_err = e
            progress.set(detail=f"该来源不稳定，换一个…（{type(e).__name__}）")
            continue
        if sha256:
            got = sha256_of(dest, progress, label="校验完整性")
            if got.lower() != sha256.lower():
                dest.unlink(missing_ok=True)
                dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
                last_err = IOError("checksum mismatch")
                continue
        progress.set(detail="")
        return
    raise IOError(f"全部来源都失败了: {last_err}")


# ---------------------------------------------------------------------- engines
def _read_manifest(path: Path) -> dict[str, dict] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data:
            return data
    except Exception:                                           # noqa: BLE001
        pass
    return None


def load_engine_manifest() -> dict[str, dict]:
    """Where each engine build lives.

    Order matters. The copy shipped *inside* the app comes first, so a fresh
    install never needs a network round-trip -- and never depends on a release URL
    being correct -- just to learn what to download. The remote copy exists only so
    new engine builds can ship without a new app version.

    (An earlier version had a hard-coded fallback dictionary that was missing its
    `url` key, so an unreachable manifest turned into `KeyError: 'url'` on the very
    first run of a real install. Hence the explicit, readable failure below.)
    """
    from .config import app_root, resource_dir
    for base in (resource_dir() / "assets", resource_dir(), app_root()):
        got = _read_manifest(base / "engines.json")
        if got:
            return got

    env = os.environ.get("BONSAI_ENGINE_MANIFEST")
    if env:
        got = _read_manifest(Path(env))
        if got:
            return got

    if "OWNER/REPO" not in ENGINE_ASSETS:
        try:
            req = urllib.request.Request(ENGINE_ASSETS,
                                         headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode("utf-8"))
            if isinstance(data, dict) and data:
                return data
        except Exception:                                       # noqa: BLE001
            pass

    raise RuntimeError(
        "找不到引擎清单（engines.json）。\n"
        "正常情况下它应该随程序一起分发；如果你是自行构建的，"
        "请确认 build/make_release.py 已经运行过，并且文件在程序旁边。")


def pick_engine_variant(settings: Settings, gpu: dict | None) -> str:
    pref = settings.get("engine_variant") or "auto"
    if pref != "auto":
        return pref
    return "cuda-ada" if gpu else "cpu"


def engine_dir_for(settings: Settings, variant: str) -> Path:
    return settings.engines_dir / variant


def engine_ready(settings: Settings, variant: str) -> bool:
    exe = engine_dir_for(settings, variant) / "llama-server.exe"
    return exe.exists()


def ensure_engine(settings: Settings, variant: str, progress: Progress,
                  manifest: dict[str, dict] | None = None) -> Path:
    """Unpack the engine zip once. A local build can be pointed at with
    BONSAI_ENGINE_DIR to keep development offline."""
    target = engine_dir_for(settings, variant)
    if engine_ready(settings, variant):
        return target

    dev = os.environ.get("BONSAI_ENGINE_DIR")
    if dev and (Path(dev) / "llama-server.exe").exists():
        progress.set(stage="extracting", label="准备推理引擎", done=0, total=0,
                     detail="使用本地构建")
        target.mkdir(parents=True, exist_ok=True)
        for f in Path(dev).iterdir():
            if f.is_file():
                shutil.copy2(f, target / f.name)
        return target

    manifest = manifest or load_engine_manifest()

    # A GPU build may simply not exist in this release. Degrade to CPU and say so,
    # rather than failing the whole app on a missing key.
    chosen = variant if variant in manifest else ("cpu" if "cpu" in manifest else None)
    if chosen is None:
        raise RuntimeError("引擎清单里没有任何可用构建（含："
                           + "、".join(sorted(manifest)) + "）")
    if chosen != variant:
        progress.set(detail=f"这一版没有 {variant} 引擎，改用 {chosen}（会慢很多）")
    spec = manifest[chosen]

    url = (spec.get("url") or "").strip()
    if not url or "OWNER/REPO" in url:
        raise RuntimeError(
            "引擎下载地址还没配置好。\n"
            "如果你是发布者，请用正确的仓库地址重新打包：\n"
            "    python build/make_release.py --with-engines --repo 你的名字/仓库名")

    target = engine_dir_for(settings, chosen)
    zip_path = settings.engines_dir / f"{chosen}.zip"
    fetch_with_fallback([url], zip_path, progress,
                        label="下载推理引擎", expect_size=int(spec.get("bytes") or 0),
                        sha256=spec.get("sha256") or "")

    progress.set(stage="extracting", label="安装推理引擎", done=0, total=0, detail="")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            # flatten on purpose: it defeats zip-slip, and Windows finds a DLL by
            # the executable's own directory, so the cuda/ folder is optional
            name = Path(member.filename).name
            if not name:
                continue
            with z.open(member) as src, (target / name).open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    zip_path.unlink(missing_ok=True)
    return target


# ----------------------------------------------------------------------- models
def model_path(settings: Settings) -> Path:
    return settings.models_dir / MODEL_FILE


def model_ready(settings: Settings) -> bool:
    p = model_path(settings)
    try:
        return p.exists() and p.stat().st_size >= MODEL_SIZE - (1 << 20)
    except OSError:
        return False


def ensure_model(settings: Settings, progress: Progress) -> Path:
    dest = model_path(settings)
    if model_ready(settings):
        return dest

    usable = [u for u in MODEL_SOURCES if _reachable(u)]
    if not usable:
        usable = MODEL_SOURCES
    progress.set(stage="checking", label="连接模型仓库", done=0, total=MODEL_SIZE, detail="")
    fetch_with_fallback(usable, dest, progress,
                        label="下载模型（5.5 GB，只需一次）",
                        expect_size=MODEL_SIZE, sha256=MODEL_SHA256)
    return dest


def bundled_lora() -> Path | None:
    """The adapter we ship. Read-only, next to the app."""
    from .config import resource_dir
    for base in (resource_dir() / "assets", resource_dir().parent / "assets"):
        p = base / LORA_FILE
        if p.exists():
            return p
    return None
