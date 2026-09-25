"""助手风格包（LoRA）。

产品语言里它叫「助手风格」：用户选的是"让它像一个什么样的人说话"，不是
"挂一个适配器"。技术上就是 llama.cpp 的 --lora，而且可以叠加 —— 我们
实测过破限适配器和风格包能同时生效。

三种来源：
  bundled    随程序分发（许可证允许的才放这里）
  installed  用户自己加的，或从精选目录一键装的
  catalog    我们知道、但不再分发的（许可证不允许） —— 一键从原作者仓库取

为什么风格包不全部打进 exe：一个 rank-16 的 q8_0 适配器就有 150-180 MB，
三个就是 300+ MB，会让"第一次下载"从 36 MB 变成 360 MB。放进目录一键装，
对用户是同一个体验，对我们的发布包是几十倍的差别。
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, resource_dir

USER_AGENT = "BonsaiLocal/1.0 (+https://github.com/Eyeing0721/bonsai-local)"
CATALOG_REMOTE = ("https://github.com/Eyeing0721/bonsai-local/releases/latest/download/"
                  "loras.json")


@dataclass
class Lora:
    id: str
    name: str
    desc: str
    path: Path | None
    origin: str                 # bundled | installed | catalog
    size: int = 0
    license: str = ""
    source: str = ""            # 原仓库，用于署名

    def as_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "desc": self.desc,
            "origin": self.origin, "size": self.size, "license": self.license,
            "source": self.source,
            "installed": self.path is not None,
            "path": str(self.path) if self.path else "",
        }


def bundled_dir() -> Path:
    return resource_dir() / "assets" / "loras"


def installed_dir(settings: Settings) -> Path:
    d = settings.data_dir / "loras"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ catalog
def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:                                           # noqa: BLE001
        return None


def load_catalog() -> list[dict]:
    """精选目录：先看包内，再看发布页，最后看环境变量（开发用）。

    和引擎清单同样的道理 —— 首次运行不该依赖一次网络往返才知道有哪些可选。
    """
    for base in (resource_dir() / "assets", resource_dir()):
        got = _read_json(base / "loras.json")
        if got and isinstance(got.get("items"), list):
            return got["items"]

    env = os.environ.get("BONSAI_LORA_CATALOG")
    if env and Path(env).exists():
        got = _read_json(Path(env))
        if got and isinstance(got.get("items"), list):
            return got["items"]

    try:
        req = urllib.request.Request(CATALOG_REMOTE, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=12) as r:
            got = json.loads(r.read().decode("utf-8"))
        if isinstance(got.get("items"), list):
            return got["items"]
    except Exception:                                           # noqa: BLE001
        pass
    return []


def scan(settings: Settings) -> list[Lora]:
    """所有可用的风格包 = 随包的 + 已安装的（后者同名时优先）。"""
    out: dict[str, Lora] = {}

    bdir = bundled_dir()
    if bdir.is_dir():
        for p in sorted(bdir.glob("*.gguf")):
            out[p.stem] = Lora(id=p.stem, name=p.stem, desc="随程序附带",
                               path=p, origin="bundled", size=p.stat().st_size)

    for p in sorted(installed_dir(settings).glob("*.gguf")):
        out[p.stem] = Lora(id=p.stem, name=p.stem, desc="已安装",
                           path=p, origin="installed", size=p.stat().st_size)

    # 目录里声明的名字/说明覆盖掉文件名
    for item in load_catalog():
        lid = str(item.get("id") or "")
        if not lid:
            continue
        existing = out.get(lid)
        if existing is not None:
            existing.name = item.get("name") or existing.name
            existing.desc = item.get("desc") or existing.desc
            existing.license = item.get("license") or ""
            existing.source = item.get("source") or ""
        else:
            out[lid] = Lora(id=lid, name=item.get("name") or lid,
                            desc=item.get("desc") or "", path=None, origin="catalog",
                            size=int(item.get("bytes") or 0),
                            license=item.get("license") or "",
                            source=item.get("source") or "")
    return list(out.values())


def find(settings: Settings, lora_id: str) -> Lora | None:
    return next((l for l in scan(settings) if l.id == lora_id), None)


# ------------------------------------------------------------------ install
def install(settings: Settings, lora_id: str, progress) -> Lora:
    """从精选目录一键安装：直接向原作者仓库取，我们不再分发。"""
    item = next((i for i in load_catalog() if str(i.get("id")) == lora_id), None)
    if item is None:
        raise RuntimeError(f"目录里没有 {lora_id}")

    urls = item.get("url")
    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        raise RuntimeError(f"{lora_id} 没有下载地址")

    dest = installed_dir(settings) / f"{lora_id}.gguf"
    part = dest.with_suffix(".gguf.part")
    total = int(item.get("bytes") or 0)
    progress.set(stage="downloading", label=f"安装「{item.get('name') or lora_id}」",
                 done=0, total=total, detail="")

    last_err: Exception | None = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r, part.open("wb") as fh:
                if not total:
                    total = int(r.headers.get("Content-Length") or 0)
                    progress.set(total=total)
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
                    progress.add(len(chunk))
            os.replace(part, dest)
            break
        except Exception as e:                                  # noqa: BLE001
            last_err = e
            part.unlink(missing_ok=True)
            continue
    else:
        raise IOError(f"全部来源都失败了: {last_err}")

    progress.set(stage="ready", label="就绪", done=0, total=0, detail="")
    lora = find(settings, lora_id)
    if lora is None:
        raise RuntimeError("安装完成但扫描不到，请重试")
    return lora


def add_file(settings: Settings, src: str | Path) -> Lora:
    """把用户自己挑的 .gguf 收进来。"""
    src = Path(src)
    if not src.is_file():
        raise RuntimeError("选中的不是文件")
    if src.suffix.lower() not in (".gguf", ".bin"):
        raise RuntimeError("只支持 .gguf 格式的风格包")

    # 文件名保持可读，但去掉路径分隔符之类
    safe = "".join(c for c in src.stem if c.isalnum() or c in "._- ").strip() or "lora"
    dest = installed_dir(settings) / f"{safe}.gguf"
    n = 2
    while dest.exists() and dest.resolve() != src.resolve():
        dest = installed_dir(settings) / f"{safe}-{n}.gguf"
        n += 1
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    return Lora(id=dest.stem, name=dest.stem, desc="我自己加的", path=dest,
                origin="installed", size=dest.stat().st_size)


def remove(settings: Settings, lora_id: str) -> None:
    """只删已安装的；随程序分发的删不掉（也不该删）。"""
    lora = find(settings, lora_id)
    if lora is None or lora.path is None:
        raise RuntimeError("没有这个风格包")
    if lora.origin == "bundled":
        raise RuntimeError("随程序附带的风格包不能删除")
    lora.path.unlink(missing_ok=True)


# ------------------------------------------------------------------ presets
def load_presets() -> list[dict]:
    """预设 = 挑好的 LoRA 组合 + 调好的权重。

    产品语言里用户选的是「像一个什么样的人说话」，不是 rank 和 scale。
    实测同一个中文文风 LoRA 在权重 0.4 / 0.7 / 1.0 下分别是散文腔 / 叙事腔 /
    小说腔，所以一个 LoRA 就能撑起三个不同的预设。
    """
    for base in (resource_dir() / "assets", resource_dir()):
        got = _read_json(base / "presets.json")
        if got and isinstance(got.get("presets"), list):
            return got["presets"]
    env = os.environ.get("BONSAI_PRESETS")
    if env and Path(env).exists():
        got = _read_json(Path(env))
        if got and isinstance(got.get("presets"), list):
            return got["presets"]
    return []


def preset(settings: Settings, preset_id: str) -> dict | None:
    return next((p for p in load_presets() if p.get("id") == preset_id), None)


def preset_uses_core(settings: Settings, preset_id: str) -> bool:
    """当前预设要不要挂去拒答适配器。

    「默认」不挂 —— 它的卖点就是最接近模型原本的样子，挂了就不再是"默认"。
    其余预设都挂：风格包的作用是让模型按你的要求说话，而拒答会把这件事抵消掉，
    两个力量对着拉，结果是风格变得时有时无。找不到预设时按挂上处理（宁可有用）。
    """
    p = preset(settings, preset_id)
    if p is None:
        return True
    return bool(p.get("abliterated", True))


def scaled_copy(settings: Settings, lora: Lora, weight: float) -> Path:
    """权重为 1 就用原文件；否则生成（并缓存）一份缩放过的副本。

    为什么不用 llama.cpp 的 --lora-scaled：它按 ':' 切参数，而 Windows 路径
    自带盘符冒号，会直接报 "lora-scaled format: FNAME:SCALE"。写进文件更稳，
    而且外部 API 调用者拿到的行为和界面上看到的完全一致。
    """
    if lora.path is None:
        raise RuntimeError(f"{lora.id} 还没安装")
    if abs(weight - 1.0) < 1e-6:
        return lora.path

    cache = installed_dir(settings) / ".scaled"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{lora.id}-x{weight:g}.gguf"
    if out.exists() and out.stat().st_mtime >= lora.path.stat().st_mtime:
        return out
    from .lora_scale import rescale_lora
    rescale_lora(str(lora.path), str(out), float(weight))
    return out


def resolve_preset(settings: Settings, preset_id: str, progress=None) -> list[Path]:
    """把预设解析成引擎能直接挂的文件列表，缺的风格包直接跳过而不是报错。"""
    p = preset(settings, preset_id)
    if p is None:
        return []
    items = p.get("loras")
    if items is None:                       # 自定义：用用户自己勾的
        return enabled_paths(settings)

    out: list[Path] = []
    for it in items or []:
        lora = find(settings, str(it.get("id")))
        if lora is None or lora.path is None:
            continue
        w = float(it.get("weight", 1.0))
        if progress is not None and abs(w - 1.0) > 1e-6:
            progress.set(stage="extracting", label=f"准备「{lora.name}」",
                         done=0, total=0, detail=f"权重 {w:g}")
        out.append(scaled_copy(settings, lora, w))
    return out


def preset_state(settings: Settings) -> dict:
    """给界面用的预设列表，并标出哪些因为没装而用不了。"""
    have = {l.id for l in scan(settings) if l.path is not None}
    items = []
    for p in load_presets():
        loras_spec = p.get("loras")
        entry = {k: p[k] for k in ("id", "name", "desc", "hint") if k in p}
        entry |= {"abliterated": bool(p.get("abliterated", True))}
        if loras_spec is None:
            entry |= {"available": True, "missing": [], "custom": True}
        else:
            missing = [str(i.get("id")) for i in loras_spec if str(i.get("id")) not in have]
            entry |= {"available": not missing, "missing": missing, "custom": False}
        items.append(entry)
    return {"items": items, "current": settings.get("preset") or "default"}


# ------------------------------------------------------------------ enable
def enabled_ids(settings: Settings) -> list[str]:
    raw = settings.get("enabled_loras") or []
    return [str(x) for x in raw] if isinstance(raw, list) else []


def set_enabled(settings: Settings, ids: list[str]) -> None:
    known = {l.id for l in scan(settings) if l.path is not None}
    settings.set("enabled_loras", [i for i in ids if i in known])
    settings.save()


def enabled_paths(settings: Settings) -> list[Path]:
    """给引擎用的实际文件路径，跳过已经不见了的。"""
    by_id = {l.id: l for l in scan(settings) if l.path is not None}
    return [by_id[i].path for i in enabled_ids(settings) if i in by_id]
