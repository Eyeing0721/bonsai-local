"""向量模型服务：知识库的语义那一路。

为什么不把向量模型塞进主引擎：llama.cpp 一个进程只服务一个模型权重。主进程
跑 27B 三值模型，向量模型只能另起一个 llama-server。好在引擎二进制是同一套
（连同 CUDA 运行库），所以"另起一个"的代价只是多一个进程和 0.6 GB 显存。

为什么按需下载：610 MB。用不上知识库的人不该为它付这次下载。所以它和风格包
一样，等用户真的启用了向量检索才去取。
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from . import fetch
from .config import Settings, free_port

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

EMBED_FILE = "Qwen3-Embedding-0.6B-Q8_0.gguf"
EMBED_SIZE = 639_150_592
EMBED_SHA256 = "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439"

EMBED_SOURCES = [
    "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/"
    "Qwen3-Embedding-0.6B-Q8_0.gguf",
    "https://hf-mirror.com/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/"
    "Qwen3-Embedding-0.6B-Q8_0.gguf",
]

# Qwen3-Embedding 是 decoder 结构，取最后一个 token 的隐状态，不是取平均。
# 用错 pooling 不会报错，只会让相似度变得没有意义 —— 所以这里写死并在
# 启动时用一组已知样本自检（见 selftest）。
POOLING = "last"

# 向量批次。切块是 420 字左右，2048 的 ubatch 一次能吃下好几个块；同时
# llama-server 要求 embeddings 的 n_batch <= n_ubatch，否则会告警并可能截断。
BATCH = 2048


def embed_model_path(settings: Settings) -> Path:
    """向量模型位置。

    BONSAI_EMBED_MODEL 是开发用旁路，和引擎的 BONSAI_ENGINE_DIR 一个道理：
    不想为了试一次检索就复制 610 MB 到测试目录里。
    """
    import os
    dev = os.environ.get("BONSAI_EMBED_MODEL")
    if dev and Path(dev).is_file():
        return Path(dev)
    return settings.models_dir / EMBED_FILE


def embed_ready(settings: Settings) -> bool:
    p = embed_model_path(settings)
    try:
        return p.is_file() and p.stat().st_size >= EMBED_SIZE
    except OSError:
        return False


def ensure_embed_model(settings: Settings, progress=None) -> Path:
    """按需取向量模型。已经在了就直接返回。"""
    dest = embed_model_path(settings)
    if embed_ready(settings):
        return dest
    progress = progress or fetch.PROGRESS
    fetch.fetch_with_fallback(EMBED_SOURCES, dest, progress, "向量模型（610 MB）",
                              expect_size=EMBED_SIZE, sha256=EMBED_SHA256)
    return dest


class EmbedServer:
    """一个只跑向量模型的 llama-server。懒启动，空闲不占资源。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self.log_path: Path | None = None
        self._log_fh = None
        self._lock = threading.RLock()
        self._dim = 0

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def dim(self) -> int:
        return self._dim

    # -- 生命周期 ------------------------------------------------------
    def _command(self, engine_dir: Path, model: Path) -> list[str]:
        exe = engine_dir / "llama-server.exe"
        if not exe.exists():
            raise RuntimeError(f"推理引擎不完整：缺少 {exe.name}")
        cmd = [
            str(exe), "-m", str(model),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--embeddings", "--pooling", POOLING,
            "-c", "8192", "-b", str(BATCH), "-ub", str(BATCH),
            "-np", "1", "--no-webui", "--alias", "embed",
        ]
        # 向量模型很小，能上 GPU 就上；显存不够时 llama.cpp 自己会退。
        if self.settings.get("engine_variant") == "cpu":
            cmd += ["-ngl", "0"]
        else:
            cmd += ["-ngl", "99"]
        return cmd

    def start(self, model: Path, variant: str = "") -> None:
        with self._lock:
            if self.running:
                return
            if not variant:
                # 不传 None 给 pick_engine_variant：它把「没探测到显卡」理解成
                # 「该用 CPU」，而这里只是没去探测而已，会静默退化成慢路径。
                from .config import detect_gpu
                variant = fetch.pick_engine_variant(self.settings, detect_gpu())
            engine_dir = fetch.engine_dir_for(self.settings, variant)
            self.port = free_port()

            self.settings.logs_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.settings.logs_dir / "embed.log"
            self._log_fh = self.log_path.open("w", encoding="utf-8", errors="replace")

            helper = _EngineEnv(self.settings)
            self.proc = subprocess.Popen(
                self._command(engine_dir, model), cwd=str(engine_dir),
                stdout=self._log_fh, stderr=subprocess.STDOUT,
                env=helper.child_env(engine_dir), creationflags=CREATE_NO_WINDOW,
            )
            self._wait_ready()

    def _wait_ready(self, timeout: float = 180.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is None:
                raise RuntimeError("向量服务未启动")
            if self.proc.poll() is not None:
                raise RuntimeError(f"向量服务启动失败（退出码 {self.proc.returncode}）\n"
                                   f"{self._log_tail()}")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                    if r.status == 200 and json.loads(r.read()).get("status") == "ok":
                        return
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(0.6)
        raise RuntimeError(f"向量服务启动超时\n{self._log_tail()}")

    def _log_tail(self, lines: int = 10) -> str:
        if not self.log_path or not self.log_path.exists():
            return ""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def stop(self) -> None:
        with self._lock:
            p, self.proc = self.proc, None
            if p is not None and p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                except Exception:                               # noqa: BLE001
                    pass
            if self._log_fh is not None:
                try:
                    self._log_fh.close()
                except Exception:                               # noqa: BLE001
                    pass
                self._log_fh = None

    # -- 调用 ----------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """一批文本 -> 一批归一化向量。"""
        if not texts:
            return []
        model = embed_model_path(self.settings)
        if not self.running:
            self.start(model)
        body = json.dumps({"input": texts, "model": "embed"}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/embeddings", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=600) as r:
            payload = json.loads(r.read())
        items = sorted(payload["data"], key=lambda d: d.get("index", 0))
        vecs = [_l2(v["embedding"]) for v in items]
        if vecs:
            self._dim = len(vecs[0])
        return vecs

    # -- 自检 ----------------------------------------------------------
    def selftest(self) -> tuple[bool, str]:
        """确认 pooling 设置是对的。

        用错 pooling 不会报错，只会让相似度失去意义 —— 这类错误在界面上完全
        看不出来，只会表现为"检索结果很随机"。所以启动后拿一对已知该相近、
        一对已知该无关的句子对一下：相近的必须明显高于无关的。
        """
        try:
            vs = self.embed([
                "堆溢出怎么利用",
                "heap overflow exploitation",
                "今天中午吃什么比较好",
            ])
        except Exception as e:                                  # noqa: BLE001
            return False, f"向量服务调用失败：{e}"
        if len(vs) < 3:
            return False, "向量服务返回的条数不对"
        near = _dot(vs[0], vs[1])
        far = _dot(vs[0], vs[2])
        ok = near > far + 0.05
        return ok, (f"相近 {near:.3f} vs 无关 {far:.3f}"
                    + ("" if ok else "  ← 区分度不足，pooling 可能设错了"))


def _l2(vec: list[float]) -> list[float]:
    import math
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class _EngineEnv:
    """借 Engine 的环境拼装逻辑，避免在三个地方各写一遍 CUDA 目录探测。"""

    def __init__(self, settings: Settings) -> None:
        from .engine import Engine
        self._engine = Engine(settings)

    def child_env(self, engine_dir: Path) -> dict:
        return self._engine._child_env(engine_dir)              # noqa: SLF001
