"""Owns the llama-server process.

Everything here exists so that the user never sees a command line: the port is
picked, the thread count is derived, GPU offload is decided from a hardware probe,
and the process is shut down gracefully so the model file is never left locked.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import fetch
from .config import Settings, detect_gpu, free_port, gpu_summary

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class EngineError(RuntimeError):
    pass


# Windows 上引擎启动失败时返回的是 NTSTATUS，直接甩给用户等于没说。
# 这几个是实际会遇到的，各自对应一句能照做的解释。
_EXIT_HINTS = {
    0xC0000135: "缺少运行库 DLL（STATUS_DLL_NOT_FOUND）。最常见的是没有 CUDA 运行库 "
                "cudart64_12 / cublas64_12 / cublasLt64_12。",
    0xC000007B: "加载了一个位数或版本不对的 DLL（STATUS_INVALID_IMAGE_FORMAT）。",
    0xC000001D: "执行了本机 CPU 不支持的指令（STATUS_ILLEGAL_INSTRUCTION）。",
    0xC0000005: "访问了非法内存（STATUS_ACCESS_VIOLATION）。",
    0xC0000409: "栈溢出或安全检查失败（STATUS_STACK_BUFFER_OVERRUN）。",
    0xC0000142: "DLL 初始化失败（STATUS_DLL_INIT_FAILED）。",
}


def explain_exit(code: int | None) -> str:
    """把引擎的退出码翻成一句人话；认不出来就如实说认不出来。"""
    if code is None:
        return ""
    hint = _EXIT_HINTS.get(code & 0xFFFFFFFF)
    shown = f"{code}（0x{code & 0xFFFFFFFF:08X}）"
    return f"退出码 {shown}：{hint}" if hint else f"退出码 {shown}"


class Engine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.proc: subprocess.Popen | None = None
        self.port: int = 0
        self.log_path: Path | None = None
        self._log_fh = None
        self._gpu: dict | None = None
        self._resolved: dict | None = None
        self._engine_dir: Path | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ state
    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def gpu(self) -> dict | None:
        if self._gpu is None:
            self._gpu = detect_gpu()
        return self._gpu

    # ------------------------------------------------------------------- start
    @staticmethod
    def _cuda_dirs(engine_dir: Path) -> list[Path]:
        """Where the CUDA runtime may live.

        A CUDA build of llama.cpp links cudart64_12, cublas64_12 and (through
        cuBLAS) cublasLt64_12 -- about 860 MB that a public download should not
        have to carry. So we look for it instead, in the order that makes the
        shipped pack self-contained first and the developer's machine last.

        真实踩过的坑：这里原来只列了 NVIDIA 安装器的默认路径，于是把 CUDA 解压到
        E:\\cuda 的机器上，引擎以 0xC0000135（找不到 DLL）退出，界面只显示一个
        NTSTATUS 数字，完全看不出缺什么。所以另外认三处：
          · 发布包把运行库直接放在引擎目录下（dist/engines/engine-cuda.zip 就是这样）
          · NVIDIA 安装器会设 CUDA_PATH / CUDA_HOME；装在非默认盘时靠它们
          · <盘>:\\cuda\\bin —— 解压式安装的常见落点
        """
        out: list[Path] = []
        packed = engine_dir / "cuda"
        if packed.is_dir():
            out.append(packed)
        if (engine_dir / "cudart64_12.dll").exists():
            out.append(engine_dir)          # 发布包：运行库与 exe 同级

        env = os.environ.get("BONSAI_CUDA_DIR")
        if env and Path(env).is_dir():
            out.append(Path(env))
        for var in ("CUDA_PATH", "CUDA_HOME", "CUDA_PATH_V12_9", "CUDA_PATH_V12_8"):
            raw = os.environ.get(var)
            if raw:
                b = Path(raw) / "bin"
                if b.is_dir():
                    out.append(b)

        import glob
        patterns = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v1[23]*\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*\bin",
        ]
        for drive in "CDEFGH":
            patterns.append(rf"{drive}:\cuda\bin")          # 解压式安装
            patterns.append(rf"{drive}:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v1[23]*\bin")
        for pattern in patterns:
            for p in sorted(glob.glob(pattern), reverse=True):
                out.append(Path(p))

        seen: set[str] = set()
        uniq: list[Path] = []
        for p in out:
            k = str(p).lower()
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        return uniq

    def _child_env(self, engine_dir: Path) -> dict:
        env = os.environ.copy()
        parts = [str(engine_dir)] + [str(d) for d in self._cuda_dirs(engine_dir)]
        parts.append(env.get("PATH", ""))
        env["PATH"] = os.pathsep.join(parts)
        return env

    def _command(self, engine_dir: Path, model: Path, loras: list[Path]) -> list[str]:
        exe = engine_dir / "llama-server.exe"
        if not exe.exists():
            raise EngineError(f"推理引擎不完整：缺少 {exe.name}")

        threads = max(4, min((os.cpu_count() or 8) // 2, 16))
        gpu = self.gpu()
        # 档位 → 实际配置。装不下会按这台机器的上限截断，并可能在需要时把 KV
        # 降到 q8_0（每 token 从 64 KiB 减半到 32 KiB，是"能开多大"的主要杠杆）。
        resolved = self.settings.resolve_context((gpu or {}).get("vram_mb"))
        self._resolved = resolved
        ctx = resolved["ctx"]

        cmd = [
            str(exe),
            "-m", str(model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(ctx),
            "-t", str(threads),
            "-np", "1",
            "--jinja",                      # honour the model's own chat template
            "--alias", "bonsai",            # a friendly id in /v1/models
            "--no-webui",                   # the product has its own interface
        ]
        if resolved.get("kv_q8"):
            cmd += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
        # 第一个永远是去拒答适配器（产品行为，不可关）；后面是用户选的助手风格包。
        # llama.cpp 允许多个 --lora 叠加，而且我们实测过它们能同时生效。
        for l in loras:
            cmd += ["--lora", str(l)]

        # n-gram 自投机解码：不需要草稿模型、不占额外显存、不用下载任何东西。
        # 原理是拿已有的上下文去匹配 n-gram，猜接下来几十个 token，主模型一次
        # 前向批量验证 —— 猜对了就白赚，因为验证一次和生成一个 token 读的权重
        # 一样多（稠密模型每出 1 个 token 都要把全部权重读一遍）。
        #
        # 实测（RTX 4060 Ti 16GB，三值 27B）：
        #   输出与上下文逐字重叠时（复述资料 / 续写 / 改代码）
        #       31.7 → 123.3 tok/s，接受率 96.5%，平均起草 47 个 token
        #   普通问答与自由写作：没有增益，但也没有代价（31.8 vs 31.9）
        # 所以默认开着：赚的场景很赚，不赚的场景不亏。
        #
        # 别换成 ngram-cache —— 实测它反而更慢（自由写作 31.8 → 26.0）。
        spec = os.environ.get("BONSAI_SPEC", "ngram-simple")
        if spec and spec.lower() != "off":
            cmd += ["--spec-type", spec]

        if gpu:
            cmd += ["-ngl", "99", "-fa", "on"]
        else:
            cmd += ["-ngl", "0"]
        return cmd

    def start(self, model: Path, loras: list[Path]) -> None:
        with self._lock:
            if self.running:
                # 以前这里是 `return`。于是「重启模型」、切风格包、开关 LoRA 调过来时
                # 引擎正跑着，start() 直接空转返回 —— 配置一个都没生效，界面却报成功。
                # 这是个沉默的谎，改成响亮地拦下来。
                raise EngineError("引擎已在运行，换配置请用 restart()")
            variant = fetch.pick_engine_variant(self.settings, self.gpu())
            engine_dir = fetch.engine_dir_for(self.settings, variant)
            self._engine_dir = engine_dir
            self.port = free_port()

            self.settings.logs_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.settings.logs_dir / "engine.log"
            self._log_fh = self.log_path.open("w", encoding="utf-8", errors="replace")

            cmd = self._command(engine_dir, model, loras)
            r = self._resolved or {}
            bits = [gpu_summary(self.gpu()), f"{r.get('ctx', 0) // 1024}K 上下文"]
            if r.get("kv_q8"):
                bits.append("KV 压缩")
            if r.get("clamped"):
                bits.append("已按本机上限截断")
            fetch.PROGRESS.set(stage="starting", label="启动推理引擎", done=0, total=0,
                               detail=" · ".join(bits))
            self.proc = subprocess.Popen(
                cmd, cwd=str(engine_dir), stdout=self._log_fh,
                stderr=subprocess.STDOUT, env=self._child_env(engine_dir),
                creationflags=CREATE_NO_WINDOW,
            )
            self._wait_ready(timeout=300)

    def _wait_ready(self, timeout: float = 300.0) -> None:
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            if self.proc is None:
                raise EngineError("引擎未启动")
            if self.proc.poll() is not None:
                code = self.proc.returncode
                tail = self._log_tail()
                msg = f"引擎启动失败（{explain_exit(code)}）"
                if (code or 0) & 0xFFFFFFFF == 0xC0000135:
                    # 缺 DLL 时把找过的目录列出来，用户才知道往哪放
                    dirs = self._cuda_dirs(self._engine_dir) if self._engine_dir else []
                    msg += "\n找过这些位置："
                    msg += "".join(f"\n  · {d}" for d in dirs) or "\n  ·（无）"
                    msg += ("\n把 CUDA 运行库放到引擎目录，或设环境变量 BONSAI_CUDA_DIR "
                            "指向含 cudart64_12.dll 的目录。")
                raise EngineError(f"{msg}\n{tail}")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=3) as r:
                    if r.status == 200 and json.loads(r.read()).get("status") == "ok":
                        return
            except Exception as e:                              # noqa: BLE001
                last = type(e).__name__
            time.sleep(0.7)
        raise EngineError(f"引擎启动超时（最后状态 {last}）\n{self._log_tail()}")

    def _log_tail(self, lines: int = 12) -> str:
        if not self.log_path or not self.log_path.exists():
            return ""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    # -------------------------------------------------------------------- stop
    def stop(self) -> None:
        """Terminate, then kill. Never leave the model file mapped."""
        with self._lock:
            p, self.proc = self.proc, None
            if p is not None and p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                except Exception:                               # noqa: BLE001
                    pass
            if self._log_fh is not None:
                try:
                    self._log_fh.close()
                except Exception:                               # noqa: BLE001
                    pass
                self._log_fh = None

    def restart(self, model: Path, loras: list[Path]) -> None:
        self.stop()
        self.start(model, loras)

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        r = self._resolved or self.settings.resolve_context(
            (self.gpu() or {}).get("vram_mb"))
        return {
            "running": self.running,
            "port": self.port,
            "context": r["ctx"],
            "kv_q8": bool(r.get("kv_q8")),
            "clamped": bool(r.get("clamped")),
            "gpu": self.gpu(),
            "log_tail": self._log_tail(6) if not self.running else "",
        }
