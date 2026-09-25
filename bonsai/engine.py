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


class Engine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.proc: subprocess.Popen | None = None
        self.port: int = 0
        self.log_path: Path | None = None
        self._log_fh = None
        self._gpu: dict | None = None
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
        """
        out: list[Path] = []
        packed = engine_dir / "cuda"
        if packed.is_dir():
            out.append(packed)
        env = os.environ.get("BONSAI_CUDA_DIR")
        if env and Path(env).is_dir():
            out.append(Path(env))
        import glob
        for pattern in (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v1[23]*\bin",
                        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*\bin"):
            for p in sorted(glob.glob(pattern), reverse=True):
                out.append(Path(p))
        return out

    def _child_env(self, engine_dir: Path) -> dict:
        env = os.environ.copy()
        parts = [str(engine_dir)] + [str(d) for d in self._cuda_dirs(engine_dir)]
        parts.append(env.get("PATH", ""))
        env["PATH"] = os.pathsep.join(parts)
        return env

    def _command(self, engine_dir: Path, model: Path, lora: Path | None) -> list[str]:
        exe = engine_dir / "llama-server.exe"
        if not exe.exists():
            raise EngineError(f"推理引擎不完整：缺少 {exe.name}")

        threads = max(4, min((os.cpu_count() or 8) // 2, 16))
        gpu = self.gpu()
        ctx = self.settings.context_size

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
        if lora is not None:
            # always applied, never exposed: the product ships one behaviour
            cmd += ["--lora", str(lora)]
        if gpu:
            cmd += ["-ngl", "99", "-fa", "on"]
        else:
            cmd += ["-ngl", "0"]
        return cmd

    def start(self, model: Path, lora: Path | None) -> None:
        with self._lock:
            if self.running:
                return
            variant = fetch.pick_engine_variant(self.settings, self.gpu())
            engine_dir = fetch.engine_dir_for(self.settings, variant)
            self.port = free_port()

            self.settings.logs_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.settings.logs_dir / "engine.log"
            self._log_fh = self.log_path.open("w", encoding="utf-8", errors="replace")

            cmd = self._command(engine_dir, model, lora)
            fetch.PROGRESS.set(stage="starting", label="启动推理引擎", done=0, total=0,
                               detail=f"{gpu_summary(self.gpu())} · "
                                      f"{self.settings.context_size // 1024}K 上下文")
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
                tail = self._log_tail()
                raise EngineError(f"引擎启动失败（退出码 {self.proc.returncode}）\n{tail}")
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

    def restart(self, model: Path, lora: Path | None) -> None:
        self.stop()
        self.start(model, lora)

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        return {
            "running": self.running,
            "port": self.port,
            "context": self.settings.context_size,
            "gpu": self.gpu(),
            "log_tail": self._log_tail(6) if not self.running else "",
        }
