#!/usr/bin/env python3
"""262144 上下文的 KV 到底分配到哪去了？

现象：-c 262144 时显存只比 -c 131072 多 84 MiB，而按每 token 64 KiB 算应该再多
8 GiB。日志里有一句 common_fit_params: failed to fit ... n_gpu_layers already set
by user to 99, abort —— 因为强制 -ngl 99，自动适配被跳过，于是按请求值硬上。

那这 8 GiB 去哪了？llama.cpp 在显存不够时会退到主机内存。所以同时量显存和
llama-server 进程的工作集，两边一对比就知道。

用法:
    python build/probe_ctx_alloc.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MODEL = Path(r"E:\models\bonsai\Ternary-Bonsai-2-27B-PTQ1_0.gguf")
BIN = Path(r"E:\src\llama-prism\build-cuda-multi\bin")
CUDA = Path(r"E:\cuda\bin")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def vram() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15,
        creationflags=CREATE_NO_WINDOW)
    return float(out.stdout.strip().splitlines()[0])


def proc_ram(pid: int) -> float:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-Process -Id {pid}).WorkingSet64"],
        capture_output=True, text=True, timeout=25,
        creationflags=CREATE_NO_WINDOW)
    try:
        return float(out.stdout.strip()) / 2**20
    except ValueError:
        return 0.0


def post(port: int, body: dict, timeout=900.0):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(ctx: int, kv: str | None) -> None:
    port = free_port()
    log = ROOT / "build" / "bench-logs" / f"alloc-{ctx}{'-' + kv if kv else ''}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(BIN / "llama-server.exe"), "-m", str(MODEL),
           "--host", "127.0.0.1", "--port", str(port),
           "-c", str(ctx), "-ngl", "99", "-fa", "on", "-np", "1",
           "--jinja", "--no-warmup", "--no-webui", "--alias", "bonsai"]
    if kv:
        cmd += ["--cache-type-k", kv, "--cache-type-v", kv]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])
    label = f"ctx={ctx}" + (f" KV={kv}" if kv else "")
    print(f"\n── {label} ──")
    proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh,
                            stderr=subprocess.STDOUT, env=env,
                            creationflags=CREATE_NO_WINDOW)
    try:
        end = time.time() + 300
        while time.time() < end:
            if proc.poll() is not None:
                print(f"  引擎退出，码 {proc.returncode}")
                print("  " + "\n  ".join(
                    log.read_text(encoding="utf-8", errors="replace")
                    .splitlines()[-6:]))
                return
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=3) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(0.5)
        time.sleep(3)
        ram0, vr0 = proc_ram(proc.pid), vram()
        print(f"  就绪后：进程工作集 {ram0:8.0f} MiB   显存 {vr0:7.0f} MiB")

        # 发一个很小的请求：如果 KV 落在主机内存，prefill 会明显变慢
        filler = "背景材料内容。" * 1200        # 约 4200 字 → 约 4000 token
        t0 = time.time()
        r = post(port, {"messages": [{"role": "user", "content": filler + "说一句结束语。"}],
                        "max_tokens": 20, "temperature": 0.0,
                        "chat_template_kwargs": {"enable_thinking": False}})
        wall = time.time() - t0
        u = r.get("usage") or {}
        pt = u.get("prompt_tokens") or 1
        print(f"  4000 token prefill：{wall:6.2f}s → {pt / wall:6.1f} tok/s")

        ram1, vr1 = proc_ram(proc.pid), vram()
        print(f"  请求后：  进程工作集 {ram1:8.0f} MiB   显存 {vr1:7.0f} MiB")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        fh.close()
        time.sleep(4)


def main() -> int:
    base_ram = vram()
    print(f"起始显存占用 {base_ram:.0f} MiB")
    for ctx, kv in ((131072, None), (262144, None), (262144, "q8_0")):
        run(ctx, kv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
