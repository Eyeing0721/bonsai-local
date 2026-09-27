#!/usr/bin/env python3
"""给一个模型找出「上下文最多能开多大，而且 KV 还留在显存里」。

为什么要区分"能启动"和"KV 在显存里"：显存不够时 llama.cpp **不会报错**，它会
把 KV 溢到主机内存，引擎照常启动照常回答 —— 只是 prefill 慢 8 倍（我们实测过
262144 时从 390 t/s 掉到 48 t/s）。用户只会觉得"模型今天怎么这么卡"。

所以判据是**两条**：
  1) 引擎起来了，并且自报的 n_ctx_slot 等于请求值（没被静默改小）
  2) 进程工作集没有异常膨胀（溢到主机内存时它会涨好几 GB）

用法:
    python build/bench_ctx_ceiling.py --model "E:\\...\\x.gguf" \
        --sizes 32768,65536,131072,262144 --engine-arg=--cpu-moe
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
BIN = Path(r"E:\src\llama-prism\build-cuda-multi\bin")
CUDA = Path(r"E:\cuda\bin")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def vram() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, creationflags=CREATE_NO_WINDOW)
    try:
        return float(out.stdout.strip().splitlines()[0])
    except ValueError:
        return 0.0


def proc_ram(pid: int) -> float:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).WorkingSet64"],
        capture_output=True, text=True, timeout=25, creationflags=CREATE_NO_WINDOW)
    try:
        return float(out.stdout.strip()) / 2**30
    except ValueError:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", default="32768,65536,131072,262144")
    ap.add_argument("--kv", default="", help="如 q8_0 / fp8；空则用默认 f16")
    ap.add_argument("--engine-arg", action="append", default=[])
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"模型不存在：{model}")
        return 1
    label = args.label or model.stem
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    base_vram = vram()
    print(f"模型 {model.name}  {model.stat().st_size / 2**30:.2f} GiB")
    print(f"基线显存 {base_vram:.0f} MiB   附加参数 {args.engine_arg or '（无）'}\n")
    print(f"{'上下文':>9}{'状态':>8}{'显存MiB':>10}{'进程GiB':>9}   判定")

    rows = []
    for ctx in sizes:
        port = free_port()
        log = ROOT / "build" / "bench-logs" / f"ceil-{label}-{ctx}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("w", encoding="utf-8", errors="replace")
        cmd = [str(BIN / "llama-server.exe"), "-m", str(model),
               "--host", "127.0.0.1", "--port", str(port),
               "-c", str(ctx), "-ngl", "99", "-fa", "on", "-np", "1",
               "--jinja", "--no-warmup", "--no-webui", "--alias", "m"]
        if args.kv:
            cmd += ["--cache-type-k", args.kv, "--cache-type-v", args.kv]
        cmd += args.engine_arg
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])
        proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh,
                                stderr=subprocess.STDOUT, env=env,
                                creationflags=CREATE_NO_WINDOW)
        ok = False
        try:
            end = time.time() + 420
            while time.time() < end:
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=3) as r:
                        if json.loads(r.read()).get("status") == "ok":
                            ok = True
                            break
                except Exception:                               # noqa: BLE001
                    pass
                time.sleep(0.6)
            if ok:
                time.sleep(3)
                v, r_ = vram(), proc_ram(proc.pid)
                text = log.read_text(encoding="utf-8", errors="replace")
                m = re.search(r"n_ctx_slot\s*=\s*(\d+)", text)
                reported = int(m.group(1)) if m else 0
                if reported and reported != ctx:
                    verdict = f"⚠ 被静默改成 {reported}"
                elif r_ > model.stat().st_size / 2**30 + 4:
                    verdict = "⚠ 疑似 KV 溢到内存"
                else:
                    verdict = "✓ KV 在显存"
                rows.append({"ctx": ctx, "vram": v, "ram": r_, "ok": True,
                             "verdict": verdict})
                print(f"{ctx:>9}{'成功':>8}{v:>10.0f}{r_:>9.2f}   {verdict}")
            else:
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-2:]
                why = next((x.strip()[:70] for x in tail if "error" in x.lower()),
                           "启动失败")
                rows.append({"ctx": ctx, "ok": False, "why": why})
                print(f"{ctx:>9}{'失败':>8}{'—':>10}{'—':>9}   {why}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            fh.close()
            time.sleep(4)

    ok_rows = [r for r in rows if r.get("ok") and r.get("verdict", "").startswith("✓")]
    print()
    if ok_rows:
        best = max(ok_rows, key=lambda r: r["ctx"])
        print(f"→ {label} 的上下文上限（KV 仍在显存）：{best['ctx']:,}")
    else:
        print(f"→ {label} 没有任何一档达标")
    return 0


if __name__ == "__main__":
    sys.exit(main())
