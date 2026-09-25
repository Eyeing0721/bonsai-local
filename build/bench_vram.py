#!/usr/bin/env python3
"""实测不同上下文长度真实吃多少显存，以及能不能开到模型的训练上限。

为什么不用引擎自报的账单：这个 fork 在 verbosity 3 下不打 "KV self size" 那些行，
而 nvidia-smi 读的是驱动层面的真实占用 —— 包含 KV、GDN 的固定递归状态、compute
buffer、CUDA context 开销、以及显存碎片，这些正是"到底能开多长"的决定因素。

顺带回答"上下文放哪"：全部落在显存里。48 层 GDN 的状态是定长的，只有 16 层普通
注意力按 token 增长 —— 所以这个模型的上下文显存账单比同规模稠密模型便宜得多。

用法:
    python build/bench_vram.py
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

CASES = [
    (8192, None), (16384, None), (32768, None),
    (65536, None), (131072, None), (262144, None),
    (262144, "q8_0"),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def vram() -> tuple[float, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15,
        creationflags=CREATE_NO_WINDOW)
    used, total = out.stdout.strip().splitlines()[0].split(",")
    return float(used), float(total)


def main() -> int:
    if not MODEL.exists():
        print(f"模型不存在：{MODEL}")
        return 1
    base_used, total = vram()
    print(f"显卡总显存 {total:.0f} MiB；当前已用 {base_used:.0f} MiB\n")

    log_dir = ROOT / "build" / "bench-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ctx, kvt in CASES:
        port = free_port()
        log = log_dir / f"vram-{ctx}{'-' + kvt if kvt else ''}.log"
        fh = log.open("w", encoding="utf-8", errors="replace")
        cmd = [str(BIN / "llama-server.exe"), "-m", str(MODEL),
               "--host", "127.0.0.1", "--port", str(port),
               "-c", str(ctx), "-ngl", "99", "-fa", "on", "-np", "1",
               "--jinja", "--no-warmup", "--no-webui", "--alias", "bonsai"]
        if kvt:
            cmd += ["--cache-type-k", kvt, "--cache-type-v", kvt]
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])
        label = f"{ctx}" + (f" (KV {kvt})" if kvt else "")
        print(f"上下文 {label:<16}", end="", flush=True)
        proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh,
                                stderr=subprocess.STDOUT, env=env,
                                creationflags=CREATE_NO_WINDOW)
        peak = 0.0
        ok = False
        try:
            end = time.time() + 300
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
                time.sleep(0.5)
            if ok:
                time.sleep(2.5)                                 # 等显存稳定
                peak = vram()[0]
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            fh.close()
            time.sleep(3)
        if ok:
            rows.append((label, peak - base_used, peak))
            print(f"  启动成功  占用 {peak:.0f} MiB（比基线多 {peak - base_used:.0f}）")
        else:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
            rows.append((label, None, None))
            print(f"  启动失败  {' | '.join(x.strip()[:80] for x in tail)}")

    print("\n" + "=" * 74)
    print(f"{'上下文':<18}{'比基线多(MiB)':>15}{'总占用(MiB)':>14}{'余量(MiB)':>12}")
    for label, delta, peak in rows:
        if delta is None:
            print(f"{label:<18}{'— 没起来':>15}")
        else:
            print(f"{label:<18}{delta:>15.0f}{peak:>14.0f}{total - peak:>12.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
