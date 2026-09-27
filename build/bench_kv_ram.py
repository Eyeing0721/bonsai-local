#!/usr/bin/env python3
"""实测 KV 放系统内存（--no-kv-offload）到底值不值。

理论账（见 build 里那套估算）：每个 token 的耗时 = 权重/显存带宽 + KV/PCIe带宽。
权重永远从显存走（20.6 ms），只有 KV 过 PCIe（4060 Ti 是 PCIe 4.0 x8，约 13 GB/s）。
所以短上下文几乎免费，长上下文崩得很快。

这个脚本把账算实：三种配置，都先把上下文填到 ~28000 token 再测生成速度。

  A  KV 全在显存，开 32K 窗口
  B  KV 全在内存（--no-kv-offload），开 32K 窗口
  C  KV 全在内存，开 128K 窗口  ← 这个窗口在显存里放不下，是内存方案真正的用处

用法:
    python build/bench_kv_ram.py
"""

from __future__ import annotations

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
MODEL = Path(r"E:\models\bonsai\Ternary-Bonsai-2-27B-PTQ1_0.gguf")
BIN = Path(r"E:\src\llama-prism\build-cuda-multi\bin")
CUDA = Path(r"E:\cuda\bin")

FILL_TOKENS = 28000
UNIT = "这是一段用来填充上下文的背景材料，内容本身没有意义。"
# 中文大约 1 字 1 token，所以按**字符数**凑长度，不是按单元个数 —— 第一版
# 用 FILL_TOKENS//6 把每个 25 字的单元当成 6 个 token，结果提示词撑到 11 万
# token，直接超出 32K 窗口吃到 HTTP 400。
UNITS = max(1, FILL_TOKENS // len(UNIT))

TAG_RE = re.compile(r"n_gen =\s*(\d+), tg =\s*([\d.]+) t/s")
EVAL_RE = re.compile(
    r"(?<!prompt )eval time =\s*[\d.]+ ms /\s+(\d+) tokens? \([^,]+,\s*"
    r"([\d.]+) tokens per second\)")
PROMPT_RE = re.compile(
    r"prompt eval time =\s*[\d.]+ ms /\s+(\d+) tokens? \([^,]+,\s*"
    r"([\d.]+) tokens per second\)")

CASES = [
    ("A KV在显存 · 窗口32K", 32768, []),
    ("B KV在内存 · 窗口32K", 32768, ["--no-kv-offload"]),
    ("C KV在内存 · 窗口128K", 131072, ["--no-kv-offload"]),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def api(port: int, path: str, payload=None, timeout=1800.0):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else None


def proc_ram(pid: int) -> float:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).WorkingSet64"],
        capture_output=True, text=True, timeout=25, creationflags=CREATE_NO_WINDOW)
    try:
        return float(out.stdout.strip()) / 2**30
    except ValueError:
        return 0.0


def main() -> int:
    filler = UNIT * UNITS
    print(f"填充约 {len(filler):,} 字（中文约 1 字 1 token，目标是 {FILL_TOKENS}）\n")
    rows = []
    for label, ctx, extra in CASES:
        port = free_port()
        log = ROOT / "build" / "bench-logs" / f"kvram-{label[0]}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("w", encoding="utf-8", errors="replace")
        cmd = [str(BIN / "llama-server.exe"), "-m", str(MODEL),
               "--host", "127.0.0.1", "--port", str(port),
               "-c", str(ctx), "-ngl", "99", "-fa", "on", "-np", "1",
               "--jinja", "--no-warmup", "--no-webui", "--alias", "m"] + extra
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])
        print(f"── {label} ──")
        proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh,
                                stderr=subprocess.STDOUT, env=env,
                                creationflags=CREATE_NO_WINDOW)
        row: dict = {"case": label, "ctx": ctx, "extra": extra}
        try:
            end = time.time() + 400
            while time.time() < end:
                if proc.poll() is not None:
                    raise RuntimeError(f"引擎退出 {proc.returncode}")
                try:
                    if api(port, "/health", timeout=3).get("status") == "ok":
                        break
                except Exception:                               # noqa: BLE001
                    pass
                time.sleep(0.6)
            else:
                raise RuntimeError("就绪超时")
            row["ram_gib"] = proc_ram(proc.pid)

            # 第一次请求：填充 + 生成，取 prefill 速度
            fh.flush()
            before = log.stat().st_size
            t0 = time.time()
            api(port, "/v1/chat/completions", {
                "messages": [{"role": "user",
                              "content": filler + "\n\n用一句话总结上面材料在讲什么。"}],
                "max_tokens": 40, "temperature": 0.0,
                "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}})
            row["fill_wall"] = time.time() - t0
            fh.flush()
            with log.open("r", encoding="utf-8", errors="replace") as rf:
                rf.seek(before)
                chunk = rf.read()
            for m in PROMPT_RE.finditer(chunk):
                n, tps = int(m.group(1)), float(m.group(2))
                if n >= 500:
                    row["prefill_tps"] = tps
                    row["prefill_tokens"] = n

            # 第二次请求：上下文已经装满了（命中缓存），只测生成
            fh.flush()
            before = log.stat().st_size
            t0 = time.time()
            api(port, "/v1/chat/completions", {
                "messages": [{"role": "user",
                              "content": filler + "\n\n用一句话总结上面材料在讲什么。"}],
                "max_tokens": 120, "temperature": 0.0,
                "cache_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False}})
            row["decode_wall"] = time.time() - t0
            fh.flush()
            with log.open("r", encoding="utf-8", errors="replace") as rf:
                rf.seek(before)
                chunk = rf.read()
            for m in EVAL_RE.finditer(chunk):
                n, tps = int(m.group(1)), float(m.group(2))
                if n >= 20:
                    row["decode_tps"] = tps
                    row["decode_tokens"] = n
            row["ram_after_gib"] = proc_ram(proc.pid)
        except Exception as e:                                  # noqa: BLE001
            row["error"] = str(e)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            fh.close()
            time.sleep(4)
        rows.append(row)
        if "error" in row:
            print(f"   失败：{row['error']}")
        else:
            print(f"   进程内存 {row.get('ram_gib', 0):5.2f} GiB → "
                  f"{row.get('ram_after_gib', 0):5.2f} GiB")
            print(f"   prefill {row.get('prefill_tps', 0):6.1f} tok/s "
                  f"（{row.get('prefill_tokens', 0)} token）")
            print(f"   decode  {row.get('decode_tps', 0):6.1f} tok/s "
                  f"（{row.get('decode_tokens', 0)} token）")

    print("\n" + "=" * 84)
    print(f"{'配置':<26}{'进程内存GiB':>13}{'prefill':>11}{'decode':>11}{'填充耗时':>11}")
    for r in rows:
        if "error" in r:
            print(f"{r['case']:<26}{'失败':>13}")
            continue
        print(f"{r['case']:<26}{r.get('ram_after_gib', 0):>13.2f}"
              f"{r.get('prefill_tps', 0):>11.1f}{r.get('decode_tps', 0):>11.1f}"
              f"{r.get('fill_wall', 0):>10.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
