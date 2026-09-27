#!/usr/bin/env python3
"""新编的 NInfer v3 引擎：探上下文上限 + 测投机解码。

引擎在首次遇到未内置的卡时会自己标定一遍（20-40 秒），我们这张 4060 Ti 不在
作者内置的 3090/4090/5090 之列，所以第一次会看到 "calibrating routes"。

用 Python 而不是 PowerShell 跑：NInfer 对请求体/命令行里的非 ASCII 很敏感，
PowerShell 5.1 会把中文按 GBK 传进去，上一次就因为这个把提示词搞成了乱码。

用法:
    python build/test_ninfer_v3.py --ctx 131072,262144,393216
    python build/test_ninfer_v3.py --ctx 262144 --spec dflash2 --drafts 5
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
APPS = Path(r"E:\build\ninfer-all\build-ninja\apps")
MODEL = Path(r"E:\models\bonsai\ninfer\Ternary-Bonsai-2-27B-ninfer-v3.ninfer")
PROMPT = "用一句话解释什么是 KV cache，然后用 Python 写一个判断回文串的函数。"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def post(port: int, path: str, body: dict, timeout: float = 900.0):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def vram() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=15)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def one(ctx: int, kv: str, spec: str, drafts: int, label: str) -> dict:
    port = free_port()
    logdir = ROOT / "build" / "bench-logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"ninfer3-{label}.log"
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(APPS / "nferve.exe") if False else str(APPS / "ninfer-serve.exe"),
           str(MODEL), "--host", "127.0.0.1", "--port", str(port),
           "--model-id", "m", "--max-context", str(ctx), "--kv-capacity", str(ctx),
           "--kv-dtype", kv, "--max-concurrency", "1", "--no-thinking",
           "--host-kv-mib", "1024", "--host-state-slots", "2",
           "--device-state-slots", "1"]
    if spec:
        cmd += ["--spec", spec]
        if drafts:
            cmd += ["--draft-tokens", str(drafts)]
    env = dict(os.environ)
    env["PATH"] = f"{APPS};E:\\cuda\\bin;" + env.get("PATH", "")
    print(f"\n{'=' * 74}\n── {label}  ctx={ctx:,} kv={kv}"
          + (f" spec={spec} drafts={drafts}" if spec else " 无投机"))
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(APPS), stdout=fh, stderr=subprocess.STDOUT,
                            env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out: dict = {"label": label, "ctx": ctx, "kv": kv, "spec": spec, "drafts": drafts}
    try:
        while time.time() - t0 < 300:
            if proc.poll() is not None:
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
                raise RuntimeError("引擎退出:\n      " + "\n      ".join(x.strip()[:140] for x in tail))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=4) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("就绪超时")
        out["load_s"] = round(time.time() - t0, 1)
        out["vram_idle"] = vram()
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        for ln in lines:
            if "capacity" in ln or "calibrating" in ln or "device profile" in ln:
                print("   " + ln.strip()[:150])
        print(f"   就绪 {out['load_s']}s   显存 {out['vram_idle']} MiB")

        r = post(port, "/v1/chat/completions", {
            "model": "m", "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 220, "temperature": 0.0, "seed": 3, "stream": False})
        u = r.get("usage") or {}
        tm = r.get("timings") or {}
        txt = ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        out["completion_tokens"] = u.get("completion_tokens")
        out["prefill_tps"] = round(tm.get("prompt_per_second", 0), 1)
        out["decode_tps"] = round(tm.get("predicted_per_second", 0), 1)
        out["answer_head"] = txt[:110].replace("\n", " ")
        print(f"   生成 {u.get('completion_tokens')} token   "
              f"prefill {out['prefill_tps']:.0f} tok/s   decode {out['decode_tps']:.1f} tok/s")
        print(f"   {out['answer_head'][:100]}")
    except Exception as e:                                      # noqa: BLE001
        out["error"] = str(e)
        print(f"   ✗ {e}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        fh.close()
        time.sleep(3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="131072")
    ap.add_argument("--kv", default="rk4v4")
    ap.add_argument("--spec", default="")
    ap.add_argument("--drafts", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if not (APPS / "ninfer-serve.exe").exists():
        print(f"引擎不存在：{APPS / 'ninfer-serve.exe'}")
        return 1
    if not MODEL.exists():
        print(f"模型不存在：{MODEL}")
        return 1
    if any(ord(c) > 127 for c in str(MODEL)):
        print("模型路径含非 ASCII，NInfer 会拒绝")
        return 1

    results = []
    for c in [int(x) for x in args.ctx.split(",")]:
        lbl = f"{c//1024}K" + (f"-{args.spec}" if args.spec else "")
        results.append(one(c, args.kv, args.spec, args.drafts, lbl))

    print(f"\n{'=' * 74}\n{'标签':<16}{'ctx':>10}{'显存MiB':>10}{'prefill':>9}{'decode':>9}")
    for r in results:
        if r.get("error"):
            print(f"{r['label']:<16}{r['ctx']:>10,}   FAILED")
        else:
            print(f"{r['label']:<16}{r['ctx']:>10,}{r.get('vram_idle',0):>10}"
                  f"{r.get('prefill_tps',0):>9.0f}{r.get('decode_tps',0):>9.1f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        print(f"\n写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
