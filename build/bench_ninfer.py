#!/usr/bin/env python3
"""测 NInfer 引擎 —— 和 llama.cpp 跑同一个模型、同一个量化，做引擎对照。

为什么这个对照最有价值：我们手上的三值 Bonsai 27B，llama.cpp 侧和 NInfer 侧
**是同一个模型、同一个 1.75 bpw 量化**，只是打包格式不同（.gguf vs .ninfer）。
所以这里量出来的差距**纯粹是引擎的差距**，不掺杂模型差异。

⚠ 2026-09-26 更正：上面这句只对**早期的 v2 `.ninfer`** 成立。作者判据.txt 附带的
v2 确实是从同一份 PTQ1_0 转的。但 **v3 不是** ——
`Ternary-Bonsai-2-27B-ninfer-v3.ninfer` 的 conversion.json 里 provenance 写明
ternary 源是 `Ternary-Bonsai-2-27B-PQ2_0.gguf`（6.71 GiB / 2.0 bpw），且额外带了
`mtp` + `dflash2` + `vision` 三个权重包。所以拿 v3 对比 llama.cpp 时，差距里
**同时包含引擎、量化精度、投机解码三件事**，不能当成纯引擎差距解读。

注意 NInfer 与 llama.cpp 的两处接口差异（作者判据.txt 里专门写了）：
  · 请求体**必须带 `model` 字段**，少了会被拒
  · 路径**必须纯 ASCII**（中文会报 invalid UTF-8）

用法:
    python build/bench_ninfer.py --model D:\\ninfer-fast\\model\\x.ninfer
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

ROOT = Path(__file__).resolve().parent.parent
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ENGINE = Path(r"D:\ninfer-fast\engine\ninfer-serve.exe")

PROMPTS = [
    ("自由型", "写一段大约 150 字的散文，描写春天傍晚的田野，用词平实、不要重复。"),
    ("问答型", "CVE-2018-8120 是哪个系统组件里的漏洞？属于哪一类？影响哪些 Windows 版本？"),
    ("推理型", "一个农夫要带狼、羊、白菜过河，船每次只能带一样东西。"
               "狼和羊单独在一起狼会吃羊，羊和白菜单独在一起羊会吃白菜。给出完整步骤。"),
]


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


def post(port: int, path: str, body: dict, timeout: float = 900.0):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--kv-dtype", default="fp8")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--spec", default="", help="如 mtp / dflash / dflash2")
    ap.add_argument("--draft-tokens", type=int, default=0)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--extra-line", default="",
                    help="一整行额外参数，按空格拆分（比反复写 --extra 好用）")
    ap.add_argument("--label", default="NInfer")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"模型不存在：{model}")
        return 1
    if any(ord(c) > 127 for c in str(model)):
        print(f"⚠ 路径含非 ASCII 字符，NInfer 会拒绝：{model}")
        return 1

    port = free_port()
    log = ROOT / "build" / "bench-logs" / "ninfer-run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(ENGINE), str(model), "--host", "127.0.0.1", "--port", str(port),
           "--model-id", "m", "--max-context", str(args.ctx),
           "--kv-capacity", str(args.ctx), "--kv-dtype", args.kv_dtype,
           "--max-concurrency", "1", "--no-thinking"]
    if args.spec:
        cmd += ["--spec", args.spec]
        if args.draft_tokens:
            cmd += ["--draft-tokens", str(args.draft_tokens)]
    cmd += args.extra
    if args.extra_line:
        cmd += args.extra_line.split()

    env = dict(os.environ)
    env["PATH"] = r"E:\cuda\bin;" + env.get("PATH", "")
    base_vram = vram()
    print(f"启动 {args.label}  ctx={args.ctx} kv={args.kv_dtype} "
          f"spec={args.spec or '无'}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                            creationflags=CREATE_NO_WINDOW)
    out: dict = {"label": args.label, "results": []}
    try:
        ready = False
        while time.time() - t0 < 600:
            if proc.poll() is not None:
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
                print("  引擎退出：")
                for x in tail:
                    print("    " + x.strip()[:140])
                return 1
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/v1/models", timeout=4) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(1.0)
        if not ready:
            print("  就绪超时")
            return 1
        load_s = time.time() - t0
        print(f"  就绪用时 {load_s:.1f}s   显存 {vram():.0f} MiB "
              f"（比基线多 {vram() - base_vram:.0f}）")
        # 打印容量行
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "capacity" in line or "weights ready" in line:
                print("    " + line.strip()[:150])

        for label, prompt in PROMPTS:
            t1 = time.time()
            r = post(port, "/v1/chat/completions", {
                "model": "m", "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens, "temperature": 0.0, "seed": 3,
                "stream": False})
            wall = time.time() - t1
            msg = (r.get("choices") or [{}])[0].get("message", {})
            u = r.get("usage") or {}
            n = u.get("completion_tokens") or 0
            ttft = r.get("timings", {}).get("prompt_ms", 0) / 1000.0
            out["results"].append({"label": label, "tokens": n, "wall": wall,
                                   "tps": n / max(wall, 1e-6),
                                   "usage": u, "timings": r.get("timings"),
                                   "text": (msg.get("content") or "")[:200]})
            extra = ""
            if r.get("timings"):
                tm = r["timings"]
                extra = (f"  prefill={tm.get('prompt_per_second', 0):.0f} t/s"
                         f"  decode={tm.get('predicted_per_second', 0):.1f} t/s")
            print(f"  [{label}] {n} tok / {wall:.1f}s = {n / max(wall, 1e-6):.1f} t/s"
                  f"{extra}")
            print(f"      {(msg.get('content') or '')[:90].replace(chr(10), ' ')}")
    except Exception as e:                                      # noqa: BLE001
        print(f"  失败：{type(e).__name__}: {e}")
        out["error"] = str(e)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
        fh.close()

    outpath = ROOT / "devdata" / f"ninfer-{args.label}.json"
    outpath.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果写入 {outpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
