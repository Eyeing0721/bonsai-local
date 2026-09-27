#!/usr/bin/env python3
"""同一个探针跑多个模型文件，比能力也比速度。

用途：回答"这个设备上到底该用哪个量化"。三值 1.75 bit 只占 5.5 GiB，代价是
信息量被压得很狠；16 GB 卡其实放得下两三倍的量化，只是会慢一些。到底值不值，
得两边都量：能力（中性问答题的考点命中率）和速度（prefill / decode tok/s）。

每个模型文件必须单独起一次引擎 —— llama.cpp 一个进程只能服务一个权重。

用法:
    python build/compare_models.py \
        --model "三值=E:\\models\\bonsai\\Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
        --model "UD-Q3_K_XL=E:\\models\\bonsai\\alts\\Qwen3.8-27B-UD-Q3_K_XL.gguf" \
        --prompts build/dev-prompts/quality-probe.txt \
        --json-out devdata/compare-models.json
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

EVAL_RE = re.compile(
    r"(?<!prompt )eval time =\s*[\d.]+ ms /\s+(\d+) tokens? \([^,]+,\s*"
    r"([\d.]+) tokens per second\)")
PROMPT_RE = re.compile(
    r"prompt eval time =\s*[\d.]+ ms /\s+(\d+) tokens? \([^,]+,\s*"
    r"([\d.]+) tokens per second\)")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def api(port: int, path: str, payload=None, timeout=1200.0):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else None


def stats(text: str) -> dict:
    out: dict = {}
    worst = None
    for m in EVAL_RE.finditer(text):
        toks, tps = int(m.group(1)), float(m.group(2))
        if toks >= 10:
            if worst is None or tps < worst:
                worst = tps
    if worst:
        out["decode_tps"] = worst
    pf = None
    for m in PROMPT_RE.finditer(text):
        n, tps = int(m.group(1)), float(m.group(2))
        if n >= 100 and (pf is None or n > pf[0]):
            pf = (n, tps)
    if pf:
        out["prefill_tps"] = pf[1]
    return out


def run_model(label: str, path: Path, prompts: list[str], ctx: int,
              lora: Path | None, n_predict: int,
              engine_args: list[str] | None = None) -> dict:
    port = free_port()
    log_dir = ROOT / "build" / "bench-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"cmpmodel-{label.replace(' ', '_')}.log"
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(BIN / "llama-server.exe"), "-m", str(path),
           "--host", "127.0.0.1", "--port", str(port),
           "-c", str(ctx), "-ngl", "99", "-fa", "on", "-np", "1",
           "--jinja", "--no-warmup", "--no-webui", "--alias", "m"]
    if lora:
        cmd += ["--lora", str(lora)]
    cmd += list(engine_args or [])
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])
    print(f"\n{'=' * 78}\n── {label} ──  {path.name}  ({path.stat().st_size / 2**30:.2f} GiB)")
    out: dict = {"label": label, "model": str(path),
                 "bytes": path.stat().st_size, "answers": []}
    proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh,
                            stderr=subprocess.STDOUT, env=env,
                            creationflags=CREATE_NO_WINDOW)
    try:
        end = time.time() + 400
        while time.time() < end:
            if proc.poll() is not None:
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-4:]
                raise RuntimeError(f"引擎退出 {proc.returncode}: " + " | ".join(tail))
            try:
                if api(port, "/health", timeout=3).get("status") == "ok":
                    break
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(0.6)
        else:
            raise RuntimeError("就绪超时")

        # 统计整个会话的 prefill / decode
        for i, p in enumerate(prompts):
            fh.flush()
            before = log.stat().st_size
            t0 = time.time()
            r = api(port, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": p}],
                "max_tokens": n_predict, "temperature": 0.2, "seed": 11,
                "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}})
            wall = time.time() - t0
            msg = (r.get("choices") or [{}])[0].get("message", {})
            text = msg.get("content") or ""
            u = r.get("usage") or {}
            fh.flush()
            with log.open("r", encoding="utf-8", errors="replace") as rf:
                rf.seek(before)
                st = stats(rf.read())
            out["answers"].append({
                "index": i, "prompt": p, "answer": text,
                "completion_tokens": u.get("completion_tokens"),
                "wall": wall, **st})
            print(f"   [{i + 1}/{len(prompts)}] {u.get('completion_tokens')} tok "
                  f"{wall:5.1f}s  decode={st.get('decode_tps', 0):5.1f}  "
                  f"{text[:44].replace(chr(10), ' ')}")
    except Exception as e:                                      # noqa: BLE001
        out["error"] = str(e)
        print(f"   失败：{e}")
    finally:
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
    ap.add_argument("--model", action="append", required=True, help="标签=路径")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--only", default="", help="只跑标签里包含这个子串的模型")
    ap.add_argument("--engine-arg", action="append", default=[],
                    help="额外传给 llama-server 的参数，可重复")
    args = ap.parse_args()

    prompts = [b.strip() for b in
               Path(args.prompts).read_text(encoding="utf-8").split("\n---\n")
               if b.strip()]
    lora = Path(args.lora) if args.lora else None

    results = []
    for spec in args.model:
        label, _, path = spec.partition("=")
        if args.only and args.only not in label:
            continue
        p = Path(path)
        if not p.exists():
            print(f"跳过 {label}：文件不存在 {p}")
            continue
        results.append(run_model(label, p, prompts, args.ctx, lora,
                                 args.max_tokens, args.engine_arg))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "prompts": prompts,
            "answers": [{"arm": r["label"], "prompt_index": a["index"],
                         "prompt": a["prompt"], "answer": a["answer"],
                         "finish_reason": "stop"}
                        for r in results for a in r.get("answers", [])],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n回答已写入 {args.json_out}")

    print("\n" + "=" * 78)
    print(f"{'模型':<22}{'大小GiB':>9}{'decode':>9}{'prefill':>9}{'总耗时':>9}")
    for r in results:
        if "error" in r:
            print(f"{r['label']:<22}{'— 失败':>9}")
            continue
        ds = [a.get("decode_tps") for a in r["answers"] if a.get("decode_tps")]
        ps = [a.get("prefill_tps") for a in r["answers"] if a.get("prefill_tps")]
        tot = sum(a["wall"] for a in r["answers"])
        print(f"{r['label']:<22}{r['bytes'] / 2**30:>9.2f}"
              f"{(sum(ds) / len(ds) if ds else 0):>9.1f}"
              f"{(sum(ps) / len(ps) if ps else 0):>9.1f}{tot:>8.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
