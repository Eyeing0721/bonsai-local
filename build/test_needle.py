#!/usr/bin/env python3
"""长上下文针测试：把码埋进长文档，看模型能不能找回来。

为什么必须做：显存能装下多少 token，和模型在那个深度还有没有记性，是两件完全
不同的事。「477K 能装」只是账算得过来；能不能用要实测。

方法（照 NInfer 作者的做法）：在一份长文档的 33% / 66% / 90% 三处各埋一个唯一
的码，然后要求模型按顺序把它们报回来。三个全中 = 那个深度还有效。

对照组很重要：如果**窗口内**（比如 200K，模型原生 256K）也找不回来，那说明是
测试方法或模型本身的问题，而不是"越界导致的退化"。

token 数用引擎自己的 /tokenize 数，不用字符数估算。

用法:
    python build/test_needle.py --model X.gguf --tokens 200000 --ctx 262144 --label 200K
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
BIN = Path(r"E:\src\llama-prism\build-cuda-multi\bin")
CUDA = Path(r"E:\cuda\bin")
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# 填充文本来源：真实、多样、非重复。构建日志是按体量最容易拿到的真实长文本。
FILLERS = [
    ROOT / "build" / "cuda-multi.log",
    ROOT / "build" / "vulkan.log",
]

MARKERS = [
    ("ALPHA", "K7Q2M"),
    ("BETA", "X4N9P"),
    ("GAMMA", "R3T8W"),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def api(port: int, path: str, payload=None, timeout=3600.0):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_filler(min_chars: int) -> str:
    """拼够 min_chars 的真实文本。"""
    parts: list[str] = []
    total = 0
    for p in FILLERS:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", errors="replace") as f:
            while total < min_chars:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                parts.append(chunk)
                total += len(chunk)
        if total >= min_chars:
            break
    return "".join(parts)


def build_doc(port: int, target_tokens: int) -> tuple[str, int]:
    """按目标 token 数裁出一份文档（用引擎的 tokenizer 精确计数）。"""
    # 先按 ~3.5 字符/token 估一个量，再多拿 20% 余量
    raw = load_filler(int(target_tokens * 3.5 * 1.2))
    if not raw:
        raise RuntimeError("找不到填充文本")
    ids = api(port, "/tokenize", {"content": raw})["tokens"]
    print(f"   候选文本 {len(raw):,} 字符 -> {len(ids):,} token")
    if len(ids) < target_tokens:
        raise RuntimeError(f"填充文本不够：只有 {len(ids):,} token，需要 {target_tokens:,}")
    # 用 token 边界反推字符位置太麻烦，改成二分裁字符，让 token 数贴近目标
    lo, hi = 0, len(raw)
    while hi - lo > 512:
        mid = (lo + hi) // 2
        n = len(api(port, "/tokenize", {"content": raw[:mid]})["tokens"])
        if n > target_tokens:
            hi = mid
        else:
            lo = mid
    text = raw[:lo]
    n = len(api(port, "/tokenize", {"content": text})["tokens"])
    return text, n


def plant(text: str, n_tokens: int) -> tuple[str, dict]:
    """在 33% / 66% / 90% 处埋码。位置用字符比例近似 token 比例。"""
    out = text
    positions = {}
    for frac, (name, code) in zip((0.90, 0.66, 0.33), MARKERS):
        # 从后往前插，避免偏移互相影响
        idx = int(len(out) * frac)
        # 挪到最近的换行，别把词切断
        nl = out.find("\n", idx)
        if nl < 0:
            nl = idx
        line = f"\n\n[记录 {name}] 这个位置的校验码是 {code}，请记住。\n\n"
        out = out[:nl] + line + out[nl:]
        positions[name] = {"code": code, "pct": int(frac * 100)}
    return out, positions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, required=True, help="文档目标 token 数")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--kv", default="q4_0", help="KV 量化类型")
    ap.add_argument("--ropeg", default="", help="如 yarn，留空=不缩放")
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"模型不存在：{model}")
        return 1
    label = args.label or f"{args.tokens // 1000}K"

    port = free_port()
    log = ROOT / "build" / "bench-logs" / f"needle-{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(BIN / "llama-server.exe"), "-m", str(model),
           "--host", "127.0.0.1", "--port", str(port),
           "-c", str(args.ctx), "-ngl", "99", "-fa", "on", "-np", "1",
           "--jinja", "--no-warmup", "--no-webui", "--alias", "m",
           "--cache-type-k", args.kv, "--cache-type-v", args.kv]
    if args.ropeg:
        cmd += ["--rope-scaling", args.ropeg]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])

    print(f"{'=' * 72}\n针测试 {label}: 目标 {args.tokens:,} token, ctx={args.ctx:,}, "
          f"KV={args.kv}" + (f", rope={args.ropeg}" if args.ropeg else ", rope=不缩放"))
    proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh, stderr=subprocess.STDOUT,
                            env=env, creationflags=CREATE_NO_WINDOW)
    out: dict = {"label": label, "target_tokens": args.tokens, "ctx": args.ctx,
                 "kv": args.kv, "rope": args.ropeg or "none"}
    try:
        t0 = time.time()
        while time.time() - t0 < 600:
            if proc.poll() is not None:
                raise RuntimeError(f"引擎退出 {proc.returncode}")
            try:
                if api(port, "/health", timeout=3).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("就绪超时")
        print(f"   引擎就绪 {time.time()-t0:.0f}s")

        print("   裁文档...")
        text, n = build_doc(port, args.tokens)
        doc, positions = plant(text, n)
        out["doc_tokens"] = n
        out["positions"] = positions
        print(f"   文档 {n:,} token，已埋码: "
              + ", ".join(f"{k}@{v['pct']}%" for k, v in positions.items()))

        question = ("\n\n---\n以上是一份长文档。文档里在不同位置埋了三个带名字的校验码"
                    "（ALPHA / BETA / GAMMA）。请只回答这三个码，格式为：\n"
                    "ALPHA=xxxxx\nBETA=xxxxx\nGAMMA=xxxxx\n"
                    "如果某个码你没找到，写 未找到。不要解释。")
        prompt = doc + question
        total = len(api(port, "/tokenize", {"content": prompt})["tokens"])
        out["prompt_tokens"] = total
        print(f"   最终提示词 {total:,} token，开始 prefill（预计 "
              f"{total/400/60:.1f} 分钟 @400 tok/s）...")

        tf = time.time()
        r = api(port, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200, "temperature": 0.0, "seed": 1,
            # 必须显式关掉思考：这个模型默认开思考，200 个 token 会被推理过程吃光，
            # content 返回空字符串 —— 第一次做这个测试就是因为这个，200K 对照组拿到
            # 了一个空回答，看起来像"模型找不到码"，其实是根本没输出。
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_prompt": False})
        wall = time.time() - tf
        msg = (r.get("choices") or [{}])[0].get("message", {})
        answer = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        out["reasoning_chars"] = len(reasoning)
        tm = r.get("timings") or {}
        out["answer"] = answer
        out["wall_s"] = wall
        out["prompt_tokens_used"] = (r.get("usage") or {}).get("prompt_tokens")
        out["prefill_tps"] = tm.get("prompt_per_second")
        out["decode_tps"] = tm.get("predicted_per_second")

        hit = {k: (v["code"] in answer) for k, v in positions.items()}
        out["hits"] = hit
        out["recall"] = sum(hit.values())
        print(f"   prefill+生成 用时 {wall/60:.1f} 分钟"
              + (f"  prefill {tm.get('prompt_per_second', 0):.0f} tok/s" if tm else ""))
        print(f"   回答: {answer[:120].replace(chr(10), ' | ')}")
        print(f"   命中: " + ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in hit.items())
              + f"   -> {out['recall']}/3")
    except Exception as e:                                      # noqa: BLE001
        out["error"] = str(e)
        print(f"   失败：{type(e).__name__}: {e}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                proc.kill()
        fh.close()
        time.sleep(2)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
