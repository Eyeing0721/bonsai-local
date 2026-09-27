#!/usr/bin/env python3
"""实测草稿模型投机解码：能不能覆盖 n-gram 覆盖不到的通用问答。

n-gram 自投机只在"输出与上下文逐字重叠"时有效（复述、续写、改代码），
对普通问答一点没快。草稿模型是另一条路：用一个同族的小模型先猜，主模型
一次前向批量验证。它不依赖文本重复，理论上对任何负载都有效 —— 前提是
草稿的分布和主模型足够接近（接受率高）。

这里用 Qwen3.8-4B-Distill（同族、tokenizer 兼容）当草稿，Q4_K_M 2.66 GB。

要看的：
  接受率      —— 低于 ~50% 就没什么赚头
  三种负载     —— 自由型能不能也快起来（这才是通用加速）
  显存成本     —— 草稿要占多少
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
DRAFT = Path(r"E:\models\bonsai\draft\Qwen3.8-4B-Distill-Q4_K_M.gguf")
BIN = Path(r"E:\src\llama-prism\build-cuda-multi\bin")
CUDA = Path(r"E:\cuda\bin")
LOG_DIR = ROOT / "build" / "bench-logs"

MATERIAL = (
    "《银杏计划第七版技术备忘》\n"
    "1. 发布流程代号为「银杏 B7」，每次发布前必须运行 build/make_release.py。\n"
    "2. 推理引擎采用多架构 CUDA 构建，覆盖 sm_75 到 sm_120。\n"
    "3. 向量模型选用 Qwen3-Embedding-0.6B，权重量化到 Q8_0。\n"
)

PROMPTS = [
    ("自由型", "写一段大约 300 字的散文，描写秋天清晨的湖面，用词平实、不要重复。"),
    ("问答型", MATERIAL + "\n根据上面备忘录回答：向量模型是哪个？为什么要跑 make_release？"),
    ("复述型", MATERIAL + "\n把上面备忘录三条一字不改地列出来。"),
    ("推理型", "一个农夫要带狼、羊、白菜过河，船每次只能带一样东西。"
               "狼和羊单独在一起狼会吃羊，羊和白菜单独在一起羊会吃白菜。给出完整步骤。"),
]

CASES = [
    ("基线（不投机）", []),
    ("ngram-simple", ["--spec-type", "ngram-simple"]),
    ("草稿 4B", ["--spec-type", "draft-simple", "--spec-draft-model", str(DRAFT),
                 "--spec-draft-ngl", "99", "--spec-draft-n-max", "8"]),
    ("草稿 4B + ngram", ["--spec-type", "draft-simple,ngram-simple",
                         "--spec-draft-model", str(DRAFT),
                         "--spec-draft-ngl", "99", "--spec-draft-n-max", "8"]),
]

EVAL_RE = re.compile(
    r"(?<!prompt )eval time =\s*[\d.]+ ms /\s*(\d+) tokens? \([^,]+,\s*"
    r"([\d.]+) tokens per second\)")
ACCEPT_RE = re.compile(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)"
                       r", mean len = ([\d.]+)")


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


def api(port: int, path: str, payload=None, timeout=900.0):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else None


def read_stats(text: str) -> dict:
    out: dict = {}
    best = None
    for m in EVAL_RE.finditer(text):
        toks, tps = int(m.group(1)), float(m.group(2))
        if toks >= 10 and (best is None or toks > best[0]):
            best = (toks, tps)
    if best:
        out["tps"], out["gen_tokens"] = best[1], best[0]
    acc = ACCEPT_RE.search(text)
    if acc:
        out["accept"] = float(acc.group(1))
        out["draft_len"] = float(acc.group(4))
    return out


def run(case: str, extra: list[str]) -> dict:
    port = free_port()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"draft-{case.replace(' ', '_').replace('（', '').replace('）', '')}.log"
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(BIN / "llama-server.exe"), "-m", str(MODEL),
           "--host", "127.0.0.1", "--port", str(port),
           "-c", "16384", "-ngl", "99", "-fa", "on", "-np", "1",
           "--jinja", "--no-warmup", "--no-webui", "--alias", "bonsai"] + extra
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])
    base_vram = vram()
    proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh,
                            stderr=subprocess.STDOUT, env=env,
                            creationflags=CREATE_NO_WINDOW)
    out: dict = {"case": case, "results": {}}
    try:
        end = time.time() + 300
        while time.time() < end:
            if proc.poll() is not None:
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-4:]
                raise RuntimeError(f"引擎退出 {proc.returncode}: " + " | ".join(tail))
            try:
                if api(port, "/health", timeout=3).get("status") == "ok":
                    break
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("就绪超时")
        time.sleep(2)
        out["vram_mib"] = vram() - base_vram

        for label, prompt in PROMPTS:
            fh.flush()
            before = log.stat().st_size
            t0 = time.time()
            r = api(port, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 260, "temperature": 0.0, "seed": 3,
                "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}})
            wall = time.time() - t0
            fh.flush()
            with log.open("r", encoding="utf-8", errors="replace") as rf:
                rf.seek(before)
                chunk = rf.read()
            sp = read_stats(chunk)
            toks = (r.get("usage") or {}).get("completion_tokens") or 0
            out["results"][label] = {
                "tps": sp.get("tps"), "wall_tps": toks / max(wall, 1e-6),
                "accept": sp.get("accept"), "draft_len": sp.get("draft_len"),
                "tokens": toks, "wall": wall,
            }
    except Exception as e:                                      # noqa: BLE001
        out["error"] = str(e)
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
    if not DRAFT.exists():
        print(f"草稿模型不存在：{DRAFT}")
        return 1
    rows = []
    for name, extra in CASES:
        print(f"跑 {name} …", flush=True)
        r = run(name, extra)
        rows.append(r)
        if "error" in r:
            print(f"   失败：{r['error'][:300]}")

    labels = [p[0] for p in PROMPTS]
    print("\n" + "=" * 104)
    print(f"{'方案':<22}" + "".join(f"{l:>13}" for l in labels)
          + f"{'显存':>10}{'接受率':>9}{'起草长':>8}")
    print("=" * 104)
    base = next((r for r in rows if r["case"].startswith("基线")), {})
    for r in rows:
        res = r.get("results") or {}
        cells = ""
        for l in labels:
            v = (res.get(l) or {}).get("tps")
            cells += f"{v:>13.1f}" if v else f"{'—':>13}"
        acc = (res.get("自由型") or {}).get("accept")
        dl = (res.get("自由型") or {}).get("draft_len")
        vr = r.get("vram_mib")
        print(f"{r['case']:<22}{cells}"
              f"{(f'{vr:.0f}M' if vr else '—'):>10}"
              f"{(f'{acc:.1%}' if acc else '—'):>9}"
              f"{(f'{dl:.1f}' if dl else '—'):>8}")
    print("\n单位 tok/s。显存是相对基线的增量（草稿模型常驻显存的开销）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
