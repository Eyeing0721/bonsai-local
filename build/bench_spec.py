#!/usr/bin/env python3
"""实测自投机解码（n-gram speculative）能不能提速。

背景：单序列生成时，每出 1 个 token 必须把 5.54 GiB 权重整个读一遍，所以
RTX 4060 Ti（288 GB/s）上纯自回归的理论上限是 48.4 tok/s。要超过它只有一条路：
一次前向验证多个 token —— 投机解码。

这个 fork 支持五种**不需要草稿模型**的 n-gram 自投机。它的原理是拿已有的上下文
去匹配 n-gram，猜接下来几个 token，然后让主模型一次前向全部验证。猜对了就白赚。

对知识库场景特别对症：模型经常大段复述检索到的原文，那种文本在上下文里已经
出现过，n-gram 命中率会很高。所以这里分两种负载测：
  复述型  —— 要求从给定材料里摘出原话（预期有增益）
  自由型  —— 开放式写作（预期没什么增益）
如果一个方法在自由型上有收益，那说明它真的在省带宽；如果只在复述型上有收益，
那就说明它只是在"抄上下文"。

用法:
    python build/bench_spec.py
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
LOG_DIR = ROOT / "build" / "bench-logs"

MATERIAL = (
    "《银杏计划第七版技术备忘》\n"
    "1. 发布流程代号为「银杏 B7」，每次发布前必须运行 build/make_release.py。\n"
    "2. 推理引擎采用多架构 CUDA 构建，覆盖 sm_75 到 sm_120。\n"
    "3. 向量模型选用 Qwen3-Embedding-0.6B，权重量化到 Q8_0。\n"
    "4. 知识库的检索走 BM25 与向量两路，用倒数排名融合。\n"
    "5. 前缀缓存按最长公共前缀复用，资料插在最后一轮之前。\n"
    "6. 拒绝回答适配器只在「默认」以外的风格预设里生效。\n"
    "7. 引擎包自检会在解压后真的启动一次 llama-server。\n"
)

PROMPTS = [
    ("复述型",
     MATERIAL + "\n请把上面备忘录里的七条**一字不改**地列出来。"),
    ("半复述型",
     MATERIAL + "\n根据上面备忘录回答：向量模型选的是哪个？为什么资料要插在最后一轮之前？"),
    ("自由型",
     "写一段大约 300 字的散文，描写秋天清晨的湖面，要求用词平实、不要重复。"),
]

CASES = [
    ("基线（不投机）", []),
    ("ngram-simple", ["--spec-type", "ngram-simple"]),
    ("ngram-map-k", ["--spec-type", "ngram-map-k"]),
    ("ngram-map-k4v", ["--spec-type", "ngram-map-k4v"]),
    ("ngram-mod", ["--spec-type", "ngram-mod"]),
    ("ngram-cache", ["--spec-type", "ngram-cache"]),
]

# 引擎自己报的周期性生成速度（只对长生成出现）
TG_RE = re.compile(r"n_gen =\s*(\d+), tg =\s*([\d.]+) t/s")
# 每次请求结束时都会打这一行。负向后视排除 prompt eval —— 两者只差一个前缀，
# 第一版正是栽在这里，把 335 tok/s 的 prefill 当成了生成速度。
EVAL_RE = re.compile(
    r"(?<!prompt )eval time =\s*[\d.]+ ms /\s*(\d+) tokens? \([^,]+,\s*"
    r"([\d.]+) tokens per second\)")
ACCEPT_RE = re.compile(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\),"
                       r" mean len = ([\d.]+)")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


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
    """从日志片段里取生成速度与投机接受率。

    优先用每次请求结束必打的 `eval time` 行；它比 `n_gen` 周期行可靠 ——
    短生成（1 秒内跑完）根本不会打周期行，第一版就是因为这个把 121 tok/s
    记成了 0。
    """
    out: dict = {}
    best = None
    for m in EVAL_RE.finditer(text):
        toks, tps = int(m.group(1)), float(m.group(2))
        if toks >= 10 and (best is None or toks > best[0]):
            best = (toks, tps)
    if best:
        out["tps"], out["gen_tokens"] = best[1], best[0]
    tg = None
    for m in TG_RE.finditer(text):
        n, v = int(m.group(1)), float(m.group(2))
        if tg is None or n > tg[0]:
            tg = (n, v)
    if tg:
        out["tg_tps"] = tg[1]
    acc = ACCEPT_RE.search(text)
    if acc:
        out["accept"] = float(acc.group(1))
        out["draft_mean_len"] = float(acc.group(4))
    return out


def run(case: str, extra: list[str]) -> dict:
    port = free_port()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"spec-{case.replace(' ', '_').replace('（', '').replace('）', '')}.log"
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(BIN / "llama-server.exe"), "-m", str(MODEL),
           "--host", "127.0.0.1", "--port", str(port),
           "-c", "16384", "-ngl", "99", "-fa", "on", "-np", "1",
           "--jinja", "--no-warmup", "--no-webui", "--alias", "bonsai"] + extra
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])
    proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh,
                            stderr=subprocess.STDOUT, env=env,
                            creationflags=CREATE_NO_WINDOW)
    out: dict = {"case": case, "results": {}}
    try:
        end = time.time() + 240
        while time.time() < end:
            if proc.poll() is not None:
                raise RuntimeError(f"引擎退出 {proc.returncode}")
            try:
                if api(port, "/health", timeout=3).get("status") == "ok":
                    break
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("就绪超时")

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
            # 引擎没打速度行时用墙钟兜底，两个数都留着 —— 差太多就说明有别的开销
            wall_tps = toks / max(wall, 1e-6)
            out["results"][label] = {
                "tps": sp.get("tps"),
                "tg_tps": sp.get("tg_tps"),
                "wall_tps": wall_tps,
                "accept": sp.get("accept"),
                "draft_len": sp.get("draft_mean_len"),
                "total_tokens": toks, "wall": wall,
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
    rows = []
    for name, extra in CASES:
        print(f"跑 {name} …", flush=True)
        r = run(name, extra)
        rows.append(r)
        if "error" in r:
            print(f"   失败：{r['error']}")

    print("\n" + "=" * 96)
    labels = [p[0] for p in PROMPTS]
    print(f"{'方案':<20}" + "".join(f"{l + ' tok/s':>17}" for l in labels)
          + f"{'复述增益':>12}{'接受率':>9}{'起草长':>8}")
    print("=" * 96)
    base = next((r for r in rows if r["case"].startswith("基线")), None)
    base_re = ((base or {}).get("results", {}).get("复述型", {}) or {}).get("tps")
    for r in rows:
        res = r.get("results") or {}
        cells = ""
        for l in labels:
            v = (res.get(l) or {}).get("tps")
            cells += f"{v:>17.1f}" if v else f"{'—':>17}"
        re_v = (res.get("复述型") or {}).get("tps")
        gain = f"×{re_v / base_re:.2f}" if re_v and base_re else "—"
        acc = (res.get("复述型") or {}).get("accept")
        dlen = (res.get("复述型") or {}).get("draft_len")
        print(f"{r['case']:<20}{cells}{gain:>12}"
              f"{(f'{acc:.1%}' if acc else '—'):>9}"
              f"{(f'{dlen:.1f}' if dlen else '—'):>8}")
    print(f"\n基线复述 {base_re:.1f} tok/s；自由型列只作对照（预期无增益）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
