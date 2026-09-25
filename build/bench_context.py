#!/usr/bin/env python3
"""实测：上下文放哪、能吃多长、输出速度、前缀缓存怎么命中。

为什么不能靠算：KV 是算得出来的，但"实际能开多长"取决于显存碎片、compute buffer、
以及那 48 层 GDN 的固定状态占多少 —— 只有把引擎真的跑起来，读它自己打的显存账单
才作数。

另外单独测一件事：我们给聊天请求注入的资料是插在**最前面**的。前缀缓存按最长公共
前缀工作，所以资料一变，整个前缀就废了。这对多轮对话意味着每一轮都要重新 prefill。
这里量一下插在前面和插在最后一轮之前的差别。

用法:
    python build/bench_context.py
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

FILLER = ("这是一段用来占位的背景材料，内容本身没有意义，只用来把提示词撑到指定长度。"
          "缓存命中与否要看前缀是否一致，所以重复的部分必须是逐字节相同的。")


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


def wait_ready(port: int, proc, timeout: float = 300.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"引擎退出，码 {proc.returncode}")
        try:
            if api(port, "/health", timeout=4).get("status") == "ok":
                return
        except Exception:                                       # noqa: BLE001
            pass
        time.sleep(0.6)
    raise RuntimeError("等待就绪超时")


MEM_PATTERNS = [
    (r"CUDA0 model buffer size\s*=\s*([\d.]+) MiB", "模型权重"),
    (r"CUDA0 KV buffer size\s*=\s*([\d.]+) MiB", "KV 缓存"),
    (r"CUDA0 compute buffer size\s*=\s*([\d.]+) MiB", "计算缓冲"),
    (r"CUDA0 RS buffer size\s*=\s*([\d.]+) MiB", "RS 缓冲"),
    (r"CUDA0 output buffer size\s*=\s*([\d.]+) MiB", "输出缓冲"),
    (r"CUDA0 model buffer size.*?=\s*([\d.]+)", "模型权重(备)"),
    (r"llama_context:.*KV self size\s*=\s*([\d.]+) MiB", "KV self(f16)"),
    (r"llama_kv_cache.*size\s*=\s*([\d.]+) MiB", "KV self"),
    (r"CUDA0 buffer size\s*=\s*([\d.]+) MiB", "CUDA0 合计"),
]


def read_mem(log: Path) -> list[tuple[str, float]]:
    text = log.read_text(encoding="utf-8", errors="replace")
    out, seen = [], set()
    for pat, name in MEM_PATTERNS:
        for m in re.finditer(pat, text):
            if name in seen:
                continue
            seen.add(name)
            try:
                out.append((name, float(m.group(1))))
            except ValueError:
                pass
    return out


def make_prompt(tokens_ish: int) -> str:
    """按字符数粗造一段提示。中文大约 1 字 1 token，够用了。"""
    unit = FILLER
    n = max(1, tokens_ish // len(unit))
    return (unit * n)


def ask(port: int, messages, max_tokens=60, effort=None, cache=True) -> dict:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "seed": 1, "cache_prompt": cache,
            "chat_template_kwargs": {"enable_thinking": False}}
    if effort:
        body["reasoning_effort"] = effort
    t0 = time.time()
    r = api(port, "/v1/chat/completions", body)
    wall = time.time() - t0
    u = r.get("usage") or {}
    return {"wall": wall, "prompt": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "out": u.get("completion_tokens"), "raw": r}


def run_at(ctx: int) -> dict:
    port = free_port()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"ctx-{ctx}.log"
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(BIN / "llama-server.exe"), "-m", str(MODEL),
           "--host", "127.0.0.1", "--port", str(port),
           "-c", str(ctx), "-ngl", "99", "-fa", "on", "-np", "1",
           "--jinja", "--no-warmup", "--no-webui", "--alias", "bonsai"]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])
    print(f"\n{'=' * 74}\n上下文 {ctx}  启动中 …")
    proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh, stderr=subprocess.STDOUT,
                            env=env, creationflags=CREATE_NO_WINDOW)
    result: dict = {"ctx": ctx}
    try:
        wait_ready(port, proc)
        result["mem"] = read_mem(log)
        for name, mib in result["mem"]:
            print(f"   {name:<16} {mib:9.1f} MiB")

        base = make_prompt(4000)
        q = "\n\n根据以上材料，用一句话说明这段文字的作用。"
        msgs = [{"role": "user", "content": base + q}]

        print("   速度与缓存：")
        first = ask(port, msgs, max_tokens=80)
        result["first"] = first
        print(f"     [1] 首次（冷启动 prefill）  prompt={first['prompt']} "
              f"cached={first['cached']}  {first['wall']:.2f}s")

        second = ask(port, msgs, max_tokens=80)
        result["second"] = second
        print(f"     [2] 同一请求再来一遍        prompt={second['prompt']} "
              f"cached={second['cached']}  {second['wall']:.2f}s")

        # 第三轮：前面加一段变化的内容（模拟多轮对话 + 资料注入在最前面）
        churn = "\n\n（第 2 轮新加入的材料）" + make_prompt(120)
        third = ask(port, [{"role": "user", "content": churn + base + q}],
                    max_tokens=80)
        result["churn_front"] = third
        print(f"     [3] 前面插 120 字新内容     prompt={third['prompt']} "
              f"cached={third['cached']}  {third['wall']:.2f}s")

        # 第四轮：变化的内容放在**最后**（缓存友好）
        fourth = ask(port, [{"role": "user", "content": base + churn + q}],
                     max_tokens=80)
        result["churn_back"] = fourth
        print(f"     [4] 把新内容放到后面       prompt={fourth['prompt']} "
              f"cached={fourth['cached']}  {fourth['wall']:.2f}s")

        # 纯解码速度：短提示、多输出
        dec = ask(port, [{"role": "user", "content": "从 1 数到 200，只写数字。"}],
                  max_tokens=400)
        result["decode"] = dec
        speed = (dec["out"] or 0) / max(dec["wall"], 1e-6)
        print(f"     [5] 纯解码 {dec['out']} token 用时 {dec['wall']:.2f}s "
              f"→ {speed:.1f} tok/s（含少量 prefill）")
        result["decode_tps"] = speed
    except Exception as e:                                      # noqa: BLE001
        print(f"   失败：{e}")
        result["error"] = str(e)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        fh.close()
        time.sleep(3)
    return result


def main() -> int:
    sizes = [int(x) for x in (sys.argv[1:] or ["8192", "32768", "131072"])]
    results = [run_at(c) for c in sizes]

    print("\n" + "=" * 74)
    print("汇总")
    print("=" * 74)
    print(f"{'上下文':>8} {'KV(MiB)':>9} {'显存合计(MiB)':>13} "
          f"{'冷 prefill':>11} {'命中 prefill':>13} {'解码 tok/s':>11}")
    for r in results:
        mem = dict(r.get("mem") or [])
        kv = mem.get("KV 缓存") or mem.get("KV self(f16)") or mem.get("KV self") or 0
        total = mem.get("CUDA0 合计") or sum(mem.values())
        f = r.get("first") or {}
        s = r.get("second") or {}
        print(f"{r['ctx']:>8} {kv:>9.1f} {total:>13.1f} "
              f"{f.get('wall', 0):>10.2f}s {s.get('wall', 0):>12.2f}s "
              f"{r.get('decode_tps', 0):>10.1f}")
        if "churn_front" in r and "churn_back" in r:
            a, b = r["churn_front"], r["churn_back"]
            print(f"         前缀变化插在前面 cached={a.get('cached')} "
                  f"({a.get('wall', 0):.2f}s)  |  放在后面 cached={b.get('cached')} "
                  f"({b.get('wall', 0):.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
