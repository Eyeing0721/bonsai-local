#!/usr/bin/env python3
"""实测思维链强度（none / minimal / low / medium / high→xhigh）的代价。

底座的 chat 模板默认 effort = **xhigh**，而 OpenAI 的标准取值里根本没有 xhigh。
所以「不传任何参数时它到底多能想」这件事，得量出来才说得清：想得多不多、多花多少
token、多等多久、答案有没有变好。

用狼羊白菜这道题当标尺 —— 它有唯一正确答案（7 步），想不想得清楚看得出来。

用法:
    python build/bench_effort.py
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

QUESTION = ("一个农夫要把狼、羊、白菜运过河。船每次只能载农夫和一样东西。"
            "狼和羊单独在一起狼会吃羊，羊和白菜单独在一起羊会吃白菜。"
            "给出完整步骤，每步说明带什么、留下什么。")
# 这里直连引擎，所以只能用它模板认识的取值。注意 OpenAI 标准的 minimal/high
# 模板**不认**（会 raise_exception 返回 500）—— 那层映射在应用的代理里做，
# 由 build/test_openai_api.py 的 [3c] 覆盖。
EFFORTS = ["none", "low", "medium", "xhigh"]


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


def main() -> int:
    port = free_port()
    log = ROOT / "build" / "bench-logs" / "effort.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(BIN / "llama-server.exe"), "-m", str(MODEL),
           "--host", "127.0.0.1", "--port", str(port),
           "-c", "16384", "-ngl", "99", "-fa", "on", "-np", "1",
           "--jinja", "--no-warmup", "--no-webui", "--alias", "bonsai"]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])
    print("启动引擎 …")
    proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh,
                            stderr=subprocess.STDOUT, env=env,
                            creationflags=CREATE_NO_WINDOW)
    rows = []
    try:
        end = time.time() + 300
        while time.time() < end:
            if proc.poll() is not None:
                raise RuntimeError(f"引擎退出 {proc.returncode}")
            try:
                if api(port, "/health", timeout=4).get("status") == "ok":
                    break
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(0.6)
        print("就绪\n")

        print(f"{'强度':<10}{'思维字数':>9}{'正文字数':>9}{'输出token':>10}"
              f"{'耗时':>8}{'tok/s':>8}   答案")
        for eff in EFFORTS:
            body = {"messages": [{"role": "user", "content": QUESTION}],
                    "max_tokens": 1600, "temperature": 0.2, "seed": 7,
                    "reasoning_effort": eff}
            t0 = time.time()
            r = api(port, "/v1/chat/completions", body)
            wall = time.time() - t0
            msg = (r.get("choices") or [{}])[0].get("message", {})
            think = msg.get("reasoning_content") or ""
            out = msg.get("content") or ""
            u = r.get("usage") or {}
            ntok = u.get("completion_tokens") or 0
            speed = ntok / max(wall, 1e-6)
            rows.append({"effort": eff, "think": len(think), "out": len(out),
                         "tokens": ntok, "wall": wall})
            # 粗略判对：经典解要么先带羊，要么先把羊留下
            ok = ("羊" in out[:80]) or ("羊" in out[:120] and "先" in out[:120])
            print(f"{eff:<10}{len(think):>9}{len(out):>9}{ntok:>10}"
                  f"{wall:>7.1f}s{speed:>8.1f}   {'看着像正确解' if ok else '—'}")
            print(f"           正文开头：{out[:70].replace(chr(10), ' ')}")

        print("\n" + "=" * 74)
        base = next((r for r in rows if r["effort"] == "none"), None)
        if base:
            print(f"以 none 为基准（{base['tokens']} token / {base['wall']:.1f}s）")
            for r in rows:
                if r["effort"] == "none":
                    continue
                print(f"  {r['effort']:<9} token ×{r['tokens'] / max(base['tokens'], 1):.2f}"
                      f"   耗时 ×{r['wall'] / max(base['wall'], 1e-6):.2f}"
                      f"   思维 {r['think']} 字")
    except Exception as e:                                      # noqa: BLE001
        print(f"失败：{e}")
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
