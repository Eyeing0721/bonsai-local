#!/usr/bin/env python3
"""验证 n-gram 自投机在产品里真的生效（走完整应用，不是直连引擎）。

前面所有投机测试都是直连 llama-server 做的。接进应用之后要确认三件事：
  1. 引擎真的带上了 --spec-type
  2. 复述型负载在应用里也确实变快（不是只有裸引擎才快）
  3. 关掉它（BONSAI_SPEC=off）能回到基线，说明我测的差异确实来自它

用法:
    python build/test_spec_inapp.py
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

MATERIAL = ("《银杏计划技术备忘》\n"
            "发布流程代号是「银杏 B7」。向量模型选 Qwen3-Embedding-0.6B。\n"
            "引擎包用多架构 CUDA 构建，覆盖 sm_75 到 sm_120。\n"
            "前缀缓存按最长公共前缀复用，资料插在最后一轮之前。\n"
            "拒绝回答适配器只在默认以外的预设里生效。\n")
COPY_Q = MATERIAL + "\n把上面备忘一字不改地抄写一遍。"
FREE_Q = "写一段大约 200 字的散文，描写冬天午后的街道，用词平实。"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def api(port: int, path: str, payload=None, token: str = "", timeout=600.0):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else None


def run_once(spec: str) -> dict:
    port = free_port()
    env = dict(os.environ)
    env["BONSAI_ENGINE_DIR"] = r"E:\src\llama-prism\build-cuda-multi\bin"
    env["BONSAI_CUDA_DIR"] = r"E:\cuda\bin"
    env["BONSAI_SPEC"] = spec
    env["PYTHONIOENCODING"] = "utf-8"
    logfile = ROOT / "devdata" / f"spec-inapp-{spec or 'default'}.log"
    fh = logfile.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "bonsai", "--no-window",
         "--data-dir", str(ROOT / "devdata"), "--port", str(port)],
        cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env,
        creationflags=CREATE_NO_WINDOW)
    out = {"spec": spec or "(默认)"}
    try:
        end = time.time() + 400
        while time.time() < end:
            if proc.poll() is not None:
                raise RuntimeError("应用退出")
            try:
                st = api(port, "/app/state")
                if st.get("engine", {}).get("running") and \
                        st.get("progress", {}).get("stage") == "ready":
                    break
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(2)
        else:
            raise RuntimeError("等待就绪超时")
        token = api(port, "/app/state")["token"]

        for label, q in (("复述型", COPY_Q), ("自由型", FREE_Q)):
            t0 = time.time()
            r = api(port, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": q}],
                "max_tokens": 300, "temperature": 0.0, "seed": 5,
                "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}}, token)
            wall = time.time() - t0
            toks = (r.get("usage") or {}).get("completion_tokens") or 0
            out[label] = {"tokens": toks, "wall": wall,
                          "tps": toks / max(wall, 1e-6)}
        # 引擎自己的日志里有没有投机接受率
        fh.flush()
        text = logfile.read_text(encoding="utf-8", errors="replace")
        out["spec_active"] = "draft acceptance" in text
        elog = ROOT / "devdata" / "logs" / "engine.log"
        if elog.exists():
            out["engine_has_flag"] = "--spec-type" in "\n".join(
                elog.read_text(encoding="utf-8", errors="replace").splitlines()[:5]
            ) or "ngram" in elog.read_text(encoding="utf-8", errors="replace")
    except Exception as e:                                      # noqa: BLE001
        out["error"] = str(e)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
        fh.close()
        time.sleep(4)
    return out


def main() -> int:
    rows = []
    for spec in ("off", "ngram-simple"):
        print(f"── BONSAI_SPEC={spec} ──", flush=True)
        r = run_once(spec)
        rows.append(r)
        if "error" in r:
            print(f"   失败：{r['error']}")
        else:
            print(f"   复述型 {r['复述型']['tps']:6.1f} tok/s "
                  f"({r['复述型']['tokens']} token / {r['复述型']['wall']:.2f}s)")
            print(f"   自由型 {r['自由型']['tps']:6.1f} tok/s "
                  f"({r['自由型']['tokens']} token / {r['自由型']['wall']:.2f}s)")
            print(f"   日志里有投机接受率: {r.get('spec_active')}")

    print("\n" + "=" * 62)
    off = next((r for r in rows if r["spec"] == "off"), {})
    on = next((r for r in rows if r["spec"] == "ngram-simple"), {})
    for label in ("复述型", "自由型"):
        a = (off.get(label) or {}).get("tps")
        b = (on.get(label) or {}).get("tps")
        if a and b:
            print(f"  {label}: {a:.1f} → {b:.1f} tok/s   ×{b / a:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
