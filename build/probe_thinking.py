#!/usr/bin/env python3
"""排查思维链对 OpenAI 接口的影响。

起因：SDK 的 responses 流式「收不到增量」。查下来不是代理的问题 —— 是模型默认
开了思维链，40 个 token 的预算全花在 reasoning 上，一个 output_text 都没产生。

对 OpenAI 兼容性来说这是大事：客户端设 max_tokens=100 会拿到**空回答**，看起来
完全像服务坏了。所以要确认：
  1. /v1/chat/completions 默认是不是开思维链
  2. 关掉的办法（chat_template_kwargs / reasoning_effort）在哪些端点上有效
  3. /v1/responses 上怎么关
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
Q = "用一句话说明什么是缓存"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def post(port: int, path: str, body: dict, token: str, timeout=300) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"__http_error__": e.code, "body": e.read().decode("utf-8", "replace")[:200]}


def main() -> int:
    data_dir = ROOT / "devdata"
    port = free_port()
    env = dict(os.environ)
    env["BONSAI_ENGINE_DIR"] = r"E:\src\llama-prism\build-cuda-multi\bin"
    env["BONSAI_CUDA_DIR"] = r"E:\cuda\bin"
    env["PYTHONIOENCODING"] = "utf-8"
    log = (data_dir / "think-probe.log").open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "bonsai", "--no-window",
         "--data-dir", str(data_dir), "--port", str(port)],
        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env,
        creationflags=CREATE_NO_WINDOW)
    try:
        for _ in range(200):
            if proc.poll() is not None:
                print("应用退出")
                return 1
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/app/state", timeout=4) as r:
                    st = json.loads(r.read())
                if st.get("engine", {}).get("running") and \
                        st.get("progress", {}).get("stage") == "ready":
                    break
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(2)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/app/state",
                                    timeout=8) as r:
            token = json.loads(r.read())["token"]

        cases = [
            ("chat 默认（不传任何开关）", "/v1/chat/completions",
             {"messages": [{"role": "user", "content": Q}], "max_tokens": 100}),
            ("chat + enable_thinking=false", "/v1/chat/completions",
             {"messages": [{"role": "user", "content": Q}], "max_tokens": 100,
              "chat_template_kwargs": {"enable_thinking": False}}),
            ("chat + reasoning_effort=none", "/v1/chat/completions",
             {"messages": [{"role": "user", "content": Q}], "max_tokens": 100,
              "reasoning_effort": "none"}),
            ("chat + reasoning_effort=low", "/v1/chat/completions",
             {"messages": [{"role": "user", "content": Q}], "max_tokens": 100,
              "reasoning_effort": "low"}),
            ("responses 默认", "/v1/responses",
             {"input": Q, "max_output_tokens": 100}),
            ("responses + reasoning.effort=none", "/v1/responses",
             {"input": Q, "max_output_tokens": 100,
              "reasoning": {"effort": "none"}}),
            ("responses + chat_template_kwargs", "/v1/responses",
             {"input": Q, "max_output_tokens": 100,
              "chat_template_kwargs": {"enable_thinking": False}}),
        ]
        for label, path, body in cases:
            t0 = time.time()
            r = post(port, path, body, token)
            dt = time.time() - t0
            if "__http_error__" in r:
                print(f"  {label:<38} HTTP {r['__http_error__']}  {r['body'][:90]}")
                continue
            if path.endswith("responses"):
                # 别只读 output_text：它不一定出现。把整个 output 数组摊平看，
                # 才知道到底是"没生成正文"还是"我取错了字段"。
                items = r.get("output") or []
                out = r.get("output_text") or ""
                reason = ""
                for it in items:
                    txt = "".join(p.get("text", "") for p in (it.get("content") or []))
                    if it.get("type") == "reasoning":
                        reason += txt
                    else:
                        out += txt
                toks = r.get("usage", {}).get("output_tokens")
                if not r.get("output_text"):
                    print(f"      （没有 output_text 字段；output 里有 "
                          f"{len(items)} 项：{[i.get('type') for i in items]}）")
            else:
                msg = (r.get("choices") or [{}])[0].get("message", {})
                out = msg.get("content") or ""
                reason = msg.get("reasoning_content") or ""
                toks = r.get("usage", {}).get("completion_tokens")
            print(f"  {label:<38} {dt:5.1f}s  正文 {len(out):>4} 字  思维 {len(reason):>4} 字  "
                  f"token={toks}")
            if not out.strip():
                print(f"      ⚠ 正文为空 —— 客户端只会看到空回答")

        print("\n=== 服务端的 chat template 默认值 ===")
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/props", timeout=10) as r:
                props = json.loads(r.read())
            tpl = props.get("chat_template") or ""
            for line in tpl.splitlines():
                if "thinking" in line.lower():
                    print("   " + line.strip()[:160])
        except Exception as e:                                  # noqa: BLE001
            print(f"  读 /props 失败：{e}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        (data_dir / "think-probe.log").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
