#!/usr/bin/env python3
"""排查：openai SDK 的 responses 流式为什么收不到增量。

裸 HTTP 打 /v1/responses 流式能拿到 109 个 response.output_text.delta，
但 SDK 迭代完一个都没有。差别只可能在 SSE 的**线格式**上：SDK 的
Responses 解析器比 Chat 的严格。所以把原始字节打出来看。
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


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> int:
    data_dir = ROOT / "devdata"
    port = free_port()
    env = dict(os.environ)
    env["BONSAI_ENGINE_DIR"] = r"E:\src\llama-prism\build-cuda-multi\bin"
    env["BONSAI_CUDA_DIR"] = r"E:\cuda\bin"
    env["PYTHONIOENCODING"] = "utf-8"
    log = (data_dir / "sse-probe.log").open("w", encoding="utf-8", errors="replace")
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

        body = json.dumps({"model": "bonsai", "input": "数到三",
                           "max_output_tokens": 40, "stream": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/responses",
                                     data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + token)
        req.add_header("Accept", "text/event-stream")
        with urllib.request.urlopen(req, timeout=300) as r:
            print(f"状态 {r.status}")
            print(f"响应头：{dict(r.headers)}")
            raw = b""
            while len(raw) < 900:
                c = r.read1(4096)
                if not c:
                    break
                raw += c
        print("\n=== 原始 SSE 前 900 字节（\\r 显示为 <CR>）===")
        print(raw[:900].decode("utf-8", "replace")
              .replace("\r", "<CR>").replace("\n", "<LF>\n"))
        print("\n=== 是否含 'event:' 行 ===")
        text = raw.decode("utf-8", "replace")
        print("  有 event: 行" if "\nevent:" in text or text.startswith("event:")
              else "  没有 event: 行（类型只在 data 的 JSON 里）")

        print("\n=== SDK 视角 ===")
        try:
            import openai
            from openai import OpenAI
            print(f"  openai {openai.__version__}")
            client = OpenAI(api_key=token,
                            base_url=f"http://127.0.0.1:{port}/v1", timeout=300)
            n = 0
            kinds: dict[str, int] = {}
            for ev in client.responses.create(model="bonsai", input="数到三",
                                              max_output_tokens=40, stream=True):
                t = getattr(ev, "type", "?")
                kinds[t] = kinds.get(t, 0) + 1
                n += 1
                if n <= 3:
                    print(f"  事件 {n}: type={t!r}  字段={list(ev.model_fields_set)[:8]}")
            print(f"  共 {n} 个事件：{kinds}")
        except Exception as e:                                  # noqa: BLE001
            print(f"  SDK 报错：{type(e).__name__}: {e}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        (data_dir / "sse-probe.log").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
