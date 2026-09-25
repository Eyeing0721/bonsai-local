#!/usr/bin/env python3
"""OpenAI 接口兼容性测试。

为什么专门量"首字节时间"：流式最典型的坏法不是报错，而是**看起来正常但其实是
一次性的** —— 代理把上游的 SSE 读满一个缓冲区才往下转，客户端要等整段生成完
才看到第一个字。功能测试全过，体验全错。所以这里把首字节时间和总时长分开记：
真正的流式，首字节应该远小于总时长；被缓冲的话两者几乎相等。

覆盖：
  GET  /v1/models
  POST /v1/chat/completions          （非流式 / 流式 / 首字节时间）
  POST /v1/completions               （流式）
  POST /v1/responses                 （非流式 / 流式 / 事件类型）
  POST /v1/responses/input_tokens     （token 计数）
  鉴权、错误格式
  知识库注入对 chat 与 responses 都生效
  装了 openai SDK 的话，用官方 SDK 再走一遍

用法:
    python build/test_openai_api.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'✓' if ok else '✗'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)
    return ok


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def call(port: int, path: str, body=None, token: str = "", stream: bool = False,
         timeout: float = 600.0):
    """返回 (status, headers, body)。stream=True 时 body 是还没读的 response。"""
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    if stream:
        return r.status, dict(r.headers), r
    with r:
        return r.status, dict(r.headers), r.read()


def read_json(payload: bytes) -> dict:
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception:                                           # noqa: BLE001
        return {}


def responses_text(j: dict) -> tuple[str, str]:
    """从 Responses 响应里取出 (正文, 思维) 两段文字。

    注意 `output_text` **不是线格式字段** —— 它是 OpenAI Python SDK 在客户端
    从 output 数组里拼出来的便利属性。裸 HTTP 拿到的响应里没有它，所以要自己
    走 output[].content[].text，并按 item.type 区分 reasoning 和 message。
    """
    out = reason = ""
    for item in j.get("output") or []:
        if not isinstance(item, dict):
            continue
        text = "".join(p.get("text", "") for p in (item.get("content") or [])
                       if isinstance(p, dict))
        if item.get("type") == "reasoning":
            reason += text
        else:
            out += text
    if not out and isinstance(j.get("output_text"), str):
        out = j["output_text"]
    return out, reason


def stream_timing(resp, expect_sse: bool = True):
    """把流式响应读完，同时记录首字节时间、事件数与总时长。"""
    t0 = time.time()
    first = None
    raw = b""
    events = 0
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        if first is None:
            first = time.time() - t0
        raw += chunk
        events += chunk.count(b"\n\n") if expect_sse else 1
    total = time.time() - t0
    resp.close()
    return raw, (first if first is not None else total), total, events


def parse_sse(raw: bytes) -> list[dict]:
    out = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                try:
                    out.append(json.loads(data))
                except Exception:                               # noqa: BLE001
                    pass
    return out


DOC = """# 项目内部笔记

本项目的发布流程代号是「银杏 B7」。发布前必须跑一遍 build/make_release.py
并确认 exe 自检七项全过。引擎包使用多架构 CUDA 构建（sm_75 到 sm_120）。
"""


def main() -> int:
    data_dir = ROOT / "devdata"
    port = free_port()
    env = dict(os.environ)
    env["BONSAI_ENGINE_DIR"] = r"E:\src\llama-prism\build-cuda-multi\bin"
    env["BONSAI_CUDA_DIR"] = r"E:\cuda\bin"
    env["BONSAI_EMBED_MODEL"] = r"E:\models\bonsai\embed\Qwen3-Embedding-0.6B-Q8_0.gguf"
    env["PYTHONIOENCODING"] = "utf-8"

    doc = data_dir / "apitest.md"
    doc.write_text(DOC, encoding="utf-8")
    log = (data_dir / "apitest-app.log").open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "bonsai", "--no-window",
         "--data-dir", str(data_dir), "--port", str(port)],
        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env,
        creationflags=CREATE_NO_WINDOW)

    base = f"http://127.0.0.1:{port}"
    try:
        print(f"启动应用（端口 {port}）…")
        deadline = time.time() + 600
        while time.time() < deadline:
            if proc.poll() is not None:
                print((data_dir / "apitest-app.log").read_text(
                    encoding="utf-8", errors="replace")[-1500:])
                return 1
            try:
                st = call(port, "/app/state")[2]
                st = json.loads(st)
                if st.get("engine", {}).get("running") and \
                        st.get("progress", {}).get("stage") == "ready":
                    break
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(2)
        else:
            print("等待就绪超时")
            return 1
        token = json.loads(call(port, "/app/state")[2])["token"]
        print("模型已就绪\n")

        call(port, "/app/kb/clear", {}, token)

        print("[1] GET /v1/models")
        s, _, b = call(port, "/v1/models", token=token)
        j = read_json(b)
        check("状态 200", s == 200, str(s))
        ids = [m.get("id") for m in (j.get("data") or [])]
        check("列出模型", bool(ids), ", ".join(str(i) for i in ids))

        print("\n[2] 鉴权")
        s, _, b = call(port, "/v1/models")
        check("无令牌被拒", s == 401, str(s))
        err = read_json(b).get("error") or {}
        check("错误体是 OpenAI 格式",
              err.get("type") == "invalid_request_error" and err.get("code"),
              json.dumps(err, ensure_ascii=False)[:80])

        print("\n[3] /v1/chat/completions 非流式")
        s, _, b = call(port, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "用一句话说明什么是栈溢出"}],
            "max_tokens": 120, "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False},
        }, token)
        j = read_json(b)
        msg = (j.get("choices") or [{}])[0].get("message", {})
        txt = msg.get("content", "")
        check("状态 200", s == 200, str(s))
        check("有内容", bool(txt.strip()), (txt or "")[:60] + "…")
        check("默认不思考（不占 max_tokens）",
              not (msg.get("reasoning_content") or "").strip(),
              f"思维 {len(msg.get('reasoning_content') or '')} 字")
        check("有 usage", isinstance(j.get("usage"), dict))

        print("\n[3b] 显式 reasoning_effort 应当真的开启思考")
        # 预算给足：xhigh 下同一道题实测要 1000+ token（思维占大头）。给 400 的
        # 话思维就把配额吃光，正文为空 —— 那不是 bug，是这道题的预期行为。
        s, _, b = call(port, "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "一个农夫要带狼、羊、白菜过河，船只能带一样。给出步骤。"}],
            "max_tokens": 1200, "temperature": 0.3, "reasoning_effort": "high",
        }, token)
        msg = (read_json(b).get("choices") or [{}])[0].get("message", {})
        raw3b = read_json(b)
        check("显式 reasoning_effort 生效",
              bool((msg.get("reasoning_content") or "").strip()),
              f"思维 {len(msg.get('reasoning_content') or '')} 字  "
              f"原始={json.dumps(raw3b, ensure_ascii=False)[:220]}")
        check("同时有正文", bool((msg.get("content") or "").strip()),
              (msg.get("content") or "")[:60] + "…")

        print("\n[3c] 标准 OpenAI 推理强度取值都要能用（底座模板只认 xhigh/medium/low）")
        for effort in ("minimal", "low", "medium", "high"):
            s, _, b = call(port, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "1+1 等于几"}],
                "max_tokens": 200, "reasoning_effort": effort,
            }, token)
            j = read_json(b)
            msg = (j.get("choices") or [{}])[0].get("message", {})
            thinking = bool((msg.get("reasoning_content") or "").strip())
            body_text = bool((msg.get("content") or "").strip())
            check(f"reasoning_effort={effort} 不报错", s == 200,
                  f"HTTP {s}" + ("" if s == 200 else
                                 "  " + json.dumps(j, ensure_ascii=False)[:120]))
            if s == 200:
                want_think = effort not in ("minimal",)
                check(f"  {effort} 思维={'开' if want_think else '关'} 符合预期",
                      thinking == want_think,
                      f"思维 {len(msg.get('reasoning_content') or '')} 字，"
                      f"正文 {len(msg.get('content') or '')} 字")
                check(f"  {effort} 有正文", body_text)

        print("\n[4] /v1/chat/completions 流式（含首字节时间）")
        s, hdr, resp = call(port, "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "写一段 200 字左右介绍海洋的短文。"}],
            "max_tokens": 200, "temperature": 0.3,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }, token, stream=True)
        raw, first, total, n_ev = stream_timing(resp)
        check("状态 200", s == 200, str(s))
        check("Content-Type 是 event-stream",
              "text/event-stream" in (hdr.get("Content-Type") or ""),
              str(hdr.get("Content-Type")))
        check("收到多个 SSE 事件", n_ev >= 3, f"{n_ev} 个")
        check("以 [DONE] 结束", raw.rstrip().endswith(b"[DONE]"), "")
        deltas = [e for e in parse_sse(raw)
                  if (e.get("choices") or [{}])[0].get("delta", {}).get("content")]
        check("逐块下发内容", len(deltas) >= 5, f"{len(deltas)} 个内容增量")
        # 关键判据：真流式的首字节必须远小于总时长
        ratio = first / max(total, 1e-6)
        check("首字节明显早于结束（真流式）", first < total * 0.5,
              f"首字节 {first:.2f}s / 总 {total:.2f}s = {ratio:.0%}")

        print("\n[5] /v1/completions 流式")
        s, _, resp = call(port, "/v1/completions", {
            "prompt": "Once upon a time", "max_tokens": 60, "stream": True,
        }, token, stream=True)
        raw, first, total, n_ev = stream_timing(resp)
        check("状态 200", s == 200, str(s))
        check("有内容增量", len(parse_sse(raw)) >= 2, f"{n_ev} 个事件")

        print("\n[6] /v1/responses 非流式")
        s, _, b = call(port, "/v1/responses", {
            "model": "bonsai",
            "instructions": "你是一个简洁的助手。",
            "input": "用一句话解释什么是堆溢出",
            "max_output_tokens": 120, "temperature": 0.2,
        }, token)
        j = read_json(b)
        check("状态 200", s == 200, str(s))
        check("id 形如 resp_", str(j.get("id", "")).startswith("resp_"), str(j.get("id")))
        out, reason = responses_text(j)
        check("有输出文字", bool(out.strip()), (out or "")[:70] + "…")
        check("默认不思考（不占 max_output_tokens）", not reason.strip(),
              f"思维 {len(reason)} 字")
        check("有 usage", isinstance(j.get("usage"), dict), str(j.get("usage"))[:60])

        print("\n[6b] 显式要求思考时应该真的思考")
        s, _, b = call(port, "/v1/responses", {
            "input": "一个农夫要带狼、羊、白菜过河，船只能带一样。给出步骤。",
            "reasoning": {"effort": "high"}, "max_output_tokens": 1200,
            "temperature": 0.3,
        }, token)
        j = read_json(b)
        out, reason = responses_text(j)
        check("显式 reasoning.effort 生效", bool(reason.strip()),
              f"思维 {len(reason)} 字  "
              f"原始={json.dumps(j, ensure_ascii=False)[:220]}")
        check("同时有正文", bool(out.strip()), (out or "")[:60] + "…")

        print("\n[7] /v1/responses 流式（事件类型 + 首字节时间）")
        s, hdr, resp = call(port, "/v1/responses", {
            "model": "bonsai",
            "input": "写一段 200 字左右介绍沙漠的短文。",
            "max_output_tokens": 200, "temperature": 0.3, "stream": True,
        }, token, stream=True)
        raw, first, total, n_ev = stream_timing(resp)
        events = parse_sse(raw)
        types = [e.get("type") for e in events]
        check("状态 200", s == 200, str(s))
        check("Content-Type 是 event-stream",
              "text/event-stream" in (hdr.get("Content-Type") or ""),
              str(hdr.get("Content-Type")))
        for want in ("response.created", "response.output_text.delta",
                     "response.completed"):
            check(f"事件 {want}", want in types, f"共 {len(types)} 个事件")
        deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
        check("多个文字增量", len(deltas) >= 5, f"{len(deltas)} 个")
        check("首字节明显早于结束（真流式）", first < total * 0.5,
              f"首字节 {first:.2f}s / 总 {total:.2f}s")

        print("\n[8] /v1/responses/input_tokens")
        s, _, b = call(port, "/v1/responses/input_tokens",
                       {"input": "这是一句用来数 token 的话。"}, token)
        j = read_json(b)
        check("状态 200", s == 200, str(s))
        check("返回 token 数", isinstance(j.get("input_tokens"), int)
              or isinstance(j.get("object"), str),
              json.dumps(j, ensure_ascii=False)[:80])

        print("\n[9] 知识库注入：chat 与 responses 都要生效")
        call(port, "/app/kb/add", {"path": str(doc)}, token)
        call(port, "/app/kb/enabled", {"enabled": True}, token)
        q = "本项目的发布流程代号是什么？只回答代号。"
        s, _, b = call(port, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": q}],
            "max_tokens": 80, "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        }, token)
        t = (read_json(b).get("choices") or [{}])[0].get("message", {}).get("content", "")
        check("chat 用上了资料", "银杏" in t or "B7" in t, (t or "")[:70])

        s, _, b = call(port, "/v1/responses", {
            "input": q, "max_output_tokens": 80, "temperature": 0.1,
        }, token)
        out, _ = responses_text(read_json(b))
        check("responses 用上了资料", "银杏" in out or "B7" in out, (out or "")[:70])

        s, _, resp = call(port, "/v1/responses", {
            "input": q, "max_output_tokens": 80, "temperature": 0.1, "stream": True,
        }, token, stream=True)
        raw, _, _, _ = stream_timing(resp)
        joined = "".join(e.get("delta", "") for e in parse_sse(raw)
                         if e.get("type") == "response.output_text.delta")
        check("responses 流式也用上了资料", "银杏" in joined or "B7" in joined,
              (joined or "")[:70])
        call(port, "/app/kb/clear", {}, token)

        print("\n[10] 官方 openai SDK")
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            print("  – 没装 openai SDK，跳过（pip install openai 可以补上）")
        else:
            print(f"  （SDK 版本 {openai.__version__}）")
            client = OpenAI(api_key=token, base_url=f"{base}/v1", timeout=600)

            def sdk(label, fn):
                try:
                    val = fn()
                    check(label, bool(str(val).strip()), str(val)[:60])
                except AttributeError as e:
                    print(f"  – {label}：这个 SDK 版本没有对应接口（{e}）")
                except Exception as e:                          # noqa: BLE001
                    check(label, False, f"{type(e).__name__}: {e}")

            sdk("SDK chat 非流式", lambda: client.chat.completions.create(
                model="bonsai", messages=[{"role": "user", "content": "说一句问候语"}],
                max_tokens=60, temperature=0.2).choices[0].message.content)

            def sdk_stream():
                got = ""
                for chunk in client.chat.completions.create(
                        model="bonsai", messages=[{"role": "user", "content": "数到五"}],
                        max_tokens=60, temperature=0.2, stream=True):
                    if chunk.choices and chunk.choices[0].delta.content:
                        got += chunk.choices[0].delta.content
                return got
            sdk("SDK chat 流式", sdk_stream)

            sdk("SDK responses 非流式", lambda: getattr(
                client.responses.create(model="bonsai", input="说一句问候语",
                                        max_output_tokens=60), "output_text", ""))

            def sdk_resp_stream():
                acc = ""
                for ev in client.responses.create(model="bonsai", input="数到五",
                                                  max_output_tokens=60, stream=True):
                    if getattr(ev, "type", "") == "response.output_text.delta":
                        acc += ev.delta
                return acc
            sdk("SDK responses 流式", sdk_resp_stream)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        doc.unlink(missing_ok=True)
        (data_dir / "apitest-app.log").unlink(missing_ok=True)

    print("\n" + "=" * 62)
    if FAILED:
        print(f"失败 {len(FAILED)} 项：")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("OpenAI 接口兼容性：全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
