#!/usr/bin/env python3
"""代理层对"上游把可解决的问题报得很难懂"的处理。

背景（实测出来的，不是设想）：模型在 write_file 的参数里写 SVG，被 max_tokens
截断时，llama.cpp 有两种反应：

  A. 返回 500，正文是 nlohmann json 的异常文本
  B. 返回 200，但 tool_calls[0].function.arguments 是未闭合的 JSON 字符串

B 更坏 —— 客户端看到 200 就去 json.loads，崩在它自己那边，而且完全看不出原因。

A 是偶发的（取决于截断落在哪个位置），没法按需复现，所以这里用**真实抓到的报错
原文**做断言；B 是稳定可复现的，除了单测还走了一次端到端。

    python tests/test_proxy_errors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonsai.server import (broken_tool_arguments,  # noqa: E402
                           recoverable_upstream_error)

FAILED: list[str] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not ok:
        FAILED.append(name)


# 真实抓到的原文（取自 build/bench-logs/pelican-multi-三值1.75bit.log）。
# 这里保留关键的头部与尾部，中段那一大坨 SVG 转义对测试没有意义。
REAL_500 = (
    '{"error":{"code":500,"message":"Failed to parse tool call arguments as JSON: '
    '[json.exception.parse_error.101] parse error at line 1, column 6199: syntax error '
    'while parsing value - invalid string: missing closing quote; '
    'last read: \'"<svg xmlns=... <ellipse cx=\\"252\\" cy=\\"458\\" rx=\\"16\\" '
    'ry=\\"6\\" fill=\\"#FF98\'","type":"server_error"}}'
)
REAL_CTX = ('{"error":{"code":500,"message":"the request exceeds the available context size, '
            'try increasing it"}}')


def tool_body(args: str) -> bytes:
    return json.dumps({
        "choices": [{"finish_reason": "tool_calls", "index": 0,
                     "message": {"role": "assistant", "content": "",
                                 "tool_calls": [{"type": "function", "function": {
                                     "name": "write_file", "arguments": args}}]}}],
    }).encode()


def main() -> int:
    print("── A. 500 正文识别（真实原文）──")
    hit = recoverable_upstream_error(REAL_500)
    check("识别出工具调用截断", hit is not None and hit[1] == "tool_call_truncated",
          str(hit[1]) if hit else "没认出来")
    check("文案里说明了可以重试",
          bool(hit) and "重试" in hit[0])
    hit2 = recoverable_upstream_error(REAL_CTX)
    check("识别出上下文超限", hit2 is not None and hit2[1] == "context_overflow",
          str(hit2[1]) if hit2 else "没认出来")

    print()
    print("── A'. 不该误伤的上游错误 ──")
    for name, text in (
        ("普通 500", '{"error":{"code":500,"message":"internal server error"}}'),
        ("模型加载失败", '{"error":{"message":"failed to load model"}}'),
        ("空正文", ""),
        ("HTML 错误页", "<html><body>502 Bad Gateway</body></html>"),
    ):
        check(f"{name} 原样放过", recoverable_upstream_error(text) is None)

    print()
    print("── B. 200 但参数不可解析 ──")
    # 真实形态：arguments 是未闭合的 JSON 字符串
    broken = '{"path":"pelican_bicycle.svg","content":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 800 600\\">\\n  <line x1=\\"300\\"'
    check("未闭合的参数被认出来", broken_tool_arguments(tool_body(broken)))
    try:
        json.loads(broken)
        check("（自检）这段确实不是合法 JSON", False, "居然能解析")
    except Exception:
        check("（自检）这段确实不是合法 JSON", True)

    good = json.dumps({"path": "p.svg", "content": "<svg/>"})
    check("合法参数不误判", not broken_tool_arguments(tool_body(good)))

    print()
    print("── B'. 其他响应不该被当成问题 ──")
    check("纯文本回答", not broken_tool_arguments(json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": "你好"}}]}).encode()))
    check("没有 tool_calls", not broken_tool_arguments(json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": ""}}]}).encode()))
    check("非 JSON 正文（比如 SSE 碎片）", not broken_tool_arguments(b"data: {}\n\n"))
    check("空正文", not broken_tool_arguments(b""))
    check("顶层是数组", not broken_tool_arguments(b'[1,2,3]'))
    check("tool_calls 里 arguments 缺失", not broken_tool_arguments(json.dumps(
        {"choices": [{"message": {"tool_calls": [{"function": {"name": "x"}}]}}]}).encode()))

    print()
    if FAILED:
        print(f"自检结果：{len(FAILED)} 项失败 -> {', '.join(FAILED)}")
        return 1
    print("自检结果：全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
