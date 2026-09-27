#!/usr/bin/env python3
"""鹈鹕骑自行车测试：给模型一套最小文件工具，看它能不能把空间关系落成可渲染的图形。

这个提示词（Simon Willison 的 pelican riding a bicycle）测的不是知识，是**把语言
描述转成结构化图形**的能力：要同时处理"鸟的形状""自行车的结构""两者叠在一起"三层
空间关系。知识题答得好的模型在这里可能一塌糊涂，反之亦然。

「给一点基本工具」是故意的：SVG 动辄几千字符，弱模型一次性吐完很容易中途跑偏或
截断。给它们 write_file 之后，可以把图形分几次写出来 —— 这更接近真实产品里
带工具调用的用法。

失败也要留证据：如果模型压根不会调工具，就从正文里抽 <svg>；如果正文里也没有，
那"生成了一个 SVG"这件事本身就是假的。三种情况分别记录。

用法:
    python build/test_pelican.py --label 三值1.75bit --model X.gguf
    python build/test_pelican.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
WORK = ROOT / "build" / "work" / "pelican"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

PROMPT = "Generate an SVG of a pelican riding a bicycle."

SYSTEM = (
    "你是一个会使用工具的助手。你可以把文件写到工作目录里。"
    "需要输出较长内容（比如 SVG）时，用 write_file 工具保存成文件，"
    "而不是把全部内容直接写在回答里。写完可以用 read_file 或 list_files 确认。"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "把一个文件写入工作目录。path 是相对路径，content 是完整内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径，例如 pelican.svg"},
                    "content": {"type": "string", "description": "文件的完整内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读回工作目录里某个文件的内容。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "相对路径"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出工作目录里已有的文件。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def api(port: int, path: str, payload=None, timeout=1800.0):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def safe_path(work: Path, rel: str) -> Path:
    """把模型给的路由夹在工作目录里，别让它写到外面。"""
    p = (work / rel.strip().lstrip("/\\")).resolve()
    if not str(p).startswith(str(work.resolve())):
        raise ValueError("路径越界")
    return p


def do_tool(work: Path, name: str, args: dict) -> str:
    try:
        if name == "write_file":
            p = safe_path(work, str(args.get("path") or "out.svg"))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(args.get("content") or ""), encoding="utf-8")
            return f"已写入 {p.relative_to(work)}（{len(str(args.get('content') or ''))} 字符）"
        if name == "read_file":
            p = safe_path(work, str(args.get("path") or ""))
            if not p.exists():
                return f"文件不存在：{args.get('path')}"
            t = p.read_text(encoding="utf-8", errors="replace")
            return t[:4000] + ("\n...(截断)" if len(t) > 4000 else "")
        if name == "list_files":
            fs = [str(x.relative_to(work)) for x in work.rglob("*") if x.is_file()]
            return "空目录" if not fs else "\n".join(fs[:40])
        return f"未知工具 {name}"
    except Exception as e:                                      # noqa: BLE001
        return f"工具执行失败：{type(e).__name__}: {e}"


XML_RE = re.compile(r"<svg\b.*?</svg>", re.IGNORECASE | re.DOTALL)
FENCE_RE = re.compile(r"```(?:svg|xml|html)?\s*(<svg\b.*?</svg>)\s*```", re.IGNORECASE | re.DOTALL)


def extract_svg(text: str) -> str | None:
    if not text:
        return None
    m = FENCE_RE.search(text) or XML_RE.search(text)
    return m.group(1) if m else None


def render(svg_path: Path, png_path: Path) -> bool:
    """用 Edge 无头模式把 SVG 渲染成 PNG。套一层 HTML 保证有白底和正确尺寸。"""
    if not EDGE.exists():
        return False
    html = png_path.with_suffix(".html")
    try:
        svg = svg_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    html.write_text(
        "<!doctype html><meta charset=utf-8>"
        "<style>html,body{margin:0;padding:0;background:#fff}"
        "svg{display:block;width:800px;height:600px}</style>" + svg,
        encoding="utf-8")
    cmd = [str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars",
           "--force-device-scale-factor=1", "--window-size=800,600",
           f"--screenshot={png_path}", html.as_uri()]
    try:
        subprocess.run(cmd, capture_output=True, timeout=90,
                       creationflags=CREATE_NO_WINDOW)
    except Exception:                                           # noqa: BLE001
        return False
    return png_path.exists() and png_path.stat().st_size > 200


def run_one(label: str, model: Path, ctx: int, max_turns: int,
            moe: int | None = None) -> dict:
    work = WORK / label
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    port = free_port()
    log = ROOT / "build" / "bench-logs" / f"pelican-{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(BIN / "llama-server.exe"), "-m", str(model),
           "--host", "127.0.0.1", "--port", str(port),
           "-c", str(ctx), "-ngl", "99", "-fa", "on", "-np", "1",
           "--jinja", "--no-warmup", "--no-webui", "--alias", "m"]
    if moe:
        cmd += ["--n-cpu-moe", str(moe)]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])

    print(f"\n{'=' * 74}\n── {label} ── {model.name}"
          + (f"  [--n-cpu-moe {moe}]" if moe else ""))
    proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh, stderr=subprocess.STDOUT,
                            env=env, creationflags=CREATE_NO_WINDOW)
    out: dict = {"label": label, "model": str(model), "prompt": PROMPT,
                 "tool_calls": 0, "turns": [], "svg_path": "", "used_tool": False}
    try:
        t0 = time.time()
        while time.time() - t0 < 300:
            if proc.poll() is not None:
                raise RuntimeError(f"引擎退出 {proc.returncode}")
            try:
                if api(port, "/health", timeout=3).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(0.7)
        else:
            raise RuntimeError("就绪超时")

        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": PROMPT}]
        for turn in range(max_turns):
            body = {"messages": messages, "tools": TOOLS, "tool_choice": "auto",
                    # 这里原来是 3000，会把 write_file 的参数写一半就截断 ——
                    # 截断后的 JSON 不完整，llama.cpp 直接返回 HTTP 500（不是优雅
                    # 降级）。实测这个任务要 3300+ token，留到 8000。
                    "max_tokens": 8000, "temperature": 0.2, "seed": 42,
                    "chat_template_kwargs": {"enable_thinking": False}}
            r = api(port, "/v1/chat/completions", body)
            msg = (r.get("choices") or [{}])[0].get("message", {}) or {}
            calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()
            out["turns"].append({"turn": turn, "tool_calls": len(calls),
                                 "content_chars": len(content)})
            print(f"   [轮 {turn+1}] 工具调用 {len(calls)} 个, 正文 {len(content)} 字符")

            messages.append({"role": "assistant", "content": content,
                             **({"tool_calls": calls} if calls else {})})
            if calls:
                out["used_tool"] = True
                for c in calls:
                    fn = (c.get("function") or {})
                    name = fn.get("name") or ""
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:                           # noqa: BLE001
                        args = {}
                    res = do_tool(work, name, args)
                    out["tool_calls"] += 1
                    print(f"        -> {name}({str(args.get('path') or '')[:40]})  {res[:60]}")
                    messages.append({"role": "tool", "tool_call_id": c.get("id") or name,
                                     "content": res})
                continue
            # 没有工具调用：可能是最终回答，也可能是它压根不会用工具
            svg = extract_svg(content)
            if svg:
                out["svg_from"] = "text"
                (work / "from_text.svg").write_text(svg, encoding="utf-8")
            out["final_text"] = content[:2000]
            break

        # 收成果：优先用模型自己写的 .svg
        svgs = sorted(work.rglob("*.svg"), key=lambda p: -p.stat().st_size)
        if svgs and svgs[0].stat().st_size > 200:
            out["svg_path"] = str(svgs[0])
            out["svg_bytes"] = svgs[0].stat().st_size
            out["svg_from"] = out.get("svg_from") or "tool"
        elif (work / "from_text.svg").exists():
            out["svg_path"] = str(work / "from_text.svg")
            out["svg_bytes"] = (work / "from_text.svg").stat().st_size

        if out["svg_path"]:
            png = work / "render.png"
            out["rendered"] = render(Path(out["svg_path"]), png)
            out["png_path"] = str(png) if out["rendered"] else ""
            print(f"   产物 {out['svg_bytes']:,} B  SVG -> 渲染{'成功' if out['rendered'] else '失败'}")
        else:
            print("   ✗ 没有任何 SVG 产物")
    except Exception as e:                                      # noqa: BLE001
        out["error"] = str(e)
        print(f"   ✗ {e}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        fh.close()
        time.sleep(2)
    return out


CANDIDATES = {
    "三值1.75bit": (r"E:\models\bonsai\Ternary-Bonsai-2-27B-PTQ1_0.gguf", None),
    "三值2.0bit": (r"E:\models\bonsai\alts\Ternary-Bonsai-2-27B-PQ2_0.gguf", None),
    "A3B-MoE-Q3": (r"E:\models\bonsai\alts\Qwen3.6-35B-A3B-uncensored-Q3_K_M.gguf", 14),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--moe", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.all:
        jobs = list(CANDIDATES.items())
    else:
        if not args.model:
            print("要么 --all，要么给 --model")
            return 2
        jobs = [(args.label or Path(args.model).stem[:20],
                 (args.model, args.moe or None))]

    results = []
    for label, (mpath, moe) in jobs:
        m = Path(mpath)
        if not m.exists():
            print(f"跳过 {label}：{m} 不存在")
            continue
        results.append(run_one(label, m, args.ctx, args.max_turns, moe))

    print(f"\n{'=' * 74}\n{'模型':<16}{'调工具':>7}{'调用数':>7}{'SVG字节':>10}{'渲染':>6}")
    for r in results:
        print(f"{r['label']:<16}{'是' if r.get('used_tool') else '否':>7}"
              f"{r.get('tool_calls', 0):>7}{r.get('svg_bytes', 0):>10,}"
              f"{'OK' if r.get('rendered') else '—':>6}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        print(f"\n写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
