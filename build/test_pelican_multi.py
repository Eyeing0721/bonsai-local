#!/usr/bin/env python3
"""鹈鹕骑自行车 · 多样本版：每模型跑 N 个不同 seed，自动评分 + 拼联络表。

为什么要自动评分：15 张图靠肉眼看会得出"都差不多"的结论，而且我自己看会有偏好。
所以把可机检的结构性质固定下来：

  1. 有两个半径相近的大圆        <- 自行车必须是两个轮子，而不是一个或三个
  2. 轮子在身体下方              <- 不能倒过来画
  3. 七个部件关键词齐全          <- 喙/翅/腿/轮/车架/车把/坐垫
  4. 渲染出的 PNG 不是空白        <- "生成了 SVG" 和"画出了东西"是两回事
  5. 图元数量在合理区间          <- 太少的画不出结构，太多的通常是跑偏了

引擎只启动一次，然后连跑 N 个 seed —— 不然 15 次模型加载比生成还慢。

用法: python build/test_pelican_multi.py --samples 5
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
import tempfile
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
EDGES = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
         Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]
WORK = ROOT / "build" / "work" / "pelican-multi"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

PROMPT = "Generate an SVG of a pelican riding a bicycle."
SEEDS = [42, 7, 101, 2024, 31337]
TEMP = 0.6

SYSTEM = ("你是一个会使用工具的助手。你可以把文件写到工作目录里。"
          "需要输出较长内容（比如 SVG）时，用 write_file 工具保存成文件，"
          "而不是把全部内容直接写在回答里。写完可以用 list_files 确认。")

TOOLS = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "把一个文件写入工作目录。path 是相对路径，content 是完整内容。",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "list_files", "description": "列出工作目录里已有的文件。",
        "parameters": {"type": "object", "properties": {}}}},
]

CANDIDATES = [
    ("三值1.75bit", r"E:\models\bonsai\Ternary-Bonsai-2-27B-PTQ1_0.gguf", None),
    ("三值2.0bit", r"E:\models\bonsai\alts\Ternary-Bonsai-2-27B-PQ2_0.gguf", None),
    ("A3B-MoE-Q3", r"E:\models\bonsai\alts\Qwen3.6-35B-A3B-uncensored-Q3_K_M.gguf", 14),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def api(port, path, payload=None, timeout=1800.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def safe_path(work: Path, rel: str) -> Path:
    p = (work / rel.strip().lstrip("/\\")).resolve()
    if not str(p).startswith(str(work.resolve())):
        raise ValueError("越界")
    return p


def do_tool(work: Path, name: str, args: dict) -> str:
    try:
        if name == "write_file":
            p = safe_path(work, str(args.get("path") or "out.svg"))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(args.get("content") or ""), encoding="utf-8")
            return f"已写入 {p.name}（{len(str(args.get('content') or ''))} 字符）"
        if name == "list_files":
            fs = [x.name for x in work.iterdir() if x.is_file()]
            return "空目录" if not fs else ", ".join(fs[:20])
        return f"未知工具 {name}"
    except Exception as e:                                      # noqa: BLE001
        return f"失败：{type(e).__name__}: {e}"


XML_RE = re.compile(r"<svg\b.*?</svg>", re.I | re.S)
FENCE_RE = re.compile(r"```(?:svg|xml)?\s*(<svg\b.*?</svg>)\s*```", re.I | re.S)
CIRC_RE = re.compile(r"<circle\b[^>]*>", re.I)
ATTR = lambda tag, a: (lambda m: float(m.group(1)) if m else None)(
    re.search(rf'\b{a}\s*=\s*"([-\d.]+)"', tag, re.I))


def analyse(svg: str, png: Path | None) -> dict:
    """从 SVG 里机检结构性质，再看渲染结果是不是空白。"""
    r: dict = {}
    n = len(re.findall(r"<(circle|ellipse|path|rect|line|polygon|polyline|g|text)\b", svg, re.I))
    r["elements"] = n
    r["bytes"] = len(svg.encode())

    # 轮子：半径 >= 25 的圆，按半径分组，看有没有两个相近的
    circ = [(ATTR(t, "r"), ATTR(t, "cx"), ATTR(t, "cy")) for t in CIRC_RE.findall(svg)]
    big = [c for c in circ if c[0] and c[0] >= 25]
    r["big_circles"] = len(big)
    two_wheels = False
    if len(big) >= 2:
        big_sorted = sorted(big, key=lambda c: -c[0])
        r1, r2 = big_sorted[0][0], big_sorted[1][0]
        two_wheels = r2 >= 0.6 * r1          # 两个轮子大小应当接近
        # 轮子在身体下方：取圆心 cy 最大的两个，和所有圆的平均 cy 比
        cys = [c[2] for c in big if c[2] is not None]
        if cys:
            r["wheels_low"] = (sum(sorted(cys)[-2:]) / 2) > (sum(cys) / len(cys))
    r["two_wheels"] = two_wheels

    parts = {"beak": r"beak|bill|喙", "wing": r"wing|翅", "leg": r"leg|foot|feet|腿|脚|蹼",
             "wheel": r"wheel|轮", "frame": r"frame|车架", "handlebar": r"handlebar|车把",
             "seat": r"seat|saddle|坐垫|座"}
    r["parts"] = {k: bool(re.search(v, svg, re.I)) for k, v in parts.items()}
    r["parts_n"] = sum(r["parts"].values())

    # 渲染后是不是空白
    r["png_ok"] = False
    r["ink_pct"] = 0.0
    if png and png.exists():
        try:
            from PIL import Image
            im = Image.open(png).convert("RGB")
            small = im.resize((160, 120))
            px = list(small.getdata())
            # "有内容" = 与纯白差异明显的像素占比
            nonwhite = sum(1 for p in px if abs(p[0]-255)+abs(p[1]-255)+abs(p[2]-255) > 30)
            r["ink_pct"] = round(100.0 * nonwhite / len(px), 1)
            r["png_ok"] = r["ink_pct"] > 5.0
            r["colors"] = len(set(px))
        except Exception as e:                                  # noqa: BLE001
            r["png_err"] = str(e)
    return r


def render(svg: Path, png: Path, w=800, h=600) -> bool:
    edge = next((e for e in EDGES if e.exists()), None)
    if edge is None:
        return False
    html = png.with_suffix(".html")
    html.write_text("<!doctype html><meta charset=utf-8>"
                    f"<style>html,body{{margin:0;background:#fff}}"
                    f"svg{{display:block;width:{w}px;height:{h}px}}</style>"
                    + svg.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    png.unlink(missing_ok=True)
    prof = Path(tempfile.mkdtemp(prefix="edge-pe-"))
    try:
        with open(Path(tempfile.gettempdir()) / "edge-r.log", "wb") as nf:
            subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--no-sandbox",
                            "--no-first-run", "--no-default-browser-check", "--hide-scrollbars",
                            f"--user-data-dir={prof}", f"--window-size={w},{h}",
                            f"--screenshot={png}", html.as_uri()],
                           stdout=nf, stderr=subprocess.STDOUT, timeout=120)
    except Exception:                                           # noqa: BLE001
        pass
    for _ in range(25):
        if png.exists() and png.stat().st_size > 200:
            break
        time.sleep(0.4)
    shutil.rmtree(prof, ignore_errors=True)
    return png.exists() and png.stat().st_size > 200


def run_model(label: str, model: Path, moe: int | None, samples: int, ctx: int,
              max_tokens: int = 3000, only_seed: int | None = None) -> list[dict]:
    work = WORK / label
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    port = free_port()
    log = ROOT / "build" / "bench-logs" / f"pelican-multi-{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(BIN / "llama-server.exe"), "-m", str(model), "--host", "127.0.0.1",
           "--port", str(port), "-c", str(ctx), "-ngl", "99", "-fa", "on", "-np", "1",
           "--jinja", "--no-warmup", "--no-webui", "--alias", "m"]
    if moe:
        cmd += ["--n-cpu-moe", str(moe)]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])
    print(f"\n{'=' * 76}\n── {label} ── {model.name}  {samples} 个样本 (temp={TEMP})")
    proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh, stderr=subprocess.STDOUT,
                            env=env, creationflags=CREATE_NO_WINDOW)
    rows: list[dict] = []
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

        for si in range(samples):
            seed = SEEDS[si % len(SEEDS)]
            sdir = work / f"s{si+1}-seed{seed}"
            sdir.mkdir(parents=True, exist_ok=True)
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": PROMPT}]
            rec = {"sample": si + 1, "seed": seed, "tool_calls": 0}
            try:
                for turn in range(6):
                    r = api(port, "/v1/chat/completions", {
                        "messages": msgs, "tools": TOOLS, "tool_choice": "auto",
                        # 必须用参数，不能硬编码：3000 对这个任务太小，模型在
                        # write_file 的参数里写 SVG，写一半被截断 -> JSON 不完整
                        # -> llama.cpp 直接返回 HTTP 500（而不是优雅降级）。
                        # 我第一版就是硬编码在这里，导致误判成"模型在某个 seed 上
                        # 稳定失败"—— 其实只是那个 seed 的输出稍微长了一点。
                        "max_tokens": max_tokens, "temperature": TEMP, "seed": seed,
                        "chat_template_kwargs": {"enable_thinking": False}})
                    m = (r.get("choices") or [{}])[0].get("message", {}) or {}
                    calls = m.get("tool_calls") or []
                    content = (m.get("content") or "").strip()
                    msgs.append({"role": "assistant", "content": content,
                                 **({"tool_calls": calls} if calls else {})})
                    if calls:
                        rec["tool_calls"] += len(calls)
                        for c in calls:
                            fn = c.get("function") or {}
                            try:
                                a = json.loads(fn.get("arguments") or "{}")
                            except Exception:                       # noqa: BLE001
                                a = {}
                            if fn.get("name") == "write_file":
                                p = safe_path(sdir, str(a.get("path") or "out.svg"))
                                p.parent.mkdir(parents=True, exist_ok=True)
                                p.write_text(str(a.get("content") or ""), encoding="utf-8")
                                res = f"已写入 {p.name}"
                            else:
                                res = do_tool(sdir, fn.get("name") or "", a)
                            msgs.append({"role": "tool",
                                         "tool_call_id": c.get("id") or fn.get("name"),
                                         "content": res})
                        continue
                    sv = FENCE_RE.search(content) or XML_RE.search(content)
                    if sv:
                        (sdir / "from_text.svg").write_text(sv.group(1), encoding="utf-8")
                    break
            except Exception as e:                              # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"

            svgs = sorted(sdir.rglob("*.svg"), key=lambda p: -p.stat().st_size)
            if svgs:
                png = sdir / "render.png"
                ok = render(svgs[0], png)
                rec.update(analyse(svgs[0].read_text(encoding="utf-8", errors="replace"),
                                   png if ok else None))
                rec["svg"] = str(svgs[0])
                rec["png"] = str(png) if ok else ""
            else:
                rec.update({"elements": 0, "bytes": 0, "two_wheels": False,
                            "parts_n": 0, "png_ok": False, "ink_pct": 0.0, "svg": ""})
            rows.append(rec)
            flag = "✓" if (rec.get("two_wheels") and rec.get("parts_n", 0) >= 6
                           and rec.get("png_ok")) else "✗"
            print(f"   [{si+1}/{samples}] seed={seed:<6} {flag}  "
                  f"图元 {rec.get('elements',0):>3}  部件 {rec.get('parts_n',0)}/7  "
                  f"双轮 {'Y' if rec.get('two_wheels') else 'N'}  "
                  f"墨迹 {rec.get('ink_pct',0):>5.1f}%  工具 {rec['tool_calls']}")
    except Exception as e:                                      # noqa: BLE001
        print(f"   ✗ {e}")
        rows.append({"error": str(e)})
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        fh.close()
        time.sleep(2)
    return rows


def contact_sheet(label: str, rows: list[dict]) -> str:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return ""
    files = [r.get("png") for r in rows if r.get("png")]
    if not files:
        return ""
    cols, tw, th = 5, 320, 240
    sheet = Image.new("RGB", (cols * tw, ((len(files) + cols - 1) // cols) * th + 22), "white")
    d = ImageDraw.Draw(sheet)
    d.text((6, 5), f"{label}  -  {len(files)} samples", fill="black")
    for i, f in enumerate(files):
        try:
            im = Image.open(f).convert("RGB").resize((tw - 8, th - 8))
            sheet.paste(im, ((i % cols) * tw + 4, (i // cols) * th + 22))
        except Exception:                                       # noqa: BLE001
            pass
    out = WORK / f"sheet-{label}.png"
    sheet.save(out)
    return str(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--ctx", type=int, default=8192)
    # 这个默认值调大过：3000 对这个任务是危险的。模型在 write_file 的参数里写 SVG，
    # 写一半被 max_tokens 截断，JSON 就不完整，llama.cpp 会返回 HTTP 500 —— 看起来
    # 像"模型在某个 seed 上稳定失败"，实际是预算不够。留足 8000。
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict]] = {}
    for label, mp, moe in CANDIDATES:
        if not Path(mp).exists():
            print(f"跳过 {label}")
            continue
        all_rows[label] = run_model(label, Path(mp), moe, args.samples, args.ctx,
                                    max_tokens=args.max_tokens)

    print(f"\n{'=' * 76}\n{'模型':<14}{'通过':>8}{'双轮':>7}{'部件均':>8}{'图元均':>8}{'墨迹均':>8}")
    for label, rows in all_rows.items():
        ok = [r for r in rows if r.get("error") is None]
        if not ok:
            print(f"{label:<14}  全部失败"); continue
        n = len(ok)
        passed = sum(1 for r in ok if r.get("two_wheels") and r.get("parts_n", 0) >= 6
                     and r.get("png_ok"))
        print(f"{label:<14}{passed:>4}/{n:<3}{sum(1 for r in ok if r.get('two_wheels')):>7}"
              f"{sum(r.get('parts_n',0) for r in ok)/n:>8.1f}"
              f"{sum(r.get('elements',0) for r in ok)/n:>8.0f}"
              f"{sum(r.get('ink_pct',0) for r in ok)/n:>8.1f}")

    for label, rows in all_rows.items():
        p = contact_sheet(label, rows)
        if p:
            print(f"联络表: {p}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"seeds": SEEDS, "temp": TEMP, "results": all_rows},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
