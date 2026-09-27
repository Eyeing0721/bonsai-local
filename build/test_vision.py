#!/usr/bin/env python3
"""视觉能力测试：模型到底能不能"看图"。

为什么要两张图 + 一个无图对照：只发一张图、得到一段像样的描述，说明不了任何事
—— 语言模型完全可以靠"图里大概是些什么"编一段通顺的话。要证明它真在看，就得让
同一套问题在不同图上得到**不同且对应**的答案。

  A 图：红圆 + 蓝方 + 绿三角 + 文字 BONSAI-42
  B 图：黄三角 + 紫圆 + 橙方 + 文字 DELTA-77
  对照：不发图，问同样的问题

判据是"能不能把 A 和 B 分开"和"能不能读出图上的字"，不是描述流不流畅。

用法:
    python build/test_vision.py --model X.gguf --mmproj Y.gguf
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
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
WORK = ROOT / "build" / "work" / "vision"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def make_image(path: Path, shapes, label: str) -> None:
    """画一张内容确定的图。用大号字体，免得模型是被字号劝退的。"""
    from PIL import Image, ImageDraw, ImageFont
    W = H = 560
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for kind, color, xy in shapes:
        x, y, r = xy
        if kind == "circle":
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)
        elif kind == "square":
            d.rectangle([x - r, y - r, x + r, y + r], fill=color)
        elif kind == "triangle":
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=color)
    font = None
    for cand in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if Path(cand).exists():
            font = ImageFont.truetype(cand, 64)
            break
    if font is None:
        font = ImageFont.load_default()
    tb = d.textbbox((0, 0), label, font=font)
    d.text(((W - (tb[2] - tb[0])) / 2 - tb[0], H - 100), label, fill="black", font=font)
    im.save(path)


A_SHAPES = [("circle", "#E53935", (150, 150, 80)),
            ("square", "#1E88E5", (410, 150, 75)),
            ("triangle", "#43A047", (280, 330, 80))]
B_SHAPES = [("triangle", "#FDD835", (150, 150, 80)),
            ("circle", "#8E24AA", (410, 150, 75)),
            ("square", "#FB8C00", (280, 330, 80))]

Q_SHAPES = "图中有哪些几何形状？分别是什么颜色？只列出来，不要解释。"
Q_TEXT = "图中下方那行文字是什么？逐字读出来，不要解释。"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def api(port, path, payload=None, timeout=900.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def ask(port: int, question: str, image: Path | None, max_tokens: int = 300) -> str:
    content: list = [{"type": "text", "text": question}]
    if image is not None:
        content.insert(0, {"type": "image_url",
                           "image_url": {"url": f"data:image/png;base64,{b64(image)}"}})
    r = api(port, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0.0, "seed": 3,
        "chat_template_kwargs": {"enable_thinking": False}})
    return ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def hit(text: str, *words: str) -> int:
    """命中计数。

    必须先把空白去掉再比："逐字读出来"这类指令会让模型输出 "B O N S A I - 4 2"，
    直接子串匹配 "BONSAI" 会假阴性 —— 第一次跑就这样误判了两项。
    """
    t = "".join((text or "").split())
    return sum(1 for w in words if "".join(w.split()).lower() in t.lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"E:\models\bonsai\Ternary-Bonsai-2-27B-PTQ1_0.gguf")
    ap.add_argument("--mmproj", default=r"E:\models\bonsai\mmproj\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    model, mmproj = Path(args.model), Path(args.mmproj)
    for p in (model, mmproj):
        if not p.exists():
            print(f"缺文件：{p}")
            return 1
    WORK.mkdir(parents=True, exist_ok=True)
    img_a, img_b = WORK / "A.png", WORK / "B.png"
    make_image(img_a, A_SHAPES, "BONSAI-42")
    make_image(img_b, B_SHAPES, "DELTA-77")
    print(f"测试图: {img_a} / {img_b}")

    port = free_port()
    log = ROOT / "build" / "bench-logs" / "vision.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(BIN / "llama-server.exe"), "-m", str(model),
           "--mmproj", str(mmproj), "--host", "127.0.0.1", "--port", str(port),
           "-c", str(args.ctx), "-ngl", "99", "-fa", "on", "-np", "1",
           "--jinja", "--no-warmup", "--no-webui", "--alias", "m"]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])
    print(f"启动引擎（含 --mmproj {mmproj.name}）...")
    proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh, stderr=subprocess.STDOUT,
                            env=env, creationflags=CREATE_NO_WINDOW)
    out: dict = {"model": str(model), "mmproj": str(mmproj), "results": {}}
    try:
        t0 = time.time()
        while time.time() - t0 < 420:
            if proc.poll() is not None:
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
                print("  引擎退出:")
                for x in tail:
                    print("    " + x.strip()[:150])
                return 1
            try:
                if api(port, "/health", timeout=3).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(0.8)
        else:
            print("  就绪超时")
            return 1
        print(f"  就绪 {time.time()-t0:.0f}s")

        for name, img, q in (("A-形状", img_a, Q_SHAPES), ("A-文字", img_a, Q_TEXT),
                             ("B-形状", img_b, Q_SHAPES), ("B-文字", img_b, Q_TEXT),
                             ("对照-无图", None, Q_TEXT)):
            try:
                ans = ask(port, q, img)
            except Exception as e:                              # noqa: BLE001
                ans = f"<失败 {type(e).__name__}: {e}>"
            out["results"][name] = {"question": q, "answer": ans,
                                    "with_image": img is not None}
            print(f"  [{name}] {ans[:110].replace(chr(10), ' ')}")

        r = out["results"]
        checks = {
            "A 认出红色": hit(r["A-形状"]["answer"], "红", "red") > 0,
            "A 认出蓝色": hit(r["A-形状"]["answer"], "蓝", "blue") > 0,
            "A 认出绿色": hit(r["A-形状"]["answer"], "绿", "green") > 0,
            "A 读出 BONSAI-42": hit(r["A-文字"]["answer"], "BONSAI", "42") >= 2,
            "B 认出黄色": hit(r["B-形状"]["answer"], "黄", "yellow") > 0,
            "B 认出紫色": hit(r["B-形状"]["answer"], "紫", "purple") > 0,
            "B 认出橙色": hit(r["B-形状"]["answer"], "橙", "orange", "橘") > 0,
            "B 读出 DELTA-77": hit(r["B-文字"]["answer"], "DELTA", "77") >= 2,
            "A/B 答案不同（真的在看图）":
                r["A-文字"]["answer"].strip() != r["B-文字"]["answer"].strip(),
            "无图时读不出文字（对照成立）":
                hit(r["对照-无图"]["answer"], "BONSAI", "42", "DELTA", "77") == 0,
        }
        out["checks"] = checks
        print()
        for k, v in checks.items():
            print(f"  [{'OK ' if v else 'FAIL'}] {k}")
        ok = sum(checks.values())
        print(f"\n  {ok}/{len(checks)} 项通过")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                proc.kill()
        fh.close()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        print(f"  写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
