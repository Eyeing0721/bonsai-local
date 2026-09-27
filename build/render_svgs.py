#!/usr/bin/env python3
"""把 pelican 测试产出的 SVG 渲染成 PNG。

单独拆出来是因为第一次内嵌在 test_pelican.py 里全失败了：从 Python 用
capture_output=True 启动 Edge 时，它挂到了已经在运行的 Edge 实例上，直接返回、
不写截图。手工在 PowerShell 里跑同一条命令却成功 —— 差别就在这里。

要的点：
  · 独立的 --user-data-dir，不跟用户正在用的 Edge 抢会话
  · 输出重定向到 DEVNULL，不用管道（管道会改变 Edge 的行为）
  · 截完轮询等文件落地，Edge 是异步的

用法: python build/render_svgs.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "build" / "work" / "pelican"
EDGES = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
         Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]


def render(svg: Path, png: Path, w: int = 820, h: int = 640) -> tuple[bool, str]:
    edge = next((e for e in EDGES if e.exists()), None)
    if edge is None:
        return False, "没找到 Edge"
    try:
        body = svg.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return False, f"读不到 SVG: {e}"

    html = png.with_suffix(".html")
    html.write_text(
        "<!doctype html><meta charset=utf-8>"
        f"<style>html,body{{margin:0;padding:0;background:#fff}}"
        f"svg{{display:block;width:{w}px;height:{h}px}}</style>" + body,
        encoding="utf-8")
    png.unlink(missing_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="edge-pe-"))
    cmd = [str(edge), "--headless=new", "--disable-gpu", "--no-sandbox",
           "--no-first-run", "--no-default-browser-check", "--hide-scrollbars",
           f"--user-data-dir={profile}", f"--window-size={w},{h}",
           f"--screenshot={png}", html.as_uri()]
    try:
        # stdout/stderr 直接丢弃：用管道会让 Edge 改行为，这是第一次失败的原因
        with open(Path(tempfile.gettempdir()) / "edge-render.log", "wb") as nullf:
            subprocess.run(cmd, stdout=nullf, stderr=subprocess.STDOUT, timeout=120)
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:                                      # noqa: BLE001
        shutil.rmtree(profile, ignore_errors=True)
        return False, f"{type(e).__name__}: {e}"

    ok = False
    for _ in range(30):                     # Edge 是异步的，等它落地
        if png.exists() and png.stat().st_size > 200:
            ok = True
            break
        time.sleep(0.4)
    shutil.rmtree(profile, ignore_errors=True)
    return ok, (f"{png.stat().st_size:,} B" if ok else "没写出文件")


def main() -> int:
    if not WORK.exists():
        print(f"没有产出目录 {WORK}")
        return 1
    found = 0
    for label_dir in sorted(WORK.iterdir()):
        if not label_dir.is_dir():
            continue
        svgs = sorted(label_dir.rglob("*.svg"), key=lambda p: -p.stat().st_size)
        if not svgs:
            print(f"  {label_dir.name:<16} 没有 SVG")
            continue
        svg = svgs[0]
        png = label_dir / "render.png"
        ok, info = render(svg, png)
        found += 1 if ok else 0
        print(f"  {label_dir.name:<16} {svg.name:<24} {svg.stat().st_size:>7,} B  ->  "
              f"{'PNG ' + info if ok else '失败: ' + info}")
        if ok:
            print(f"      {png}")
    print(f"\n{found} 个渲染成功")
    return 0


if __name__ == "__main__":
    sys.exit(main())
