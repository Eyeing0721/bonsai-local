#!/usr/bin/env python3
"""SVG 测试组：和鹈鹕骑车同类，但有三项是**可机检的几何正确性**。

鹈鹕那个测试只能靠肉眼看"像不像"。这里补上几个答案唯一的题，让对错不取决于
我的主观判断：

  clock-345   画一个指向 3:45 的指针时钟。
              可机检：两条最长且共端点的线段，长的应指向 270°（分针指 9），
              短的应指向 112.5°（3 点 45 分时时针的位置）。
  barchart    三个柱 A=30 B=50 C=20。
              可机检：三根柱子的高度比应当是 5:3:2（容差内）。
  target      三个同心圆，半径 150/100/50。
              可机检：三个圆心重合，半径比 3:2:1。

另外两项（猫弹钢琴 / 机器人在公园长椅旁浇花）是纯视觉的，和鹈鹕同类。

用法:
    python build/test_svg_battery.py --samples 2
"""
from __future__ import annotations

import argparse
import json
import math
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
WORK = ROOT / "build" / "work" / "svg-battery"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

SYSTEM = ("你是一个会使用工具的助手。需要输出较长内容（比如 SVG）时，"
          "用 write_file 工具保存成文件，而不是把全部内容写在回答里。")
TOOLS = [{"type": "function", "function": {
    "name": "write_file", "description": "把文件写入工作目录。",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string"},
                                  "content": {"type": "string"}},
                   "required": ["path", "content"]}}}]

BATTERY: list[tuple[str, str]] = [
    ("cat-piano", "Generate an SVG of a cat playing a grand piano."),
    ("clock-345", "Generate an SVG of an analog clock showing exactly 3:45. "
                  "The minute hand must point at the 9 and the hour hand must be "
                  "three quarters of the way from the 3 to the 4."),
    ("barchart", "Generate an SVG of a bar chart with three vertical bars labeled "
                 "A, B and C, with values 30, 50 and 20."),
    ("target", "Generate an SVG of a target made of three concentric circles with "
               "radii 150, 100 and 50 pixels."),
    ("robot-bench", "Generate an SVG of a robot watering a flower next to a park bench."),
]

SEEDS = [11, 202]
TEMP = 0.4
CANDIDATES = [
    ("三值1.75bit", r"E:\models\bonsai\Ternary-Bonsai-2-27B-PTQ1_0.gguf", None),
    ("A3B-MoE-Q3", r"E:\models\bonsai\alts\Qwen3.6-35B-A3B-uncensored-Q3_K_M.gguf", 14),
]

NUM = r"(-?\d+(?:\.\d+)?)"

# ---------------------------------------------------------------- 几何解析
# 关键：必须算 transform。模型画时钟很自然会写
#     <line x1="200" y1="200" x2="200" y2="60" transform="rotate(112.5 200 200)"/>
# 也就是"竖着画再旋转"。第一版只读原始坐标，于是把两根指针都读成指向 12 点，
# 判成"模型画错了" —— 实际上它画的是对的。图片一看就露馅。
#
# 用 ElementTree 遍历（而不是正则扫 <line>），把祖先链上的变换累积成仿射矩阵。
_M = tuple[float, float, float, float, float, float]      # a b c d e f


def _mul(m: _M, n: _M) -> _M:
    a1, b1, c1, d1, e1, f1 = m
    a2, b2, c2, d2, e2, f2 = n
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def _apply(m: _M, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def _parse_transform(text: str | None) -> _M:
    """只实现 SVG 里画图常用的几种；认不出来的部分跳过（不猜）。"""
    m: _M = (1, 0, 0, 1, 0, 0)
    if not text:
        return m
    for name, argstr in re.findall(r"(\w+)\s*\(([^)]*)\)", text):
        args = [float(v) for v in re.findall(NUM, argstr)]
        n: _M | None = None
        if name == "rotate" and len(args) >= 1:
            th = math.radians(args[0])
            cs, sn = math.cos(th), math.sin(th)
            r: _M = (cs, sn, -sn, cs, 0, 0)
            if len(args) >= 3:                              # rotate(a cx cy)
                cx, cy = args[1], args[2]
                n = _mul(_mul((1, 0, 0, 1, cx, cy), r), (1, 0, 0, 1, -cx, -cy))
            else:
                n = r
        elif name == "translate" and args:
            n = (1, 0, 0, 1, args[0], args[1] if len(args) > 1 else 0.0)
        elif name == "scale" and args:
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            n = (sx, 0, 0, sy, 0, 0)
        elif name == "matrix" and len(args) >= 6:
            n = tuple(args[:6])                             # type: ignore[assignment]
        if n is not None:
            m = _mul(m, n)
    return m


def svg_well_formed(svg: str) -> bool:
    """严格 XML 能不能解析。

    实测踩到的：模型偶尔会多吐一个 </svg>，浏览器宽容照样渲染，严格解析器直接拒。
    这本身是模型的缺陷（"生成了 SVG"和"生成了合法 SVG"不是一回事），所以要单独记
    一笔，而不是悄悄当成"没画出东西"。
    """
    import xml.etree.ElementTree as ET
    try:
        ET.fromstring(svg)
        return True
    except Exception:                                           # noqa: BLE001
        return False


_TAG_RE = re.compile(r"<(\w+)\b([^>]*?)/?>", re.S)
_ATTR_RE = re.compile(r'([\w:-]+)\s*=\s*"([^"]*)"')


def _walk(svg: str):
    """产出 (tag, attrs, 累积矩阵)。

    XML 解析失败时退到"正则扫标签 + 不认变换"的尽力模式 —— 返回空会让畸形 SVG
    看起来像"什么都没画"，那是另一种误判（第一版就静默返回过空）。
    """
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(svg)
    except Exception:                                           # noqa: BLE001
        for m in _TAG_RE.finditer(svg):
            attrs = {k: v for k, v in _ATTR_RE.findall(m.group(2))}
            yield m.group(1).lower(), attrs, _parse_transform(attrs.get("transform"))
        return
    stack = [(root, (1, 0, 0, 1, 0, 0))]
    while stack:
        el, m = stack.pop()
        m = _mul(m, _parse_transform(el.get("transform")))
        tag = el.tag.split("}")[-1].lower()
        yield tag, el.attrib, m
        for child in list(el):
            stack.append((child, m))


def _f(attrs: dict, key: str) -> float | None:
    try:
        return float(attrs[key])
    except (KeyError, TypeError, ValueError):
        return None


def lines_of(svg: str) -> list[tuple[float, float, float, float]]:
    out = []
    for tag, a, m in _walk(svg):
        if tag != "line":
            continue
        v = [_f(a, k) for k in ("x1", "y1", "x2", "y2")]
        if any(x is None for x in v):
            continue
        p1 = _apply(m, v[0], v[1])                          # type: ignore[arg-type]
        p2 = _apply(m, v[2], v[3])                          # type: ignore[arg-type]
        out.append((p1[0], p1[1], p2[0], p2[1]))
    return out


def circles_of(svg: str) -> list[tuple[float, float, float]]:
    out = []
    for tag, a, m in _walk(svg):
        if tag != "circle":
            continue
        cx, cy, r = _f(a, "cx"), _f(a, "cy"), _f(a, "r")
        if None in (cx, cy, r):
            continue
        # 只用旋转/平移时半径可照搬；带非均匀缩放就跳过（免得算错）
        sx = math.hypot(m[0], m[1])
        sy = math.hypot(m[2], m[3])
        if abs(sx - sy) > 1e-6:
            continue
        c = _apply(m, cx, cy)                               # type: ignore[arg-type]
        out.append((c[0], c[1], r * sx))                    # type: ignore[operator]
    return out


def rects_of(svg: str) -> list[tuple[float, float, float, float]]:
    out = []
    for tag, a, m in _walk(svg):
        if tag != "rect":
            continue
        x, y, w, h = (_f(a, "x") or 0.0, _f(a, "y") or 0.0,
                      _f(a, "width"), _f(a, "height"))
        if w is None or h is None:
            continue
        # 柱状图关心的是"高"；有旋转就没法直接比，跳过
        if abs(m[1]) > 1e-6 or abs(m[2]) > 1e-6:
            continue
        sh = math.hypot(m[2], m[3])
        p = _apply(m, x, y)
        out.append((p[0], p[1], w * math.hypot(m[0], m[1]), h * sh))
    return out


def ang_from_12(cx: float, cy: float, x: float, y: float) -> float:
    """从 12 点方向顺时针算的角度（SVG 的 y 轴朝下）。"""
    a = math.degrees(math.atan2(x - cx, -(y - cy)))
    return (a + 360.0) % 360.0


def ang_close(got: float, want: float, tol: float = 20.0) -> bool:
    d = abs((got - want + 180) % 360 - 180)
    return d <= tol


def check_clock(svg: str) -> dict:
    """两条共端点的长线段 = 时针和分针。长的指 270°，短的指 112.5°。"""
    ls = lines_of(svg)
    best = None
    for i in range(len(ls)):
        for j in range(i + 1, len(ls)):
            x1, y1, x2, y2 = ls[i]
            a1, b1, a2, b2 = ls[j]
            for (px, py), (qx, qy) in (((x1, y1), (x2, y2)), ((x2, y2), (x1, y1))):
                for (rx, ry), (sx, sy) in (((a1, b1), (a2, b2)), ((a2, b2), (a1, b1))):
                    if abs(px - rx) < 6 and abs(py - ry) < 6:   # 共用圆心
                        l1 = math.hypot(qx - px, qy - py)
                        l2 = math.hypot(sx - rx, sy - ry)
                        if l1 < 20 or l2 < 20:
                            continue
                        if best is None or l1 + l2 > best[0]:
                            best = (l1 + l2, px, py, qx, qy, sx, sy, l1, l2)
    if best is None:
        return {"hands": 0, "minute_ok": False, "hour_ok": False}
    _, cx, cy, qx, qy, sx, sy, l1, l2 = best
    long_ang = ang_from_12(cx, cy, qx, qy) if l1 >= l2 else ang_from_12(cx, cy, sx, sy)
    short_ang = ang_from_12(cx, cy, sx, sy) if l1 >= l2 else ang_from_12(cx, cy, qx, qy)
    return {"hands": 2, "minute_angle": round(long_ang, 1),
            "hour_angle": round(short_ang, 1),
            "minute_ok": ang_close(long_ang, 270.0),
            "hour_ok": ang_close(short_ang, 112.5)}


def check_bars(svg: str) -> dict:
    """按位置把柱子映射到 A/B/C，检查高度比是不是 30:50:20。

    踩过的两个坑：
      · 按高度排序会把 A/B/C 的对应关系丢掉 —— 而"哪根柱对应哪个标签"本身就是
        题目的一部分。3:2:1 和 5:3:2 归一化之后只差 0.33，只有带上位置才分得开。
      · 容差给到 1.0（20%）时 3:2:1 会被判成通过。假阳性比漏报更坏：它会让人
        以为模型会做这道题。
    """
    rs = [r for r in rects_of(svg) if r[3] >= 15 and r[2] >= 5]
    if len(rs) >= 4:                    # 画布/背景通常是面积最大的那个
        rs.sort(key=lambda r: -(r[2] * r[3]))
        rs = rs[1:]
    if len(rs) < 3:
        return {"bars": len(rs), "ratio_ok": False}
    rs = sorted(rs, key=lambda r: r[0])[:3]          # 按 x 排 -> A, B, C
    ha, hb, hc = rs[0][3], rs[1][3], rs[2][3]
    if hb <= 0:
        return {"bars": len(rs), "ratio_ok": False}
    got = [ha / hb * 5.0, 5.0, hc / hb * 5.0]        # 以 B 为 5 归一
    want = [3.0, 5.0, 2.0]
    err = max(abs(g - w) for g, w in zip(got, want))
    return {"bars": len(rs), "heights_abc": [ha, hb, hc],
            "ratio_abc": [round(x, 2) for x in got], "ratio_err": round(err, 2),
            "ratio_ok": err <= 0.4}


def check_target(svg: str) -> dict:
    """三个同心圆，半径比 3:2:1。"""
    cs = [c for c in circles_of(svg) if c[2] >= 8]
    cs.sort(key=lambda c: -c[2])
    if len(cs) < 3:
        return {"circles": len(cs), "concentric_ok": False, "ratio_ok": False}
    top = cs[:3]
    rmax = top[0][2]
    dx = max(c[0] for c in top) - min(c[0] for c in top)
    dy = max(c[1] for c in top) - min(c[1] for c in top)
    concentric = dx <= 0.08 * rmax and dy <= 0.08 * rmax
    ratio = [c[2] / rmax for c in top]
    ratio_ok = max(abs(g - w) for g, w in zip(ratio, (1.0, 2 / 3, 1 / 3))) <= 0.10
    return {"circles": len(cs), "radii": [round(c[2], 1) for c in top],
            "concentric_ok": concentric, "ratio_ok": ratio_ok}


CHECKERS = {"clock-345": check_clock, "barchart": check_bars, "target": check_target}


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


def safe(work: Path, rel: str) -> Path:
    p = (work / rel.strip().lstrip("/\\")).resolve()
    if not str(p).startswith(str(work.resolve())):
        raise ValueError("越界")
    return p


def render(svg: Path, png: Path, w=520, h=420) -> bool:
    edge = next((e for e in EDGES if e.exists()), None)
    if edge is None:
        return False
    html = png.with_suffix(".html")
    html.write_text("<!doctype html><meta charset=utf-8>"
                    f"<style>html,body{{margin:0;background:#fff}}"
                    f"svg{{display:block;width:{w}px;height:{h}px}}</style>"
                    + svg.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    png.unlink(missing_ok=True)
    prof = Path(tempfile.mkdtemp(prefix="edge-sb-"))
    try:
        with open(Path(tempfile.gettempdir()) / "edge-sb.log", "wb") as nf:
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


def run_model(label, model: Path, moe, samples: int, max_tokens: int) -> dict:
    work = WORK / label
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    port = free_port()
    log = ROOT / "build" / "bench-logs" / f"svg-battery-{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w", encoding="utf-8", errors="replace")
    cmd = [str(BIN / "llama-server.exe"), "-m", str(model), "--host", "127.0.0.1",
           "--port", str(port), "-c", "8192", "-ngl", "99", "-fa", "on", "-np", "1",
           "--jinja", "--no-warmup", "--no-webui", "--alias", "m"]
    if moe:
        cmd += ["--n-cpu-moe", str(moe)]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(BIN), str(CUDA), env.get("PATH", "")])
    print(f"\n{'=' * 76}\n── {label} ──  {len(BATTERY)} 题 × {samples} 样本")
    proc = subprocess.Popen(cmd, cwd=str(BIN), stdout=fh, stderr=subprocess.STDOUT,
                            env=env, creationflags=CREATE_NO_WINDOW)
    out: dict = {}
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

        for pid, prompt in BATTERY:
            rows = []
            for si in range(samples):
                seed = SEEDS[si % len(SEEDS)]
                sdir = work / f"{pid}-s{si+1}"
                sdir.mkdir(parents=True, exist_ok=True)
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt}]
                rec: dict = {"prompt": prompt, "seed": seed, "tools": 0}
                try:
                    for _turn in range(6):
                        r = api(port, "/v1/chat/completions", {
                            "messages": msgs, "tools": TOOLS, "tool_choice": "auto",
                            "max_tokens": max_tokens, "temperature": TEMP, "seed": seed,
                            "chat_template_kwargs": {"enable_thinking": False}})
                        m = (r.get("choices") or [{}])[0].get("message", {}) or {}
                        calls = m.get("tool_calls") or []
                        msgs.append({"role": "assistant", "content": m.get("content") or "",
                                     **({"tool_calls": calls} if calls else {})})
                        if calls:
                            rec["tools"] += len(calls)
                            for c in calls:
                                fn = c.get("function") or {}
                                try:
                                    a = json.loads(fn.get("arguments") or "{}")
                                except Exception:                   # noqa: BLE001
                                    a = {}
                                if fn.get("name") == "write_file":
                                    p = safe(sdir, str(a.get("path") or "out.svg"))
                                    p.parent.mkdir(parents=True, exist_ok=True)
                                    p.write_text(str(a.get("content") or ""), encoding="utf-8")
                                    res = f"已写入 {p.name}"
                                else:
                                    res = "未知工具"
                                msgs.append({"role": "tool",
                                             "tool_call_id": c.get("id") or "t",
                                             "content": res})
                            continue
                        txt = m.get("content") or ""
                        mm = re.search(r"<svg\b.*?</svg>", txt, re.I | re.S)
                        if mm:
                            (sdir / "from_text.svg").write_text(mm.group(0), encoding="utf-8")
                        break
                except Exception as e:                          # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"

                svgs = sorted(sdir.rglob("*.svg"), key=lambda p: -p.stat().st_size)
                if svgs:
                    body = svgs[0].read_text(encoding="utf-8", errors="replace")
                    rec["bytes"] = len(body.encode())
                    rec["elements"] = len(re.findall(
                        r"<(circle|ellipse|path|rect|line|polygon|polyline|g|text)\b", body, re.I))
                    png = sdir / "render.png"
                    rec["rendered"] = render(svgs[0], png)
                    rec["svg"] = str(svgs[0])
                    rec["png"] = str(png) if rec["rendered"] else ""
                    fn = CHECKERS.get(pid)
                    if fn:
                        rec.update(fn(body))
                else:
                    rec.update({"bytes": 0, "elements": 0, "rendered": False,
                                "svg": "", "png": ""})
                    if pid in CHECKERS:
                        rec.update(CHECKERS[pid](""))
                rows.append(rec)
            out[pid] = rows
            marks = []
            for r in rows:
                fn = CHECKERS.get(pid)
                if fn is None:
                    marks.append("✓" if r.get("rendered") else "✗")
                else:
                    keys = [k for k in r if k.endswith("_ok")]
                    marks.append("✓" if all(r.get(k) for k in keys) else "✗")
            print(f"   {pid:<12} {' '.join(marks)}  "
                  f"图元 {sum(r.get('elements',0) for r in rows)//max(1,len(rows))}")
            for r in rows:
                det = {k: v for k, v in r.items()
                       if k.endswith("_ok") or k in ("minute_angle", "hour_angle",
                                                     "ratio", "radii", "circles", "bars", "hands")}
                if det:
                    print(f"        seed={r['seed']:<5} {det}")
    except Exception as e:                                      # noqa: BLE001
        print(f"   ✗ {e}")
        out["error"] = str(e)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)

    all_res = {}
    for label, mp, moe in CANDIDATES:
        if not Path(mp).exists():
            print(f"跳过 {label}")
            continue
        all_res[label] = run_model(label, Path(mp), moe, args.samples, args.max_tokens)

    print(f"\n{'=' * 76}\n机检项通过率（visual 项只统计有没有画出可渲染的图）")
    for label, res in all_res.items():
        if "error" in res and len(res) == 1:
            print(f"  {label}: 失败")
            continue
        print(f"  [{label}]")
        for pid, _ in BATTERY:
            rows = res.get(pid) or []
            if not rows:
                continue
            fn = CHECKERS.get(pid)
            if fn is None:
                ok = sum(1 for r in rows if r.get("rendered"))
                print(f"     {pid:<12} 渲染通过 {ok}/{len(rows)}   (视觉题)")
            else:
                ok = sum(1 for r in rows
                         if all(r.get(k) for k in r if k.endswith("_ok")))
                print(f"     {pid:<12} 机检通过 {ok}/{len(rows)}")

    # 联络表
    try:
        from PIL import Image, ImageDraw
        for label in all_res:
            pngs = sorted((WORK / label).rglob("render.png"))
            if not pngs:
                continue
            cols, tw, th = 5, 270, 225
            rows_n = (len(pngs) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * tw, rows_n * th + 20), "white")
            d = ImageDraw.Draw(sheet)
            d.text((6, 4), f"{label}  {len(pngs)} images", fill="black")
            for i, p in enumerate(pngs):
                try:
                    im = Image.open(p).convert("RGB").resize((tw - 8, th - 8))
                    sheet.paste(im, ((i % cols) * tw + 4, (i // cols) * th + 20))
                except Exception:                               # noqa: BLE001
                    pass
            out = WORK / f"sheet-{label}.png"
            sheet.save(out)
            print(f"  联络表: {out}")
    except Exception as e:                                      # noqa: BLE001
        print(f"  联络表生成失败: {e}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(all_res, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        print(f"写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
