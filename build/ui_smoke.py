#!/usr/bin/env python3
"""用真浏览器（Edge 无头 + CDP）验证界面，重点是抓 JS 运行时异常。

为什么非做不可：后端可以完全正确而界面整个炸掉。设置面板里任何一个元素取不到
（$('...') 返回 null）都会抛异常，而 renderSettings 是在 openSettings 里同步调用
的 —— 一抛，整个设置面板白屏。语法检查和元素 id 扫描都只能证明"看起来对"，
证明不了"跑起来不炸"。

用法:
    python build/ui_smoke.py http://127.0.0.1:8129
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def find_edge() -> str | None:
    for c in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              shutil.which("msedge"), shutil.which("chrome")):
        if c and Path(c).exists():
            return c
    return None


class CDP:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.n = 0
        self.events: list[dict] = []

    async def send(self, method: str, params: dict | None = None) -> dict:
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method,
                                       "params": params or {}}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                return msg
            if "method" in msg:
                self.events.append(msg)

    async def drain(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            try:
                msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=0.4))
            except asyncio.TimeoutError:
                continue
            if "method" in msg:
                self.events.append(msg)

    async def js(self, expr: str):
        r = await self.send("Runtime.evaluate",
                            {"expression": expr, "returnByValue": True,
                             "awaitPromise": True})
        res = r.get("result", {})
        if "exceptionDetails" in res:
            return {"__error__": str(res["exceptionDetails"])}
        return res.get("result", {}).get("value")


async def run(url: str) -> int:
    import websockets

    edge = find_edge()
    if not edge:
        print("找不到 Edge/Chrome，跳过浏览器验证")
        return 2
    port = free_port()
    profile = Path(os.environ.get("TEMP", ".")) / f"ui-smoke-{port}"
    proc = subprocess.Popen(
        [edge, "--headless=new", f"--remote-debugging-port={port}",
         f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
         "--disable-gpu", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW)

    ws_url = ""
    for _ in range(60):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json/list", timeout=2) as r:
                tabs = json.loads(r.read())
            pages = [t for t in tabs if t.get("type") == "page"]
            if pages:
                ws_url = pages[0]["webSocketDebuggerUrl"]
                break
        except Exception:                                       # noqa: BLE001
            time.sleep(0.4)
    if not ws_url:
        proc.terminate()
        print("连不上 Edge 调试端口")
        return 2

    fails: list[str] = []
    async with websockets.connect(ws_url, max_size=32 << 20) as ws:
        cdp = CDP(ws)
        await cdp.send("Runtime.enable")
        await cdp.send("Log.enable")
        await cdp.send("Page.enable")
        await cdp.send("Page.navigate", {"url": url})
        await cdp.drain(9.0)

        errors = [e for e in cdp.events
                  if e["method"] == "Runtime.exceptionThrown"
                  or (e["method"] == "Log.entryAdded"
                      and e["params"]["entry"].get("level") == "error")
                  or (e["method"] == "Runtime.consoleAPICalled"
                      and e["params"].get("type") == "error")]

        def report(label: str, ok: bool, detail: str = "") -> None:
            print(f"  {'✓' if ok else '✗'} {label}" + (f"   {detail}" if detail else ""))
            if not ok:
                fails.append(label)

        print(f"\n用 {Path(edge).name} 打开 {url}\n")
        title = await cdp.js("document.title")
        report("页面标题", bool(title), str(title))

        # 应用界面：等引擎就绪才显示；就绪与否都该能开设置面板
        shown = await cdp.js(
            "(() => { const a=document.getElementById('app');"
            " return a && !a.classList.contains('hidden'); })()")
        print(f"    （主界面可见 = {shown}）")

        # 打开设置面板 —— 这一步会同步调用 renderKb()，有异常就在这里爆
        await cdp.js("document.getElementById('open-settings').click()")
        await cdp.drain(1.2)

        opened = await cdp.js(
            "!document.getElementById('settings').classList.contains('hidden')")
        report("设置面板能打开", bool(opened))

        for eid in ("kb-list", "kb-add-file", "kb-add-folder", "kb-toggle",
                    "kb-dense", "kb-state", "kb-dense-state", "kb-note"):
            exists = await cdp.js(f"!!document.getElementById('{eid}')")
            report(f"元素 #{eid}", bool(exists))

        note = await cdp.js("document.getElementById('kb-note').textContent")
        report("知识库说明文字已填充", bool(note and len(note) > 10),
               (note or "")[:56] + "…")

        dense_state = await cdp.js("document.getElementById('kb-dense-state').textContent")
        report("语义检索开关有文案", bool(dense_state), str(dense_state))

        rows = await cdp.js("document.querySelectorAll('#kb-list .kb-doc').length")
        print(f"    （当前资料条目数 = {rows}）")

        stage = await cdp.js("document.getElementById('kb-dense').disabled")
        report("开关可用性可读", stage is not None, f"disabled={stage}")

        report("没有 JS 运行时异常", not errors,
               f"{len(errors)} 条" if errors else "")

        # 光看"有 404"没用，要知道是哪个资源。逐个回请求一遍，把失败路径列出来。
        bad = await cdp.js("""
          (async () => {
            const urls = performance.getEntriesByType('resource').map(r => r.name);
            const out = [];
            for (const u of urls) {
              try { const r = await fetch(u, {method:'GET'}); if (!r.ok) out.push(r.status + ' ' + u); }
              catch (e) { out.push('ERR ' + u); }
            }
            return out;
          })()
        """)
        if bad:
            print(f"    失败资源 {len(bad)} 个：")
            for b in bad:
                print(f"      · {b}")
        for e in errors[:6]:
            if e["method"] == "Runtime.exceptionThrown":
                d = e["params"]["exceptionDetails"]
                print(f"      ✗ {d.get('text')}  {d.get('exception', {}).get('description', '')[:200]}")
            elif e["method"] == "Log.entryAdded":
                print(f"      ✗ {e['params']['entry'].get('text', '')[:200]}")
            else:
                args = e["params"].get("args", [])
                print(f"      ✗ {[a.get('value') or a.get('description') for a in args]}")

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    shutil.rmtree(profile, ignore_errors=True)

    print("\n" + "=" * 56)
    if fails:
        print(f"失败 {len(fails)} 项：" + "、".join(fails))
        return 1
    print("界面验证全部通过")
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8129"
    sys.exit(asyncio.run(run(target)))
