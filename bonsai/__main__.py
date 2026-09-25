"""Entry point.

Order of operations matters here:

  1. claim a single instance, so double-clicking twice does not start two models;
  2. bind the interface *before* the model loads, so the user sees a progress
     screen instead of a dead icon while 5.5 GB arrives;
  3. only then boot the engine in the background.

Shutdown is graceful in the same order reversed -- the tunnel, then llama-server,
so the model file is never left memory-mapped.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser

from . import fetch
from .config import (APP_TITLE, APP_VERSION, Settings, default_data_dir,
                     free_port, resource_dir)
from .engine import Engine
from .server import App, make_server
from .tunnel import Tunnel

SINGLETON_PORT = 47_921        # fixed, only used as a "is another copy running?" probe


def selftest() -> int:
    """冻结自检：只在打包版里才会缺的东西，在这里一次性验掉。

    这个检查是有来历的，不是预防性妄想：
      * numpy 曾被写进 spec 的 excludes —— 源码一切正常，只有 exe 里选非 1.0
        权重的预设才会崩（lora_scale 顶层 import 它）；
      * 更早一次 ggml-cuda.dll 没打进引擎包，表现为启动即 0xC0000135。

    两次都是"源码跑得好、发布版炸"。所以打包流程每次跑一遍这个，让它自己发现。
    """
    from pathlib import Path as _P
    fails: list[str] = []
    lines: list[str] = []

    def check(name: str, fn) -> None:
        try:
            detail = fn()
            lines.append(f"  [OK ] {name}" + (f"   {detail}" if detail else ""))
        except Exception as e:                                  # noqa: BLE001
            fails.append(name)
            lines.append(f"  [FAIL] {name}   {type(e).__name__}: {e}")

    def _numpy():
        import numpy as np
        a = np.zeros((2, 3), dtype="float32")
        assert a.shape == (2, 3)
        return np.__version__

    def _pypdf():
        import pypdf
        return pypdf.__version__

    def _gguf():
        import gguf
        return getattr(gguf, "__version__", "ok")

    def _lora_scale():
        import struct
        from .lora_scale import GGML_F32, GGML_Q8_0, scale_tensor
        raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        got = struct.unpack("<4f", scale_tensor(raw, 4, GGML_F32, 0.5))
        assert abs(got[0] - 0.5) < 1e-6 and abs(got[3] - 2.0) < 1e-6, got
        # Q8_0: 缩放只动每块的 fp16 尺度 d，不动 int8 部分
        block = struct.pack("<e", 2.0) + bytes([1] * 32)
        out = scale_tensor(block, 32, GGML_Q8_0, 0.5)
        assert len(out) == 34, len(out)
        assert abs(struct.unpack("<e", out[:2])[0] - 1.0) < 1e-6
        assert out[2:] == bytes([1] * 32)
        return "F32 与 Q8_0 都对"

    def _knowledge():
        from . import knowledge as kb
        chunks = kb.chunk_text("第一段。\n\n第二段，稍微长一点点内容。")
        assert chunks, "切块返回空"
        idx = kb.BM25([kb.tokenize(t) for t in chunks])
        assert idx.n == len(chunks)
        hits = kb.Knowledge(Settings()).search("随便")     # 空库不该炸
        assert hits == []
        return f"{len(chunks)} 块"

    def _ui():
        base = resource_dir()
        missing = [p for p in ("ui/index.html", "ui/app.js", "ui/styles.css",
                               "assets/presets.json", "assets/loras.json")
                   if not (base / p).exists()]
        assert not missing, f"缺少 {missing}"
        return "界面与资源齐"

    def _bundled_lora():
        from .fetch import bundled_lora
        p = bundled_lora()
        assert p is not None and p.is_file(), "随包适配器不见了"
        return f"{p.name} {p.stat().st_size // 1024} KB"

    check("numpy", _numpy)
    check("pypdf（读 PDF）", _pypdf)
    check("gguf", _gguf)
    check("LoRA 权重缩放", _lora_scale)
    check("知识库（切块/BM25/空库）", _knowledge)
    check("界面与资源文件", _ui)
    check("随包去拒答适配器", _bundled_lora)

    print("\n".join(lines))
    print(f"\n自检结果：{'全部通过' if not fails else '失败 ' + str(len(fails)) + ' 项 —— ' + ', '.join(fails)}")
    return 0 if not fails else 1


def already_running() -> bool:
    """Cheap single-instance check: if the probe port answers, we are the second copy."""
    with socket.socket() as s:
        s.settimeout(0.4)
        try:
            s.connect(("127.0.0.1", SINGLETON_PORT))
            return True
        except OSError:
            return False


def hold_singleton() -> socket.socket | None:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", SINGLETON_PORT))
        s.listen(1)
        s.setblocking(False)
        return s
    except OSError:
        s.close()
        return None


def pick_public_port(settings: Settings) -> int:
    """Reuse the previous port when possible so bookmarks and LAN URLs keep working."""
    want = int(settings.get("_public_port") or 0)
    if want:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", want))
                return want
            except OSError:
                pass
    port = free_port()
    settings.set("_public_port", port)
    return port


def _utf8_when_piped() -> None:
    """被重定向到管道时，把标准输出改成 UTF-8。

    Windows 上 Python 按控制台的本地代码页写（中文机器是 936）。在终端里这是
    对的 —— 强制 UTF-8 反而会让 cp936 的控制台显示成乱码。但被重定向到管道时
    它仍然按 936 写，读的那一端按 UTF-8 解就是乱码：打包脚本读 --selftest 的
    结果时正是这样，明明全都通过了，看起来却是一堆问号。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                       # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser(prog="bonsai", description=f"{APP_TITLE} {APP_VERSION}")
    ap.add_argument("--no-window", action="store_true",
                    help="不打开应用窗口，只在终端里跑（调试用）")
    ap.add_argument("--browser", action="store_true",
                    help="用默认浏览器打开界面，而不是内置窗口")
    ap.add_argument("--data-dir", default="",
                    help="覆盖数据目录（调试用）")
    ap.add_argument("--port", type=int, default=0,
                    help="指定界面端口（调试用）")
    ap.add_argument("--selftest", action="store_true",
                    help="自检依赖与资源是否齐全，然后退出（打包流程用）")
    args = ap.parse_args()
    _utf8_when_piped()

    if args.selftest:
        return selftest()

    if already_running():
        print("已经有一个实例在运行了。如果没看到窗口，请检查任务栏或托盘。")
        return 0
    guard = hold_singleton()

    settings = Settings()
    if args.data_dir:
        # 先把 data_dir 设好，load() 才知道该读哪个文件；读完再把它钉回命令行
        # 给的值 —— 文件里可能记着另一个路径。
        #
        # 少了这两行，--data-dir 就只是换了个写盘位置：Settings() 不读文件，
        # 于是已有的令牌/预设/风格包全被忽略，一直跑内存里的默认值（包括每次
        # 现生成一个新令牌）。调试时会被这个坑很久。
        settings.set("data_dir", args.data_dir)
        settings.load()
        settings.set("data_dir", args.data_dir)
    settings.ensure_dirs()
    if not settings.path.exists():
        settings.save()

    engine = Engine(settings)
    tunnel = Tunnel(settings)
    app = App(settings, engine, tunnel)

    port = args.port or pick_public_port(settings)
    app.public_port = port
    app.lan_port = port
    server = make_server(app, "127.0.0.1", port)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://127.0.0.1:{port}"
    print(f"{APP_TITLE} {APP_VERSION}")
    print(f"  界面: {url}")
    print(f"  数据: {settings.data_dir}")
    if settings.get("remote_enabled"):
        print("  远程访问上次是开着的，启动后会重新生成一个新网址。")

    app.boot_async()

    def shutdown() -> None:
        try:
            tunnel.stop()
        finally:
            # 向量服务是第二个 llama-server，也要收掉，否则它会一直占着显存，
            # 用户下次开别的程序时才发现显卡被吃了。
            try:
                app.embed.stop()
            except Exception:                                   # noqa: BLE001
                pass
            engine.stop()
            try:
                server.shutdown()
            except Exception:                                   # noqa: BLE001
                pass
            if guard is not None:
                guard.close()

    # ---- present the interface -------------------------------------------
    if args.no_window:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        shutdown()
        return 0

    if not args.browser:
        try:
            import webview                                    # pywebview
            win = webview.create_window(APP_TITLE, url, width=1180, height=780,
                                        min_size=(880, 600), background_color="#0b0d10")
            webview.start()                                    # blocks until closed
            _ = win
            shutdown()
            return 0
        except Exception as e:                                  # noqa: BLE001
            print(f"内置窗口不可用（{e}），改用默认浏览器。")

    webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
