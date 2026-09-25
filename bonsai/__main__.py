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
from .config import APP_TITLE, APP_VERSION, Settings, default_data_dir, free_port
from .engine import Engine
from .server import App, make_server
from .tunnel import Tunnel

SINGLETON_PORT = 47_921        # fixed, only used as a "is another copy running?" probe


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
    args = ap.parse_args()

    if already_running():
        print("已经有一个实例在运行了。如果没看到窗口，请检查任务栏或托盘。")
        return 0
    guard = hold_singleton()

    settings = Settings()
    if args.data_dir:
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
