"""The single front door.

One HTTP server on one port does three jobs:

  /               the product's own interface (token injected for local callers)
  /static/*       its assets
  /app/*          settings, progress and status -- **localhost only**
  /v1/*           an OpenAI-compatible API, proxied to llama-server, token-gated

Routing remote traffic through here rather than exposing llama-server directly is
what makes the tunnel safe to switch on: settings cannot be changed from outside,
and every model request has to present the token.
"""

from __future__ import annotations

import http.client
import json
import mimetypes
import os
import socket
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import fetch
from .config import (APP_TITLE, APP_VERSION, MEMORY_TIERS, Settings,
                     local_ip, resource_dir)

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate",
              "proxy-authorization", "te", "trailers", "transfer-encoding",
              "upgrade"}

# Headers a proxy in front of us adds. cloudflared runs on this machine, so a
# tunneled request also arrives from 127.0.0.1 -- checking the peer address alone
# would treat the whole internet as "local" and hand out the token in the page.
FORWARDED = ("x-forwarded-for", "x-forwarded-proto", "x-forwarded-host",
             "cf-connecting-ip", "cf-ray", "cf-ipcountry", "x-real-ip")


def is_local(addr: str) -> bool:
    return addr in ("127.0.0.1", "::1", "localhost") or addr.startswith("127.")


class App:
    """Holds the long-lived objects and the background boot sequence."""

    def __init__(self, settings: Settings, engine, tunnel) -> None:
        self.settings = settings
        self.engine = engine
        self.tunnel = tunnel
        self.public_port = 0
        self.lan_port = 0
        self.last_error = ""
        self.boot_done = threading.Event()
        self._boot_lock = threading.Lock()

    # ------------------------------------------------------------------ boot
    def boot(self) -> None:
        """First run downloads; every run afterwards just starts the engine."""
        with self._boot_lock:
            try:
                fetch.PROGRESS.set(stage="checking", label="检查运行环境",
                                   done=0, total=0, detail="")
                self.settings.ensure_dirs()
                gpu = self.engine.gpu()
                variant = fetch.pick_engine_variant(self.settings, gpu)
                fetch.ensure_engine(self.settings, variant, fetch.PROGRESS)

                model = fetch.ensure_model(self.settings, fetch.PROGRESS)
                lora = fetch.bundled_lora()

                self.engine.start(model, lora)
                self.settings.update(first_run_done=True)
                self.settings.save()
                fetch.PROGRESS.set(stage="ready", label="就绪", done=0, total=0,
                                   detail="")
                self.last_error = ""
            except Exception as e:                              # noqa: BLE001
                self.last_error = str(e)
                fetch.PROGRESS.set(stage="error", label="启动失败", error=str(e))
            finally:
                self.boot_done.set()

    def boot_async(self) -> None:
        threading.Thread(target=self.boot, daemon=True).start()

    # ------------------------------------------------------------------ state
    def state(self, local: bool) -> dict:
        s = self.settings
        gpu = self.engine.gpu()
        return {
            "title": APP_TITLE,
            "version": APP_VERSION,
            "progress": fetch.PROGRESS.snapshot(),
            "engine": self.engine.status(),
            "tunnel": self.tunnel.status(),
            "error": self.last_error,
            "gpu": gpu,
            "context": s.context_size,
            "tiers": {k: {"label": v["label"], "hint": v["hint"], "ctx": v["ctx"]}
                      for k, v in MEMORY_TIERS.items()},
            "settings": {
                "memory_tier": s.get("memory_tier"),
                "data_dir": str(s.data_dir),
                "remote_enabled": bool(s.get("remote_enabled")),
            },
            # the token is a secret; only the machine it lives on gets to see it
            "token": s.get("api_token") if local else "",
            "local": local,
            "urls": {
                "local": f"http://127.0.0.1:{self.public_port}",
                "lan": f"http://{local_ip()}:{self.lan_port}" if self.lan_port else "",
            },
            "model_present": fetch.model_ready(s),
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "BonsaiLocal"
    protocol_version = "HTTP/1.1"

    app: App = None            # type: ignore[assignment]

    # ------------------------------------------------------------------ plumbing
    def log_message(self, fmt, *args):                          # noqa: A003
        pass                                                    # quiet by default

    @property
    def forwarded(self) -> bool:
        """True when something proxied this request, so it is not the local user.

        Fail-safe by construction: a client that forges X-Forwarded-For only makes
        itself look *remote*, which loses privileges rather than gaining them.
        """
        return any(self.headers.get(h) for h in FORWARDED)

    @property
    def local(self) -> bool:
        return is_local(self.client_address[0]) and not self.forwarded

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, data: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        """Parsed request body.

        The bytes are always drained by do_POST before routing, because an
        unconsumed body on a keep-alive HTTP/1.1 connection desynchronises the
        next request -- which shows up as a baffling 501 from the handler.
        """
        raw = getattr(self, "_raw", b"")
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:                                       # noqa: BLE001
            return {}

    # ------------------------------------------------------------------ routing
    def do_GET(self):                                           # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._serve_ui()
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path == "/app/state":
            return self._json(self.app.state(self.local))
        if path == "/app/qr":
            return self._qr()
        if path.startswith("/v1/"):
            return self._proxy_raw(None)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):                                          # noqa: N802
        # drain the body up front: see the note on _body()
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        self._raw = self.rfile.read(n) if 0 < n <= (8 << 20) else b""

        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/v1/"):
            return self._proxy_raw(self._raw)
        if path == "/app/settings":
            return self._set_settings()
        if path == "/app/token/rotate":
            return self._rotate_token()
        if path == "/app/remote":
            return self._toggle_remote()
        if path == "/app/restart":
            return self._restart()
        if path == "/app/pick-folder":
            return self._pick_folder()
        return self._json({"error": "not found"}, 404)

    # --------------------------------------------------------------------- UI
    def _ui_dir(self) -> Path:
        return resource_dir() / "ui"

    def _serve_ui(self) -> None:
        index = self._ui_dir() / "index.html"
        if not index.exists():
            return self._bytes(b"ui missing", "text/plain", 500)
        html = index.read_text(encoding="utf-8")
        token = self.app.settings.get("api_token") if self.local else ""
        # A distinct placeholder for the value: replacing the identifier
        # __BONSAI_TOKEN__ would rewrite `window.__BONSAI_TOKEN__` itself and
        # leave the page with invalid JavaScript.
        html = html.replace("__BONSAI_VALUE__", token or "")
        html = html.replace("__BONSAI_TITLE__", APP_TITLE)
        html = html.replace("__BONSAI_VERSION__", APP_VERSION)
        return self._bytes(html.encode("utf-8"), "text/html; charset=utf-8")

    def _serve_static(self, rel: str) -> None:
        base = self._ui_dir().resolve()
        target = (base / rel).resolve()
        if not str(target).startswith(str(base)) or not target.is_file():
            return self._bytes(b"not found", "text/plain", 404)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return self._bytes(target.read_bytes(), ctype)

    # ----------------------------------------------------------------- settings
    def _set_settings(self) -> None:
        if not self.local:
            return self._json({"error": "设置只能在运行这个程序的电脑上修改"}, 403)
        data = self._body()
        s = self.app.settings
        restart = False

        tier = data.get("memory_tier")
        if tier in MEMORY_TIERS and tier != s.get("memory_tier"):
            s.set("memory_tier", tier)
            restart = True

        new_dir = (data.get("data_dir") or "").strip()
        if new_dir and Path(new_dir) != s.data_dir:
            s.set("data_dir", new_dir)
            s.ensure_dirs()
            restart = True            # the model may not be where we are looking

        variant = data.get("engine_variant")
        if variant in ("auto", "cpu", "cuda-ada") and variant != s.get("engine_variant"):
            s.set("engine_variant", variant)
            restart = True

        s.save()
        if restart:
            self.app.boot_async()
        return self._json({"ok": True, "restarting": restart,
                           "state": self.app.state(self.local)})

    def _rotate_token(self) -> None:
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        tok = self.app.settings.rotate_token()
        return self._json({"ok": True, "token": tok})

    def _qr(self) -> None:
        """Rendered here rather than by an online service: the URL is a secret,
        and the machine may well be offline aside from the tunnel itself."""
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        data = (q.get("data") or [""])[0]
        if not data:
            return self._bytes(b"", "image/svg+xml", 400)
        try:
            import io

            import qrcode
            import qrcode.image.svg
            img = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage,
                              box_size=10, border=2)
            buf = io.BytesIO()
            img.save(buf)
            return self._bytes(buf.getvalue(), "image/svg+xml")
        except Exception as e:                                  # noqa: BLE001
            return self._bytes(f"<!-- {e} -->".encode(), "image/svg+xml", 500)

    def _pick_folder(self) -> None:
        """A native folder picker, because asking a person to type a path is the
        exact kind of configuration this product is supposed to avoid."""
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(title="选择存放模型和记录的文件夹")
            root.destroy()
            return self._json({"ok": bool(path), "path": path or ""})
        except Exception as e:                                  # noqa: BLE001
            return self._json({"ok": False, "error": f"无法打开文件夹选择窗口：{e}"})

    def _restart(self) -> None:
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        if self.app.tunnel.running:
            self.app.tunnel.stop()
        self.app.boot_done.clear()
        self.app.boot_async()
        return self._json({"ok": True})

    # ------------------------------------------------------------------- remote
    def _toggle_remote(self) -> None:
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        want = bool(self._body().get("enabled"))
        s = self.app.settings
        try:
            if want and not self.app.tunnel.running:
                fetch.PROGRESS.set(stage="starting", label="开启远程访问",
                                   done=0, total=0, detail="")
                url = self.app.tunnel.start(self.app.public_port, fetch.PROGRESS)
                fetch.PROGRESS.set(stage="ready", label="就绪", detail="")
                s.set("remote_enabled", True)
                s.save()
                return self._json({"ok": True, "url": url})
            if not want and self.app.tunnel.running:
                self.app.tunnel.stop()
                s.set("remote_enabled", False)
                s.save()
                return self._json({"ok": True, "url": ""})
            return self._json({"ok": True, "url": self.app.tunnel.url})
        except Exception as e:                                  # noqa: BLE001
            fetch.PROGRESS.set(stage="ready", label="就绪", detail="")
            return self._json({"ok": False, "error": str(e)}, 500)

    # --------------------------------------------------------------------- proxy
    def _authorised(self) -> bool:
        want = self.app.settings.get("api_token")
        if not want:
            return True
        auth = self.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer ") and auth[7:].strip() == want:
            return True
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        return (q.get("token") or [""])[0] == want

    def _proxy_raw(self, payload: bytes | None) -> None:
        if not self.app.engine.running:
            return self._json({"error": {"message": "模型还没就绪", "type": "unavailable"}},
                              503)
        if not self._authorised():
            return self._json({"error": {"message": "缺少或错误的 API 令牌",
                                         "type": "invalid_request_error",
                                         "code": "invalid_api_key"}}, 401)

        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_BY_HOP and k.lower() != "host"}
        conn = http.client.HTTPConnection("127.0.0.1", self.app.engine.port, timeout=600)
        try:
            conn.request(self.command, self.path, body=payload, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in HOP_BY_HOP or k.lower() == "content-length":
                    continue
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass                                                # client hung up
        except Exception as e:                                  # noqa: BLE001
            try:
                self._json({"error": {"message": f"转发失败: {e}"}}, 502)
            except Exception:                                   # noqa: BLE001
                pass
        finally:
            conn.close()


def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def lan_ip_and_port(port: int) -> int:
    return port
