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

from . import embed as embed_mod
from . import fetch
from . import knowledge as kb_mod
from . import loras as lora_mod
from .config import (APP_TITLE, APP_VERSION, MEMORY_TIERS, Settings,
                     gpu_summary, local_ip, resource_dir)

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
        self.kb = kb_mod.Knowledge(settings)
        self.embed = embed_mod.EmbedServer(settings)
        self.kb_busy = ""
        self.kb_error = ""
        self.last_hits: list[dict] = []
        self._kb_lock = threading.Lock()

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
                self.engine.start(model, self.lora_paths())
                self.settings.update(first_run_done=True)
                self.settings.save()
                self.kb.load()
                if self.settings.get("kb_dense") and embed_mod.embed_ready(self.settings):
                    # 上次用过向量检索，这次后台把它接回来，不挡启动
                    threading.Thread(target=self._kb_warm_embed, daemon=True).start()
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

    # ------------------------------------------------------------------ loras
    def lora_paths(self) -> list:
        """交给引擎的 LoRA 链。

        去拒答适配器排在最前面，但它不再是无条件的 —— 「默认」预设不挂它，
        因为那个预设的卖点就是"最接近模型原本的样子"。其余预设都挂：风格包
        负责让模型顺着你说，拒答负责把它拽回去，两个力对着拉的结果是风格
        时有时无。后面接当前预设挑好的风格包（权重已经烤进文件里，因为
        --lora-scaled 的冒号解析在 Windows 路径上会直接报错）。
        """
        out: list = []
        preset_id = self.settings.get("preset") or "default"
        if lora_mod.preset_uses_core(self.settings, preset_id):
            core = fetch.bundled_lora()
            if core is not None:
                out.append(core)
        for p in lora_mod.resolve_preset(self.settings, preset_id, fetch.PROGRESS):
            if p not in out:
                out.append(p)
        return out

    def lora_state(self) -> dict:
        core = fetch.bundled_lora()
        preset_id = self.settings.get("preset") or "default"
        return {
            "core": {"name": "去拒答适配器",
                     "size": core.stat().st_size if core else 0,
                     "active": lora_mod.preset_uses_core(self.settings, preset_id),
                     "always_on": False},
            "items": [l.as_dict() for l in lora_mod.scan(self.settings)],
            "enabled": lora_mod.enabled_ids(self.settings),
            "presets": lora_mod.preset_state(self.settings),
        }

    # -------------------------------------------------------------- knowledge
    def kb_embed_fn(self):
        """检索时用的向量函数。返回 None 表示这次只用关键词检索。

        刻意做成"能降级就降级"：向量服务没起来、模型没下完、显存不够，都不该
        让知识库整个不可用 —— BM25 那一半零依赖，永远在。
        """
        if not self.settings.get("kb_dense") or not self.embed.running:
            return None
        return self.embed.embed

    def kb_state(self) -> dict:
        st = self.kb.state(dense_ready=embed_mod.embed_ready(self.settings))
        st["busy"] = self.kb_busy
        st["error"] = self.kb_error
        st["embed_running"] = self.embed.running
        st["last_hits"] = self.last_hits[-6:]
        return st

    def kb_begin(self, label: str) -> bool:
        """抢一个后台任务位。同一时刻只跑一个，避免两个索引进程互相覆写。"""
        with self._kb_lock:
            if self.kb_busy:
                return False
            self.kb_busy = label
            self.kb_error = ""
            return True

    def kb_end(self) -> None:
        with self._kb_lock:
            self.kb_busy = ""

    def kb_set_enabled(self, on: bool) -> None:
        self.settings.set("kb_enabled", bool(on))
        self.settings.save()

    def kb_set_dense(self, on: bool) -> None:
        self.settings.set("kb_dense", bool(on))
        self.settings.save()
        if on:
            threading.Thread(target=self._kb_build_vectors, daemon=True).start()
        else:
            self.embed.stop()

    def _kb_build_vectors(self) -> None:
        """下载向量模型 -> 起服务 -> 自检 -> 重算全部向量。

        自检不是可选项。pooling 设错的时候接口一切正常，只是相似度变成噪声，
        界面上完全看不出来，只会表现为"检索结果很随机"。宁可在这里明确失败，
        也不要给用户一个看起来在工作的坏检索 —— 那比没有检索更误导人。
        """
        if not self.kb_begin("准备向量模型"):
            return
        try:
            model = embed_mod.ensure_embed_model(self.settings, fetch.PROGRESS)
            with self._kb_lock:
                self.kb_busy = "启动向量服务"
            variant = fetch.pick_engine_variant(self.settings, self.engine.gpu())
            self.embed.start(model, variant)
            ok, msg = self.embed.selftest()
            if not ok:
                raise RuntimeError(f"自检没通过（{msg}）")
            with self._kb_lock:
                self.kb_busy = "建立向量索引"
            n = self.kb.reindex_vectors(self.embed.embed, fetch.PROGRESS)
            fetch.PROGRESS.set(stage="ready", label="就绪", done=0, total=0,
                               detail=f"向量索引已建立（{n} 块）")
        except Exception as e:                                  # noqa: BLE001
            self.kb_error = str(e)
            self.settings.set("kb_dense", False)
            self.settings.save()
            self.embed.stop()
        finally:
            self.kb_end()

    def kb_add(self, path: str, folder: bool) -> dict:
        p = Path(path)
        if not p.exists():
            raise RuntimeError("路径不存在")
        if folder:
            if not self.kb_begin(f"导入 {p.name}"):
                raise RuntimeError(f"正在忙：{self.kb_busy}")
            threading.Thread(target=self._kb_import_folder, args=(p,),
                             daemon=True).start()
            return {"ok": True, "async": True}
        with self._kb_lock:
            doc = self.kb.add_file(p, embed=self.kb_embed_fn(),
                                   progress=fetch.PROGRESS)
        return {"ok": True, "doc": doc.as_dict()}

    def _kb_import_folder(self, folder: Path) -> None:
        try:
            res = self.kb.add_dir(folder, embed=self.kb_embed_fn(),
                                  progress=fetch.PROGRESS)
            self.kb_error = ("" if not res["failed"]
                             else "部分文件没读进来：" + "；".join(res["failed"][:3]))
        except Exception as e:                                  # noqa: BLE001
            self.kb_error = str(e)
        finally:
            self.kb_end()
            fetch.PROGRESS.set(stage="ready", label="就绪", done=0, total=0, detail="")

    def kb_remove(self, doc_id: str) -> None:
        self.kb.remove(doc_id)

    def kb_clear(self) -> None:
        self.kb.clear()

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
            "gpu_text": gpu_summary(gpu),
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
            "loras": self.lora_state(),
            "knowledge": self.kb_state(),
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
        if path == "/app/loras":
            return self._json(self.app.lora_state())
        if path == "/app/kb":
            return self._json(self.app.kb_state())
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
        if path == "/app/pick-file":
            return self._pick_file()
        if path == "/app/loras/toggle":
            return self._lora_toggle()
        if path == "/app/loras/preset":
            return self._lora_preset()
        if path == "/app/loras/install":
            return self._lora_install()
        if path == "/app/loras/add":
            return self._lora_add()
        if path == "/app/loras/remove":
            return self._lora_remove()
        if path == "/app/kb/add":
            return self._kb_add()
        if path == "/app/kb/remove":
            return self._kb_remove()
        if path == "/app/kb/clear":
            return self._kb_clear()
        if path == "/app/kb/enabled":
            return self._kb_enabled()
        if path == "/app/kb/dense":
            return self._kb_dense()
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
        if variant in ("auto", "cpu", "cuda") and variant != s.get("engine_variant"):
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
        kind = str(self._body().get("kind") or "data")
        title = ("选择要加入知识库的文件夹" if kind == "docs"
                 else "选择存放模型和记录的文件夹")
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(title=title)
            root.destroy()
            return self._json({"ok": bool(path), "path": path or ""})
        except Exception as e:                                  # noqa: BLE001
            return self._json({"ok": False, "error": f"无法打开文件夹选择窗口：{e}"})

    def _pick_file(self) -> None:
        """挑一个文件。同样不让用户手打路径。"""
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        kind = str(self._body().get("kind") or "lora")
        if kind == "docs":
            title = "选择要加入知识库的文件"
            types = [("文档与代码",
                      "*.txt *.md *.pdf *.json *.csv *.py *.js *.ts *.java *.c *.h "
                      "*.cpp *.go *.rs *.html *.yaml *.yml *.log"),
                     ("所有文件", "*.*")]
        else:
            title = "选择一个风格包（.gguf）"
            types = [("风格包", "*.gguf"), ("所有文件", "*.*")]
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(title=title, filetypes=types)
            root.destroy()
            return self._json({"ok": bool(path), "path": path or ""})
        except Exception as e:                                  # noqa: BLE001
            return self._json({"ok": False, "error": f"无法打开文件选择窗口：{e}"})

    def _reload_later(self) -> None:
        """改完 LoRA 链要重启引擎才生效；放到后台，别让请求卡住。"""
        self.app.boot_done.clear()
        self.app.boot_async()

    def _lora_preset(self) -> None:
        """切预设。产品上用户只看到「像一个什么样的人说话」，看不到 rank 和 scale。"""
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        pid = str(self._body().get("id") or "")
        if lora_mod.preset(self.app.settings, pid) is None:
            return self._json({"ok": False, "error": f"没有这个预设: {pid}"}, 400)
        self.app.settings.set("preset", pid)
        self.app.settings.save()
        self._reload_later()
        return self._json({"ok": True, "preset": pid,
                           "state": self.app.lora_state()})

    def _lora_toggle(self) -> None:
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        data = self._body()
        lid = str(data.get("id") or "")
        want = bool(data.get("enabled"))
        ids = lora_mod.enabled_ids(self.app.settings)
        if want and lid not in ids:
            ids.append(lid)
        elif not want and lid in ids:
            ids.remove(lid)
        lora_mod.set_enabled(self.app.settings, ids)
        self._reload_later()
        return self._json({"ok": True, "enabled": lora_mod.enabled_ids(self.app.settings)})

    def _lora_install(self) -> None:
        """从精选目录一键安装。150 MB 量级，放后台下，进度走 /app/state。"""
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        lid = str(self._body().get("id") or "")
        if not lid:
            return self._json({"ok": False, "error": "缺少 id"}, 400)
        settings = self.app.settings

        def work() -> None:
            try:
                lora_mod.install(settings, lid, fetch.PROGRESS)
                ids = lora_mod.enabled_ids(settings)
                if lid not in ids:
                    ids.append(lid)
                lora_mod.set_enabled(settings, ids)
                self._reload_later()
            except Exception as e:                              # noqa: BLE001
                fetch.PROGRESS.set(stage="error", label="安装失败", error=str(e))

        threading.Thread(target=work, daemon=True).start()
        return self._json({"ok": True, "installing": lid})

    def _lora_add(self) -> None:
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        path = str(self._body().get("path") or "")
        if not path:
            return self._json({"ok": False, "error": "缺少文件路径"}, 400)
        try:
            lora = lora_mod.add_file(self.app.settings, path)
        except Exception as e:                                  # noqa: BLE001
            return self._json({"ok": False, "error": str(e)}, 400)
        ids = lora_mod.enabled_ids(self.app.settings)
        if lora.id not in ids:
            ids.append(lora.id)
        lora_mod.set_enabled(self.app.settings, ids)
        self._reload_later()
        return self._json({"ok": True, "id": lora.id, "state": self.app.lora_state()})

    def _lora_remove(self) -> None:
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        lid = str(self._body().get("id") or "")
        try:
            lora_mod.remove(self.app.settings, lid)
        except Exception as e:                                  # noqa: BLE001
            return self._json({"ok": False, "error": str(e)}, 400)
        ids = [i for i in lora_mod.enabled_ids(self.app.settings) if i != lid]
        lora_mod.set_enabled(self.app.settings, ids)
        self._reload_later()
        return self._json({"ok": True, "state": self.app.lora_state()})

    def _restart(self) -> None:
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        if self.app.tunnel.running:
            self.app.tunnel.stop()
        self.app.boot_done.clear()
        self.app.boot_async()
        return self._json({"ok": True})

    # --------------------------------------------------------------- knowledge
    def _kb_add(self) -> None:
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        data = self._body()
        path = str(data.get("path") or "")
        folder = bool(data.get("folder"))
        try:
            res = self.app.kb_add(path, folder)
        except Exception as e:                                  # noqa: BLE001
            return self._json({"ok": False, "error": str(e)}, 400)
        res["state"] = self.app.kb_state()
        return self._json(res)

    def _kb_remove(self) -> None:
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        try:
            self.app.kb_remove(str(self._body().get("id") or ""))
        except Exception as e:                                  # noqa: BLE001
            return self._json({"ok": False, "error": str(e)}, 400)
        return self._json({"ok": True, "state": self.app.kb_state()})

    def _kb_clear(self) -> None:
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        self.app.kb_clear()
        return self._json({"ok": True, "state": self.app.kb_state()})

    def _kb_enabled(self) -> None:
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        self.app.kb_set_enabled(bool(self._body().get("enabled")))
        return self._json({"ok": True, "state": self.app.kb_state()})

    def _kb_dense(self) -> None:
        if not self.local:
            return self._json({"error": "只能在本地操作"}, 403)
        self.app.kb_set_dense(bool(self._body().get("enabled")))
        return self._json({"ok": True, "state": self.app.kb_state()})

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

    @staticmethod
    def _last_user_text(messages: list) -> str:
        """取最后一条用户消息的文字部分。多模态消息里只挑 text 字段。"""
        for m in reversed(messages):
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content
                                if isinstance(p, dict) and p.get("type") in (None, "text"))
        return ""

    def _augment(self, payload: bytes) -> bytes:
        """给聊天请求插入检索到的资料。

        只在 /v1/chat/completions 上做：/v1/completions 是裸补全，没有消息
        结构可以挂资料；/v1/embeddings 更是不能碰。任何一步不对就原样放行，
        宁可少一次检索，也不能把用户的请求弄坏。
        """
        if not payload:
            return payload
        if urllib.parse.urlparse(self.path).path != "/v1/chat/completions":
            return payload
        app = self.app
        if not app.settings.get("kb_enabled", True):
            return payload
        try:
            body = json.loads(payload.decode("utf-8"))
        except Exception:                                       # noqa: BLE001
            return payload
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return payload
        query = self._last_user_text(messages)
        if not query.strip():
            return payload
        try:
            context, hits = app.kb.build_context(query, embed=app.kb_embed_fn())
        except Exception:                                       # noqa: BLE001
            return payload
        if not context:
            return payload
        body["messages"] = kb_mod.build_messages(messages, context)
        app.last_hits = [{"doc": h.doc_name, "score": h.score,
                          "preview": h.text[:80]} for h in hits]
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

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
        if payload is not None:
            payload = self._augment(payload)
            # 正文长度变了就必须重写 Content-Length，否则上游会按旧长度读，
            # 表现为请求挂住或截断 —— 这类错误在日志里几乎看不出来。
            headers = {k: v for k, v in headers.items()
                       if k.lower() != "content-length"}
            headers["Content-Length"] = str(len(payload))
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
