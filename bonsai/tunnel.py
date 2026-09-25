"""Cloudflare quick tunnel -- free public HTTPS with no account and no config.

We use `cloudflared tunnel --url ...`, which prints a throwaway
`https://<words>.trycloudflare.com` address on stderr. That is the whole feature:
one switch, one link, nothing for the user to sign up for.

The tunnel exposes the *app* server, not llama-server, so every remote request
still passes the token check and the settings API stays unreachable from outside.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .config import app_root

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

CLOUDFLARED_URLS = [
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
]


def find_cloudflared(settings) -> Path | None:
    """Bundled copy, a downloaded copy, then whatever is on PATH."""
    candidates = [
        app_root() / "cloudflared.exe",
        settings.data_dir / "cloudflared.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    which = shutil.which("cloudflared")
    if which:
        return Path(which)
    # a common installer location on Windows
    for p in (r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
              r"C:\Program Files\cloudflared\cloudflared.exe"):
        if Path(p).exists():
            return Path(p)
    return None


def download_cloudflared(settings, progress) -> Path | None:
    import urllib.request
    dest = settings.data_dir / "cloudflared.exe"
    dest.parent.mkdir(parents=True, exist_ok=True)
    progress.set(stage="downloading", label="下载内网穿透组件", done=0, total=0, detail="")
    for url in CLOUDFLARED_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BonsaiLocal/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r, dest.open("wb") as fh:
                total = int(r.headers.get("Content-Length") or 0)
                if total:
                    progress.set(total=total)
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
                    progress.add(len(chunk))
            return dest
        except Exception:                                       # noqa: BLE001
            continue
    return None


class Tunnel:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.proc: subprocess.Popen | None = None
        self.url: str = ""
        self.error: str = ""
        self._lines: list[str] = []
        self._thread: threading.Thread | None = None
        self._log_fh = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, port: int, progress=None) -> str:
        """Returns the public URL, or raises with a readable reason."""
        if self.running:
            return self.url
        exe = find_cloudflared(self.settings)
        if exe is None and progress is not None:
            exe = download_cloudflared(self.settings, progress)
        if exe is None:
            raise RuntimeError("找不到 cloudflared，且自动下载失败")

        self.settings.logs_dir.mkdir(parents=True, exist_ok=True)
        self._log_fh = (self.settings.logs_dir / "tunnel.log").open(
            "w", encoding="utf-8", errors="replace")
        cmd = [str(exe), "tunnel", "--url", f"http://127.0.0.1:{port}",
               "--no-autoupdate"]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=CREATE_NO_WINDOW,
        )
        self.url, self.error, self._lines = "", "", []
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

        deadline = time.time() + 60
        while time.time() < deadline:
            if self.url:
                return self.url
            if not self.running:
                break
            time.sleep(0.3)
        self.stop()
        raise RuntimeError(self.error or "隧道启动超时")

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.append(line.rstrip())
            if self._log_fh is not None:
                try:
                    self._log_fh.write(line)
                    self._log_fh.flush()
                except Exception:                               # noqa: BLE001
                    pass
            if not self.url:
                m = URL_RE.search(line)
                if m:
                    self.url = m.group(0)
            low = line.lower()
            if "error" in low or "failed" in low:
                self.error = line.strip()

    def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=10)
            except Exception:                                   # noqa: BLE001
                try:
                    p.kill()
                except Exception:                               # noqa: BLE001
                    pass
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:                                   # noqa: BLE001
                pass
            self._log_fh = None
        self.url = ""

    def status(self) -> dict:
        return {"running": self.running, "url": self.url, "error": self.error}
