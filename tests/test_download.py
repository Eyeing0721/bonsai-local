#!/usr/bin/env python3
"""下载器独立测试 —— 不碰真的 5.5 GB 模型。

真实用户会在下载到 4 GB 的时候断网一次，所以断点续传、来源回退、
校验失败重来这三条路径必须是被真的跑过，而不是"看起来写了"。

    python tests/test_download.py
"""

from __future__ import annotations

import hashlib
import http.server
import os
import socket
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonsai.fetch import Progress, download, fetch_with_fallback  # noqa: E402

PAYLOAD = bytes(range(256)) * (4096 * 5)          # 5 MB，可校验的确定内容
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
RANGE_HITS: list[str] = []
FULL_HITS: list[int] = []


class RangeHandler(http.server.BaseHTTPRequestHandler):
    """一个支持 Range 的最小服务器（标准库那个不支持，所以自己写）。"""

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                        # noqa: A003
        pass

    def _headers(self, code: int, start: int, end: int, total: int):
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(end - start))
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end - 1}/{total}")
        self.end_headers()

    def do_HEAD(self):                                # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):                                 # noqa: N802
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            RANGE_HITS.append(rng)
            start = int(rng.split("=")[1].split("-")[0])
            end = len(PAYLOAD)
            self._headers(206, start, end, len(PAYLOAD))
            self.wfile.write(PAYLOAD[start:end])
        else:
            FULL_HITS.append(1)
            self._headers(200, 0, len(PAYLOAD), len(PAYLOAD))
            self.wfile.write(PAYLOAD)


class BrokenHandler(RangeHandler):
    """永远 500，用来验证来源回退。"""

    def do_HEAD(self):                                # noqa: N802
        self.send_response(500)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):                                 # noqa: N802
        self.send_response(500)
        self.send_header("Content-Length", "0")
        self.end_headers()


def serve(handler) -> tuple[http.server.ThreadingHTTPServer, str]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}/blob"


def check(name: str, ok: bool, extra: str = "") -> bool:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    return ok


def main() -> int:
    good = True
    srv, url = serve(RangeHandler)
    tmp = Path(tempfile.mkdtemp(prefix="bonsai-dl-"))

    try:
        # 1) 完整下载
        p = Progress()
        dest = tmp / "full.bin"
        download(url, dest, p, "测试完整下载")
        good &= check("完整下载内容一致", dest.read_bytes() == PAYLOAD,
                      f"{dest.stat().st_size} 字节")

        # 2) 断点续传：先伪造一个半截的 .part，再下
        part = dest.with_suffix(dest.suffix + ".part")
        RANGE_HITS.clear()
        dest.unlink()
        part.write_bytes(PAYLOAD[: 2 * 1024 * 1024])
        download(url, dest, p, "测试续传")
        good &= check("续传后内容一致", dest.read_bytes() == PAYLOAD)
        good &= check("确实用了 Range 而不是重下", bool(RANGE_HITS), str(RANGE_HITS[:1]))

        # 3) .part 已经比目标还大时不应该死循环
        dest.unlink()
        part.write_bytes(PAYLOAD + b"x" * 16)
        try:
            download(url, dest, p, "测试超长 .part")
            good &= check("超长 .part 被直接采用", dest.stat().st_size >= len(PAYLOAD))
        except Exception as e:                        # noqa: BLE001
            good &= check("超长 .part 不应抛异常", False, repr(e))

        # 4) 来源回退：坏的先试，好的兜底
        bsrv, bad = serve(BrokenHandler)
        p2 = Progress()
        dest2 = tmp / "fallback.bin"
        dest2.unlink(missing_ok=True)
        dest2.with_suffix(dest2.suffix + ".part").unlink(missing_ok=True)
        try:
            fetch_with_fallback([bad, url], dest2, p2, "测试回退")
            good &= check("坏来源失败后自动换到好来源", dest2.read_bytes() == PAYLOAD)
        except Exception as e:                        # noqa: BLE001
            good &= check("坏来源失败后自动换到好来源", False, repr(e))
        bsrv.shutdown()

        # 5) 校验和不匹配必须报错，而不是留下坏文件
        p3 = Progress()
        dest3 = tmp / "bad.bin"
        dest3.unlink(missing_ok=True)
        dest3.with_suffix(dest3.suffix + ".part").unlink(missing_ok=True)
        try:
            fetch_with_fallback([url], dest3, p3, "测试校验", sha256="0" * 64)
            good &= check("校验和不匹配时应当报错", False, "居然成功了")
        except IOError:
            good &= check("校验和不匹配时应当报错", True)
            good &= check("坏文件没有留下", not dest3.exists())

        # 6) 正确的校验和应当通过
        p4 = Progress()
        dest4 = tmp / "good.bin"
        dest4.unlink(missing_ok=True)
        dest4.with_suffix(dest4.suffix + ".part").unlink(missing_ok=True)
        fetch_with_fallback([url], dest4, p4, "测试校验通过", sha256=DIGEST)
        good &= check("正确校验和应通过", dest4.read_bytes() == PAYLOAD)
    finally:
        srv.shutdown()

    print("\n结果:", "全部通过" if good else "有失败项")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
