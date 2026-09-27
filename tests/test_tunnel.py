#!/usr/bin/env python3
"""内网穿透地址的识别。

2026-09 实测到的洞：cloudflared 建隧道**失败**时会印

    failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": ...

里面那个是它的管理接口地址（就写在 cloudflared.exe 里），同样以
.trycloudflare.com 结尾。原来的正则 [a-z0-9-]+ 把它当成隧道地址收下，
_pump 一设上 self.url，start() 就立刻返回"成功" —— 隧道根本没起来，
界面却给出一个打不开的网址。这个洞是真实撞到的：发布版 exe 的
「开启远程访问」返回了 https://api.trycloudflare.com。

下面这些行都是从这台机器上真跑出来的原文。

    python tests/test_tunnel.py
"""

from __future__ import annotations

import re
import sys
import tempfile
import threading
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonsai import tunnel as tun   # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not ok:
        FAILED.append(name)


# 这台机器上真出现过的 quick tunnel 域名（4 个词，3 个连字符）
REAL_HOSTS = [
    "https://dry-prompt-historical-betting.trycloudflare.com",
    "https://loop-defend-docs-oldest.trycloudflare.com",
    "https://kitchen-hopkins-attribute-rough.trycloudflare.com",
    "https://residents-jersey-trains-thank.trycloudflare.com",
]

# 真跑出来的两行原文
LINE_REQUESTING = "2026-09-27T09:56:27Z INF Requesting new quick Tunnel on trycloudflare.com...\n"
LINE_CREATED = "2026-09-27T09:56:32Z INF |  https://kitchen-hopkins-attribute-rough.trycloudflare.com                                 |\n"
LINE_BANNER = ("2026-09-27T09:56:32Z INF Your quick Tunnel has been created! "
               "Visit it at (it may take some time to be reachable):\n")
# 失败原文的形状：报错模板 + 管理接口地址（两者都在 cloudflared.exe 里）
LINE_FAILED = ('2026-09-27T09:57:10Z ERR failed to request quick Tunnel: '
               'Post "https://api.trycloudflare.com/tunnel": '
               'dial tcp 104.16.0.1:443: connectex: A connection attempt failed\n')


# ------------------------------------------------------------------ 1. 正则
print("\n1. URL_RE 只认 quick tunnel 的形状")
for h in REAL_HOSTS:
    check(f"接受 {h.split('//')[1][:28]}…", bool(tun.URL_RE.search(h)))

check("拒绝管理接口地址 https://api.trycloudflare.com",
      not tun.URL_RE.search("https://api.trycloudflare.com"))
check("拒绝 api.trycloudflare.com/tunnel（带路径）",
      not tun.URL_RE.search('Post "https://api.trycloudflare.com/tunnel"'))
check("拒绝 www.trycloudflare.com", not tun.URL_RE.search("https://www.trycloudflare.com"))
check("拒绝裸域名 trycloudflare.com",
      not tun.URL_RE.search("Requesting new quick Tunnel on trycloudflare.com..."))
check("拒绝别的域名", not tun.URL_RE.search("https://example.com"))


# --------------------------------------------------------------- 2. _pump
class FakeSettings:
    def __init__(self, d):
        self.data_dir = d
        self.logs_dir = d


def pump_over(lines: list[str]):
    """让真实的 _pump 跑一遍这些行，返回那个 Tunnel。"""
    t = tun.Tunnel(FakeSettings(Path(tempfile.gettempdir())))
    t.proc = types.SimpleNamespace(stdout=iter(lines))
    t._log_fh = None
    t._pump()
    return t


print("\n2. 正常成功：能认出隧道地址")


t = pump_over([LINE_REQUESTING, LINE_BANNER, LINE_CREATED])
check("url 被认出来", t.url == REAL_HOSTS[2], t.url)
check("没有误报错误", t.error == "", t.error)

print("\n3. 建隧道失败：**不能**报出成功网址")
t = pump_over([LINE_REQUESTING, LINE_FAILED])
check("url 保持为空", t.url == "", f"实际 {t.url!r}")
check("错误被记下来", "failed to request quick Tunnel" in t.error, t.error[:70])
check("（自检）这一行确实含网址，不是空跑",
      "https://api.trycloudflare.com" in LINE_FAILED)

print("\n4. 失败之后才成功：仍然算成功")
t = pump_over([LINE_REQUESTING, LINE_FAILED, LINE_BANNER, LINE_CREATED])
check("url 是真正的隧道地址", t.url == REAL_HOSTS[2], t.url)


# ------------------------------------------------------- 5. start() 端到端
print("\n5. start() 在隧道失败时必须抛错，而不是返回一个网址")


class FakeStdout:
    def __init__(self, lines, done):
        self.lines, self.done = lines, done

    def __iter__(self):
        yield from self.lines
        self.done.set()


class FakePopen:
    def __init__(self, lines, done):
        self.stdout = FakeStdout(lines, done)
        self._done = done

    def poll(self):
        return None if not self._done.is_set() else 1

    def terminate(self):
        self._done.set()

    def kill(self):
        self._done.set()

    def wait(self, timeout=None):
        return 0


def drive_start(lines):
    done = threading.Event()
    saved_pop, saved_find = tun.subprocess.Popen, tun.find_cloudflared
    tun.subprocess.Popen = lambda *a, **k: FakePopen(lines, done)
    tun.find_cloudflared = lambda s: Path("cloudflared.exe")
    try:
        t = tun.Tunnel(FakeSettings(Path(tempfile.mkdtemp())))
        try:
            return t.start(1234), None
        except RuntimeError as e:
            return None, str(e)
    finally:
        tun.subprocess.Popen, tun.find_cloudflared = saved_pop, saved_find


url, err = drive_start([LINE_REQUESTING, LINE_FAILED])
check("失败时没有返回网址", url is None, f"实际返回 {url!r}")
check("抛出的原因里带着 cloudflared 的原话",
      err is not None and "failed to request quick Tunnel" in err, (err or "")[:70])

url, err = drive_start([LINE_REQUESTING, LINE_BANNER, LINE_CREATED])
check("成功时返回隧道地址", url == REAL_HOSTS[2], repr(url))

# ------------------------------------------------- 6. 重复点击不能返回空网址
print("\n6. 隧道还在起的时候再点一次，不能返回空网址")


class AliveProc:
    """假装 cloudflared 还活着，但地址还没解析出来。

    poll() 先回两次 None（代表"还在跑"），之后算退出 —— 这样 start() 的等待
    循环会很快结束，测试不用真的耗 60 秒。
    """

    def __init__(self):
        self.stdout = iter([])
        self.calls = 0

    def poll(self):
        self.calls += 1
        return None if self.calls <= 2 else 1

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


t = tun.Tunnel(FakeSettings(Path(tempfile.mkdtemp())))
t.proc = AliveProc()
t._pump = lambda: None          # 不启动 pump，url 永远是空的
try:
    got = t.start(1234)
    raised = None
except RuntimeError as e:
    got, raised = None, str(e)

check("没有返回空串", got is None, f"实际返回 {got!r}")
check("抛错说明了还在启动", raised is not None, (raised or "")[:60])

print()
if FAILED:
    print(f"  {len(FAILED)} 项失败：")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  全部通过")
