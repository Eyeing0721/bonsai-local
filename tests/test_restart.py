#!/usr/bin/env python3
"""「重启引擎」这条路径的回归测试。

2026-09 实测出来的三个洞，同一个根因 —— engine.start() 在引擎已经跑着的时候
是 `return`（静默成功）：

  1. boot() 调的是 start()，不是 restart()。于是「重启模型」按钮、切风格预设、
     开关 LoRA、改上下文档位，全都是空操作：前后 PID 一样、端口一样、进程没换，
     界面却报 ok。这一条从 1.0.0 就在，engine.restart() 是从没被调用过的死代码。
  2. start() 见到 running 就 return，把"配置其实没生效"瞒下来了。
  3. _set_settings 改完档位只调 boot_async()，没先 begin_boot()，所以引擎
     （本来也没重启）和界面一起装作无事发生。

为什么单测能挡：这三条都是"该调 A 结果调了 B"的错误，不需要真的起模型。
端到端那一次另外做（改档位后对比 llama-server 的 PID）。

    python tests/test_restart.py
"""

from __future__ import annotations

import re
import sys
import threading
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonsai import server as srv              # noqa: E402
from bonsai.engine import Engine, EngineError  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not ok:
        FAILED.append(name)


# --------------------------------------------------------------------- fakes
class FakeProc:
    """冒充 subprocess.Popen：poll() 返回 None 就代表"还活着"。"""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.alive = True

    def poll(self):
        return None if self.alive else self.returncode


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def gpu(self):
        return {"name": "fake", "vram_mb": 16376}

    def restart(self, model, loras):
        self.calls.append("restart")

    def start(self, model, loras):
        self.calls.append("start")


class FakeTunnel:
    running = False

    def stop(self):
        pass


class FakeKB:
    def load(self):
        pass


class FakeSettings:
    """只实现 boot() 真正碰到的那几个方法。"""

    def __init__(self) -> None:
        self.data = {"kb_dense": False}

    def ensure_dirs(self):
        pass

    def get(self, k, default=None):
        return self.data.get(k, default)

    def set(self, k, v):
        self.data[k] = v

    def update(self, **kw):
        self.data.update(kw)

    def save(self):
        pass


def patch_fetch() -> None:
    """把 boot() 会碰的文件系统和网络换掉。

    注意：必须一直 patch 着，不能在 make_app() 里还原 —— boot() 是在 make_app()
    **返回之后**才跑的，还原了它就会去真的下 5 GB 模型。
    """
    srv.fetch.pick_engine_variant = lambda s, g: "cuda"
    srv.fetch.ensure_engine = lambda s, v, p: None
    srv.fetch.ensure_model = lambda s, p: Path("fake.gguf")
    srv.kb_mod.Knowledge = lambda s: FakeKB()
    srv.embed_mod.EmbedServer = lambda s: types.SimpleNamespace()


def make_app(engine):
    app = srv.App(FakeSettings(), engine, FakeTunnel())
    app.lora_paths = lambda: []          # 不碰真实的 LoRA 目录
    return app


# ------------------------------------------------------------------- 1. 守卫
patch_fetch()
print("\n1. engine.start() 不能再静默返回")
eng = Engine(FakeSettings())
proc = FakeProc()
eng.proc = proc
try:
    eng.start(Path("x.gguf"), [])
    raised = False
except EngineError:
    raised = True
check("引擎已在运行时 start() 抛 EngineError", raised)
check("start() 没有偷偷把 proc 换掉", eng.proc is proc)

eng.proc = None
check("引擎没在跑时 running 为 False", eng.running is False)

# ------------------------------------------------- 2. boot() 必须走 restart()
print("\n2. boot() 要求引擎用新配置重新加载")
fe = FakeEngine()
app = make_app(fe)
app.boot()
check("boot() 调的是 engine.restart", fe.calls == ["restart"], f"实际 {fe.calls}")
check("boot() 没有误调 engine.start", "start" not in fe.calls)
check("boot() 之后 engine_ready 为 True", app.engine_ready is True)
check("boot() 之后 boot_done 已置位", app.boot_done.is_set())

# ------------------------------------------- 3. begin_boot() 要把状态打回去
print("\n3. begin_boot() 打回未就绪（前端据此回准备页）")
fe2 = FakeEngine()
app2 = make_app(fe2)
app2.boot()
app2.begin_boot()
check("begin_boot() 之后 engine_ready 为 False", app2.engine_ready is False)
check("begin_boot() 之后 boot_done 被清掉", not app2.boot_done.is_set())
app2.boot()
check("再 boot() 一次后恢复 True", app2.engine_ready is True)

# ----------------------------------------- 4. 失败不能留下"假装就绪"的状态
print("\n4. 启动失败必须如实置 False")


class BoomEngine(FakeEngine):
    def restart(self, model, loras):
        self.calls.append("restart")
        raise EngineError("假装模型文件没了")


app3 = make_app(BoomEngine())
app3.boot()
check("失败后 engine_ready 为 False", app3.engine_ready is False)
check("失败原因记进了 last_error", "假装模型文件没了" in app3.last_error)
app3.begin_boot()
check("重试时清掉旧错误（否则界面一直停在失败页）", app3.last_error == "")

# ------------------------------------- 5. _reload_later 先打状态再起线程
print("\n5. _reload_later() 的顺序：先 begin_boot，再后台重启")


class GatedEngine(FakeEngine):
    """重启会停在闸门上，好让我们确定地观察"正在重启"那一瞬间的状态。

    不加闸门的话假 boot 是微秒级的，后台线程会在断言之前就跑完，
    测出来的东西全是竞态。
    """

    def __init__(self):
        super().__init__()
        self.gate = threading.Event()
        self.ready_at_restart: list[bool] = []
        self.app = None

    def restart(self, model, loras):
        self.calls.append("restart")
        self.ready_at_restart.append(self.app.engine_ready)
        self.gate.wait(timeout=10)


ge = GatedEngine()
app4 = make_app(ge)
ge.app = app4

ge.gate.set()
app4.boot()                          # 第一次启动，放行
check("先让引擎就绪", app4.engine_ready is True)

ge.gate.clear()
ge.ready_at_restart.clear()

# _reload_later 是 Handler 的方法，不是 App 的 —— 借一个空壳实例来调
handler = srv.Handler.__new__(srv.Handler)
handler.app = app4
handler._reload_later()
check("_reload_later() 同步地把 engine_ready 打成 False", app4.engine_ready is False)

ge.gate.set()
app4.boot_done.wait(timeout=10)
check("后台 restart 时 engine_ready 已经是 False（前端不会闪回聊天页）",
      ge.ready_at_restart == [False], f"实际 {ge.ready_at_restart}")
check("后台 boot 完成后又回到 True", app4.engine_ready is True)

# ------------------------------------------------ 6. 结构：设置改动走同一路
print("\n6. _set_settings 改档位必须走 _reload_later")
src = Path(srv.__file__).read_text(encoding="utf-8")
body = re.search(r"def _set_settings\(self\).*?(?=\n    def )", src, re.S)
check("找得到 _set_settings", body is not None)
if body:
    text = body.group(0)
    check("用了 _reload_later()", "_reload_later()" in text)
    check("没有裸调 boot_async()（会漏掉 begin_boot）",
          "self.app.boot_async()" not in text)

print()
if FAILED:
    print(f"  {len(FAILED)} 项失败：")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  全部通过")
