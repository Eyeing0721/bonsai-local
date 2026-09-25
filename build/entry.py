"""PyInstaller 的入口。

不能让 PyInstaller 直接打包 `bonsai/__main__.py`：它会把那个文件当成顶层脚本执行，
于是 `from . import fetch` 这类相对导入会失败（ImportError: attempted relative import
with no known parent package）。这里用一个绝对导入的壳把它拉起来。

另外这里兜住"双击了但什么都没发生"这种情况：发布版没有控制台，异常会静默消失，
对一个陌生人来说这是最糟的失败方式。所以崩溃时弹一个框并留下日志文件。
"""

import multiprocessing
import os
import sys

from bonsai.__main__ import main


def _report_crash(exc: BaseException) -> None:
    import traceback

    tb = traceback.format_exc()
    where = "(日志写入失败)"
    try:
        from pathlib import Path
        base = Path(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")) / "BonsaiLocal"
        base.mkdir(parents=True, exist_ok=True)
        log = base / "crash.log"
        log.write_text(tb, encoding="utf-8")
        where = str(log)
    except Exception:                                           # noqa: BLE001
        pass

    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None,
            f"启动失败：\n\n{type(exc).__name__}: {exc}\n\n"
            f"详细信息已写入：\n{where}\n\n"
            f"可以把这个文件发给开发者，或到 Releases 页面反馈。",
            "Bonsai 本地助手",
            0x10,                       # MB_ICONERROR
        )
    except Exception:                                           # noqa: BLE001
        pass


if __name__ == "__main__":
    # 打包后如果不声明，子进程会重新执行整个程序
    multiprocessing.freeze_support()
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as e:                                  # noqa: BLE001
        _report_crash(e)
        raise
