#!/usr/bin/env python3
"""首次运行的回归测试。

真实发生过的故障：联网取 engines.json 打到了占位地址（404），回退字典里我写的是
`"zip"` 而不是 `"url"`，于是 `spec["url"]` 抛 KeyError: 'url' —— 用户双击后看到的
就是这个。这类"回退路径从没被跑过"的 bug 只能靠断言拦住，不能靠通读代码。

    python tests/test_first_run.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonsai.config import Settings                    # noqa: E402
from bonsai.fetch import Progress, ensure_engine, load_engine_manifest  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not ok:
        FAILED.append(name)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bonsai-firstrun-"))

    # 1) 清单里缺 url：必须给出人话，而不是 KeyError
    s = Settings()
    s.set("data_dir", str(tmp / "a"))
    try:
        ensure_engine(s, "cuda-ada", Progress(), manifest={"cuda-ada": {"bytes": 0}})
        check("清单缺 url 时应当报错", False, "居然没报错")
    except KeyError as e:
        check("清单缺 url 时不应抛 KeyError", False, f"KeyError: {e}")
    except RuntimeError as e:
        msg = str(e)
        check("清单缺 url 时给出可读错误", "地址还没配置好" in msg, msg.splitlines()[0])
    except Exception as e:                                    # noqa: BLE001
        check("清单缺 url 时给出可读错误", False, f"{type(e).__name__}: {e}")

    # 2) 占位地址也必须被识别出来
    s2 = Settings()
    s2.set("data_dir", str(tmp / "b"))
    try:
        ensure_engine(s2, "cuda-ada", Progress(),
                      manifest={"cuda-ada": {"url": "https://github.com/OWNER/REPO/x.zip"}})
        check("占位下载地址应当被拒绝", False, "居然接受了")
    except RuntimeError:
        check("占位下载地址应当被拒绝", True)
    except Exception as e:                                    # noqa: BLE001
        check("占位下载地址应当被拒绝", False, f"{type(e).__name__}: {e}")

    # 3) 请求的 GPU 变体不存在时，应自动退回 cpu 而不是崩
    s3 = Settings()
    s3.set("data_dir", str(tmp / "c"))
    try:
        ensure_engine(s3, "cuda-ada", Progress(),
                      manifest={"cpu": {"url": "https://example.invalid/engine-cpu.zip"}})
        check("缺 GPU 变体时不应直接崩", False, "居然下载成功了")
    except (RuntimeError, OSError) as e:
        # 能走到"下载失败"就说明已经正确退回到 cpu 了；
        # 不允许的是"没有可用构建"或 KeyError
        check("缺 GPU 变体时退回到 cpu", "没有任何可用构建" not in str(e), str(e)[:60])
    except KeyError as e:
        check("缺 GPU 变体时退回到 cpu", False, f"KeyError: {e}")
    except Exception as e:                                    # noqa: BLE001
        check("缺 GPU 变体时退回到 cpu", False, f"{type(e).__name__}: {e}")

    # 4) 清单完全为空时给出可读错误
    s4 = Settings()
    s4.set("data_dir", str(tmp / "d"))
    try:
        ensure_engine(s4, "cuda-ada", Progress(), manifest={"cpu": {}})
        check("空条目应当报错", False, "居然没报错")
    except KeyError as e:
        check("空条目不应抛 KeyError", False, f"KeyError: {e}")
    except Exception:                                         # noqa: BLE001
        check("空条目应当报错", True)

    # 5) 打包时应自带清单；源码运行时至少要能明确失败
    try:
        m = load_engine_manifest()
        check("能找到引擎清单", isinstance(m, dict) and bool(m), f"变体: {sorted(m)}")
    except RuntimeError as e:
        check("清单缺失时给出可读错误", "engines.json" in str(e), str(e).splitlines()[0])

    print("\n结果:", "全部通过" if not FAILED else f"失败 {len(FAILED)} 项: {FAILED}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
