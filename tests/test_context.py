#!/usr/bin/env python3
"""上下文档位与显存模型的回归测试。

这个模型不是拍脑袋写的：两个常数是从两次实收回推并校准到零偏差的 ——
  · f16  KV, ctx=131072 → 实测总计 15.50 GiB
  · q8_0 KV, ctx=262144 → 实测总计 15.96 GiB
如果谁改了 KV_BYTES_* 或 BUFFER_*，这两个断言会立刻拦住，因为那种改动会让
"界面说能开" 和 "引擎真能开" 分家 —— 用户看到的是后者崩掉。

    python tests/test_context.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonsai.config import (BUFFER_BASE_MIB, BUFFER_PER_TOKEN_MIB,  # noqa: E402
                           KV_BYTES_F16, KV_BYTES_Q8, MEMORY_TIERS,
                           TIER_ORDER, WEIGHTS_MIB, Settings, best_tier,
                           context_ceiling, kv_mib_per_token, resolve_tier)

FAILED: list[str] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not ok:
        FAILED.append(name)


def total_gib(ctx: int, q8: bool) -> float:
    """按模型的公式算总占用，用来和实测对账。"""
    kv = ctx * kv_mib_per_token(q8)
    return (WEIGHTS_MIB + kv + BUFFER_BASE_MIB + BUFFER_PER_TOKEN_MIB * ctx) / 1024


def main() -> int:
    # 1) KV 每 token 的字节数：64 层里只有 16 层是普通注意力，其余是 GDN（状态定长）
    check("f16 每 token = 64 KiB", KV_BYTES_F16 == 64 * 1024, f"{KV_BYTES_F16} B")
    check("q8_0 每 token = 32 KiB", KV_BYTES_Q8 == 32 * 1024, f"{KV_BYTES_Q8} B")
    check("q8_0 正好是 f16 的一半", KV_BYTES_Q8 * 2 == KV_BYTES_F16)

    # 2) 校准对账：公式必须复现两次实测（容差 0.01 GiB）
    for name, ctx, q8, measured in (("f16@131072", 131072, False, 15.50),
                                    ("q8_0@262144", 262144, True, 15.96)):
        got = total_gib(ctx, q8)
        check(f"{name} 回代实测 {measured} GiB", abs(got - measured) < 0.01,
              f"算出 {got:.3f}")

    # 3) 这台开发机的真实档位：16 GB 卡应当能开到「拉满」，且自动用 q8_0
    vram = 16380
    r = resolve_tier("max", vram)
    check("16 GB 卡可以开「拉满」", r["available"], f"ceiling={r['ceiling']:,}")
    check("「拉满」自动改用 q8_0 KV", r["kv_q8"])
    check("16 GB 卡推荐档位就是「拉满」", best_tier(vram) == "max", best_tier(vram))

    # 4) 小卡：必须如实报告装不下，而不是乐观放行
    small = 8192
    check("8 GB 卡开不了 131072", not resolve_tier("extreme", small)["available"])
    check("8 GB 卡能开 16384", resolve_tier("standard", small)["available"])
    check("8 GB 卡推荐档位不越界",
          MEMORY_TIERS[best_tier(small)]["ctx"] <= context_ceiling(small, True),
          f"{best_tier(small)} = {MEMORY_TIERS[best_tier(small)]['ctx']}")

    # 5) 无 GPU：建议值要压到 CPU 上限之内
    r = resolve_tier("max", None)
    check("无 GPU 时 ctx 被压到 CPU 上限内", r["ctx"] <= 32768, f"ctx={r['ctx']}")
    check("无 GPU 时标记为不可用", not r["available"])

    # 6) 档位单调递增，且「拉满」就是模型原生上限
    ctxs = [MEMORY_TIERS[k]["ctx"] for k in TIER_ORDER]
    check("档位从小到大单调", ctxs == sorted(ctxs), str(ctxs))
    check("最高档 = 262144（模型原生）", ctxs[-1] == 262144, str(ctxs[-1]))

    # 7) 截断：用户选了装不下的档位，应当得到"按本机上限截断"而不是抛异常
    tmp = Path(tempfile.mkdtemp(prefix="bonsai-ctx-"))
    s = Settings()
    s.set("data_dir", str(tmp))
    s.set("memory_tier", "max")
    res = s.resolve_context(8192)
    check("装不下时按上限截断而不是报错", res["ctx"] > 0 and res.get("clamped"),
          f"ctx={res['ctx']} clamped={res.get('clamped')}")
    check("截断值不超过本机上限", res["ctx"] <= context_ceiling(8192, True),
          f"{res['ctx']} <= {context_ceiling(8192, True)}")
    check("截断值对齐到 1024", res["ctx"] % 1024 == 0, str(res["ctx"]))

    # 8) 选得下的档位不应该被截断
    s.set("memory_tier", "standard")
    res = s.resolve_context(16380)
    check("装得下就不截断", not res.get("clamped") and res["ctx"] == 16384,
          f"ctx={res['ctx']} clamped={res.get('clamped')}")

    print()
    if FAILED:
        print(f"自检结果：{len(FAILED)} 项失败 -> {', '.join(FAILED)}")
        return 1
    print("自检结果：全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
