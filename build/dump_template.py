#!/usr/bin/env python3
"""从模型的 GGUF 里读出 chat template，看它接受哪些 reasoning_effort 取值。

起因：测试里传 reasoning_effort="high" 得到 HTTP 500，模板抛
"Unexpected reasoning effort"。所以取值是被模板硬校验的，得看它到底收什么。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 上游 gguf 包还不认识 PTQ1_0（类型 143），补一个伪成员
from gguf import GGMLQuantizationType as Q

_prev = Q.__dict__.get("_missing_")


def _missing_(cls, value):
    if value == 143:
        o = int.__new__(cls, 143)
        o._name_ = "PTQ1_0"
        o._value_ = 143
        cls._value2member_map_[143] = o
        return o
    return _prev.__func__(cls, value) if _prev else None


Q._missing_ = classmethod(_missing_)

# 光让枚举认得还不够：_build_tensors 还要拿块大小去算张量偏移。
# PTQ1_0 的布局是 {uint8 qs[24]; uint8 qh[2]; fp16 d} = 每 128 个值 28 字节。
from gguf import GGML_QUANT_SIZES                             # noqa: E402

PTQ1_0 = Q(143)          # 通过 _missing_ 造出伪成员（属性名 Q.PTQ1_0 并不存在）
PTQ1_0._name_ = "PTQ1_0"
Q._member_map_["PTQ1_0"] = PTQ1_0
GGML_QUANT_SIZES[PTQ1_0] = (128, 28)

from gguf import GGUFReader                                  # noqa: E402

MODEL = Path(sys.argv[1] if len(sys.argv) > 1
             else r"E:\models\bonsai\Ternary-Bonsai-2-27B-PTQ1_0.gguf")


def main() -> int:
    reader = GGUFReader(str(MODEL))
    tpl = None
    for key, field in reader.fields.items():
        if "chat_template" in key:
            try:
                tpl = field.contents()
            except Exception:                                   # noqa: BLE001
                parts = getattr(field, "parts", None)
                if parts:
                    tpl = "".join(
                        p.decode("utf-8", "replace") if isinstance(p, bytes) else str(p)
                        for p in parts)
            print(f"找到 {key}（{len(tpl or '')} 字）\n")
            break
    if not tpl:
        print("没找到 chat_template")
        return 1

    lines = tpl.splitlines()
    print("=== 与 reasoning / thinking 有关的片段 ===")
    for i, line in enumerate(lines, 1):
        low = line.lower()
        if "reasoning" in low or "thinking" in low:
            lo = max(0, i - 2)
            hi = min(len(lines), i + 2)
            print(f"--- 第 {i} 行附近 ---")
            for j in range(lo, hi):
                print(f"  {j + 1:>4}: {lines[j][:170]}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
