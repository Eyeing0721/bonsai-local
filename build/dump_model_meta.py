#!/usr/bin/env python3
"""读模型的架构元数据，用来算 KV 缓存和上下文上限。

这个模型有个不寻常的地方值得单独确认：64 层里只有 16 层是普通注意力，另外 48 层
是 GDN（门控线性注意力/状态空间）。后者每层的状态是**定长**的，不随上下文增长 ——
所以"上下文能吃多长"的显存账单和普通 64 层 transformer 完全不是一回事。
"""

from __future__ import annotations

import sys
from pathlib import Path

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
PTQ1_0 = Q(143)
PTQ1_0._name_ = "PTQ1_0"
Q._member_map_["PTQ1_0"] = PTQ1_0

from gguf import GGML_QUANT_SIZES, GGUFReader                # noqa: E402

GGML_QUANT_SIZES[PTQ1_0] = (128, 28)

MODEL = Path(sys.argv[1] if len(sys.argv) > 1
             else r"E:\models\bonsai\Ternary-Bonsai-2-27B-PTQ1_0.gguf")


def val(field):
    try:
        v = field.contents()
    except Exception:                                           # noqa: BLE001
        raw = getattr(field, "parts", None)
        if not raw:
            return None
        v = raw
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    return v


def main() -> int:
    r = GGUFReader(str(MODEL))
    meta: dict[str, object] = {}
    for key, field in r.fields.items():
        if key.startswith(("qwen35.", "general.", "llama.")):
            if "chat_template" in key or "tokenizer.ggml" in key:
                continue
            meta[key] = val(field)

    print(f"模型 {MODEL.name}  {MODEL.stat().st_size / 2**30:.2f} GiB\n")
    print("=== 架构元数据 ===")
    for k in sorted(meta):
        v = meta[k]
        if isinstance(v, list) and len(v) > 12:
            v = f"[{len(v)} 项] {v[:6]}…"
        print(f"  {k:<46} {v}")

    def g(*names, default=None):
        for n in names:
            for k, v in meta.items():
                if k.endswith(n):
                    return v
        return default

    n_layer = g("block_count", default=0)
    n_embd = g("embedding_length", default=0)
    n_head = g("head_count", default=0)
    n_kv = g("head_count_kv", default=n_head)
    n_ff = g("feed_forward_length", default=0)
    klen = g("attention.key_length", default=n_embd // max(n_head, 1))
    vlen = g("attention.value_length", default=klen)
    ctx_train = g("context_length", default=0)

    print("\n=== 推导 ===")
    print(f"  层数 {n_layer}  隐藏 {n_embd}  注意力头 {n_head}  KV 头 {n_kv}")
    print(f"  head_dim {klen}/{vlen}  FFN {n_ff}")
    print(f"  训练上下文上限 {ctx_train}")

    # 哪些层是"普通注意力"：元数据里如果有逐层的 attention 类型就用它，
    # 否则按之前实测的 i%4==3 规律（64 层里 16 层全注意力）推。
    per_layer = None
    for k, v in meta.items():
        if "attention" in k and isinstance(v, list) and len(v) == n_layer:
            per_layer = (k, v)
            break
    if per_layer:
        k, v = per_layer
        n_full = sum(1 for x in v if str(x) in ("1", "full", "Full", "b'full'"))
        print(f"  逐层注意力类型来自 {k}：{v[:8]}… → 全注意力 {n_full} 层")
    else:
        n_full = n_layer // 4
        print(f"  元数据里没有逐层类型，按实测规律 i%4==3 推：全注意力 {n_full} 层")

    kbytes = 2 * n_kv * klen * 2          # K 和 V，fp16
    print(f"\n  每 token 的 KV = 2(K,V) × {n_kv} 头 × {klen} 维 × 2 字节 "
          f"= {kbytes / 1024:.1f} KiB/层")
    for ctx in (8192, 16384, 32768, 65536, 131072, 262144):
        mib = ctx * kbytes * n_full / 2**20
        print(f"    上下文 {ctx:>7} → 全注意力层 KV 合计 {mib:8.1f} MiB"
              + ("  ← 界面提供的一档" if ctx in (8192, 16384, 32768) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
