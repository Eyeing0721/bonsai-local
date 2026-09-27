#!/usr/bin/env python3
"""通用 GGUF 元数据 + 张量构成读取。

不针对任何特定模型：把所有 KV 打出来，再按张量名后缀归类，算清
「哪些权重可以放 CPU、哪些必须留 GPU」。

MoE 模型的关键问题：专家权重占绝大多数体积，但每个 token 只激活几个。
llama.cpp 的 --n-cpu-moe / -cmoe N 就是把 N 层的专家张量甩给 CPU —— 所以
要先知道「专家张量占多少、非专家张量占多少」，才能判断放不放得下。

用法:
    python build/gguf_info.py E:\\models\\bonsai\\alts\\xxx.gguf
    python build/gguf_info.py xxx.gguf --tensors     # 列出张量明细
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def fmt(v, maxlen=140):
    if isinstance(v, (list, tuple)):
        if len(v) > 10:
            return f"[{len(v)} 项] {list(v[:6])}…"
        return str(list(v))
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    s = str(v)
    return s if len(s) <= maxlen else s[:maxlen] + "…"


def val(field):
    try:
        v = field.contents()
    except Exception:
        v = getattr(field, "parts", None)
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    return v


def classify(name: str) -> str:
    """把张量名归到「用途」桶里，用于算 CPU/GPU 分配。"""
    n = name.lower()
    if "exps" in n or "experts" in n or "_moe" in n:
        return "MoE 专家"
    if "attn_" in n or "attention" in n:
        return "注意力"
    if "ffn_" in n or "feed_forward" in n:
        return "FFN(非专家)"
    if "token_embd" in n or "embed" in n:
        return "词嵌入"
    if "output" in n or "lm_head" in n:
        return "输出头"
    if "norm" in n:
        return "归一化"
    if "ssm" in n or "conv" in n or "dt_" in n or "a_log" in n or "gdn" in n:
        return "GDN/SSM"
    return "其他"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tensors", action="store_true", help="列出每个张量")
    args = ap.parse_args()

    p = Path(args.model)
    if not p.exists():
        print(f"不存在：{p}")
        return 1

    try:
        from gguf import GGUFReader
    except ImportError:
        print("需要 gguf 模块：pip install gguf")
        return 1

    r = GGUFReader(str(p))
    print(f"模型 {p.name}")
    print(f"文件 {p.stat().st_size:,} B = {p.stat().st_size / 2**30:.2f} GiB")
    print(f"张量 {len(r.tensors)} 个\n")

    print("=== 元数据 ===")
    skip = ("tokenizer.ggml.tokens", "tokenizer.ggml.scores",
            "tokenizer.ggml.merges", "tokenizer.ggml.token_type",
            "tokenizer.chat_template")
    for k in sorted(r.fields):
        if any(k.startswith(s) for s in skip):
            print(f"  {k:<48} (省略)")
            continue
        print(f"  {k:<48} {fmt(val(r.fields[k]))}")

    print("\n=== 张量构成（按用途）===")
    agg = defaultdict(lambda: [0, 0])          # 桶 -> [个数, 字节]
    per_type = defaultdict(lambda: [0, 0])
    total = 0
    for t in r.tensors:
        nbytes = int(t.n_bytes)
        total += nbytes
        b = classify(t.name)
        agg[b][0] += 1
        agg[b][1] += nbytes
        qt = getattr(t.tensor_type, "name", str(t.tensor_type))
        per_type[qt][0] += 1
        per_type[qt][1] += nbytes
    for b, (n, sz) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {b:<14} {n:>5} 个  {sz / 2**30:>8.3f} GiB  {100.0 * sz / total:>5.1f}%")
    print(f"  {'合计':<14} {sum(v[0] for v in agg.values()):>5} 个  {total / 2**30:>8.3f} GiB")

    print("\n=== 量化类型分布 ===")
    for qt, (n, sz) in sorted(per_type.items(), key=lambda kv: -kv[1][1]):
        print(f"  {qt:<22} {n:>5} 个  {sz / 2**30:>8.3f} GiB  {100.0 * sz / total:>5.1f}%")

    if args.tensors:
        print("\n=== 张量明细（按体积降序，前 60）===")
        ts = sorted(r.tensors, key=lambda t: -int(t.n_bytes))
        for t in ts[:60]:
            qt = getattr(t.tensor_type, "name", str(t.tensor_type))
            print(f"  {int(t.n_bytes) / 2**20:>9.2f} MiB  {qt:<16} {t.name}")

    # MoE 相关的关键结论
    exp = agg.get("MoE 专家", [0, 0])[1]
    if exp:
        print("\n=== MoE 分配建议 ===")
        print(f"  专家权重 {exp / 2**30:.3f} GiB（{100.0 * exp / total:.1f}%）")
        print(f"  非专家   {(total - exp) / 2**30:.3f} GiB")
        print("  llama.cpp 用 -ngl 99 把非专家全部放 GPU，再用 -cmoe N 控制"
              "多少层的专家留在 CPU；\n  N 越大显存越省、但每 token 要过 PCIe/内存的次数越多。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
