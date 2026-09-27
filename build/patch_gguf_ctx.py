#!/usr/bin/env python3
"""改一份 GGUF 副本的 context_length 元数据，用来做长上下文针测试。

为什么需要这个：llama-server 会把单个会话的槽位上下文截到模型的
n_ctx_train（= GGUF 里的 context_length），而且**没有开关能绕过**
（tools/server/server-context.cpp:1159）。这个截断本身是合理的安全网 ——
防止用户在不知情的情况下外推到质量未知的区间。

做针测试时必须先掀掉它，否则 -c 开到 300K 也只会有 256K 真正可用。这里只改
元数据里的那一个 uint32，**张量数据一个字节都不动**，而且永远改副本。

用法:
    python build/patch_gguf_ctx.py 源.gguf 目标.gguf 1048576
"""
from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# GGUF 元数据类型编号 -> (结构体格式, 定长字节数)
SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
STRING, ARRAY = 8, 9


def read_string(buf: bytes, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    s = buf[off:off + n].decode("utf-8", "replace")
    return s, off + n


def skip_value(buf: bytes, off: int, vtype: int) -> int:
    if vtype in SCALAR:
        return off + SCALAR[vtype][1]
    if vtype == STRING:
        _, off = read_string(buf, off)
        return off
    if vtype == ARRAY:
        (etype,) = struct.unpack_from("<I", buf, off)
        off += 4
        (count,) = struct.unpack_from("<Q", buf, off)
        off += 8
        if etype in SCALAR:
            return off + SCALAR[etype][1] * count
        if etype == STRING:
            for _ in range(count):
                _, off = read_string(buf, off)
            return off
        raise ValueError(f"数组元素类型 {etype} 不支持")
    raise ValueError(f"未知类型 {vtype}")


def find_kv(buf: bytes, key_want: str) -> tuple[int, int, object]:
    """-> (值起始偏移, 类型, 当前值)"""
    if buf[:4] != b"GGUF":
        raise ValueError("不是 GGUF 文件（magic 不对）")
    version, = struct.unpack_from("<I", buf, 4)
    off = 8 + 8 + 8                      # version 之后是 tensor_count + kv_count
    (kv_count,) = struct.unpack_from("<Q", buf, 8 + 8)
    for _ in range(kv_count):
        key, off = read_string(buf, off)
        (vtype,) = struct.unpack_from("<I", buf, off)
        off += 4
        vstart = off
        if key == key_want:
            if vtype in SCALAR:
                fmt, _n = SCALAR[vtype]
                val, = struct.unpack_from(fmt, buf, off)
            else:
                val = f"<类型 {vtype}>"
            return vstart, vtype, val
        off = skip_value(buf, off, vtype)
    raise KeyError(f"元数据里没有 {key_want}（版本 {version}）")


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    src, dst, new_ctx = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    if not src.exists():
        print(f"源文件不存在：{src}")
        return 1
    if src.resolve() == dst.resolve():
        # 只改副本是硬规矩：原模型还要给产品用
        print("拒绝原地修改，源和目标是同一个文件")
        return 1

    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        print(f"复制 {src.name} -> {dst.name}  ({src.stat().st_size / 2**30:.2f} GiB)")
        shutil.copyfile(src, dst)

    buf = bytearray(dst.read_bytes())
    for key in ("qwen35.context_length", "llama.context_length",
                "general.context_length"):
        try:
            off, vtype, old = find_kv(bytes(buf), key)
        except KeyError:
            continue
        if vtype != 4:
            print(f"{key} 类型是 {vtype}，不是 UINT32，拒绝改")
            return 1
        struct.pack_into("<I", buf, off, new_ctx)
        print(f"{key}: {old} -> {new_ctx}")
        dst.write_bytes(bytes(buf))
        # 读回来确认
        _, _, check = find_kv(dst.read_bytes(), key)
        print(f"回读校验: {check}  {'OK' if check == new_ctx else '失败'}")
        return 0 if check == new_ctx else 1
    print("没找到任何 context_length 字段")
    return 1


if __name__ == "__main__":
    sys.exit(main())
