"""给 LoRA 调权重：把 b 缩放一个倍数，写成新文件。

为什么不能用 llama.cpp 的 `--lora-scaled FNAME:SCALE`
----------------------------------------------------
它的解析是按 ':' 切分的，而 Windows 路径自带盘符冒号
（`E:\\models\\x.gguf:0.7` 会被切成三段）→ 直接报
"lora-scaled format: FNAME:SCALE"。和 `--control-vector-scaled` 是同一个坑。
所以权重改在文件里，应用用朴素的 `--lora` 挂载，外部 API 调用者的行为也一致。

关键点：必须按 GGML 类型分别处理
--------------------------------
第一版只支持 F32，结果把 Q8_0 的 lora_a 当成原始字节重写，形状直接坏掉
（17408x2 变成 18496x2），引擎报 "incorrect shape (hint: maybe wrong base model?)"。
我们自己的适配器恰好是 F32，所以这个 bug 藏了很久。

Q8_0 的缩放有个便宜做法：它的表示是 v = q * d（每 32 个值共享一个 fp16 的 d），
整块乘以 w 完全等价于**只把 d 乘以 w**，不需要反量化再量化，精确无损。
"""

from __future__ import annotations

import struct

import numpy as np

# ggml 类型 id -> 名字
GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8

QK8_0 = 32
BLOCK_Q8_0 = 2 + QK8_0          # fp16 scale + 32 个 int8


def _fp16_bytes_to_array(raw: np.ndarray) -> np.ndarray:
    return raw.view(np.float16).astype(np.float32)


def scale_tensor(raw: bytes, n_elements: int, type_id: int, factor: float) -> bytes:
    """返回按 factor 缩放后的张量原始字节。不支持的量化类型会明确报错。"""
    if type_id == GGML_F32:
        a = np.frombuffer(raw, dtype=np.float32).astype(np.float32) * np.float32(factor)
        return a.tobytes()

    if type_id == GGML_F16:
        a = np.frombuffer(raw, dtype=np.float16).astype(np.float32) * np.float32(factor)
        return a.astype(np.float16).tobytes()

    if type_id == GGML_Q8_0:
        if n_elements % QK8_0:
            raise ValueError(f"Q8_0 元素数 {n_elements} 不是 {QK8_0} 的倍数")
        n_blk = n_elements // QK8_0
        buf = bytearray(raw)
        for b in range(n_blk):
            off = b * BLOCK_Q8_0
            # 只动 fp16 的 scale，q 原样保留 —— 数学上等价于整块缩放
            d = struct.unpack_from("<e", buf, off)[0]
            struct.pack_into("<e", buf, off, d * float(factor))
        return bytes(buf)

    raise ValueError(
        f"不支持的张量类型 id={type_id}。目前支持 F32(0) / F16(1) / Q8_0(8)。"
        f"先用 convert_lora_to_gguf.py 转成这几种再调权重。")


def rescale_lora(src: str, dst: str, factor: float) -> dict:
    """把 LoRA 的 b 缩放 factor 倍，写到 dst。返回统计信息。"""
    import gguf

    rd = gguf.GGUFReader(src)
    w = gguf.GGUFWriter(dst, str(
        (rd.fields["general.architecture"].contents()
         if "general.architecture" in rd.fields else "qwen35")))

    copied = 0
    for f in rd.fields.values():
        name = str(f.name)
        if name == "general.architecture":
            continue
        try:
            v = f.contents()
        except Exception:                                       # noqa: BLE001
            continue
        if isinstance(v, str):
            w.add_string(name, v)
        elif isinstance(v, bool):
            w.add_bool(name, v)
        elif isinstance(v, int):
            w.add_uint32(name, v)
        elif isinstance(v, float):
            w.add_float32(name, v)
        else:
            continue
        copied += 1
    w.add_float32("adapter.rescale_factor", float(factor))

    scaled = kept = 0
    for t in rd.tensors:
        # 两个形状要分清（这是本文件最容易写错的地方）：
        #   t.shape      —— 逻辑形状，ggml 的 ne 顺序
        #   t.data.shape —— numpy 顺序，量化张量的末维是「字节数」
        # add_tensor 的 raw_shape 要的是后者（它内部会做 byte->逻辑 的换算）。
        ne_shape = tuple(int(x) for x in t.shape)
        d = np.asarray(t.data)
        n = int(np.prod(ne_shape)) if ne_shape else 0
        type_id = int(t.tensor_type)
        raw = d.tobytes()

        if t.name.endswith(".lora_b"):
            out = scale_tensor(raw, n, type_id, factor)
            scaled += 1
        else:
            out = raw
            kept += 1

        if type_id == GGML_F32:
            w.add_tensor(t.name, np.frombuffer(out, dtype=np.float32).reshape(d.shape))
        elif type_id == GGML_F16:
            w.add_tensor(t.name, np.frombuffer(out, dtype=np.float16).reshape(d.shape))
        elif type_id == GGML_Q8_0:
            w.add_tensor(t.name, np.frombuffer(out, dtype=np.uint8).reshape(d.shape),
                         raw_shape=d.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            raise ValueError(f"{t.name}: 不支持的张量类型 id={type_id}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"scaled": scaled, "kept": kept, "metadata": copied}
