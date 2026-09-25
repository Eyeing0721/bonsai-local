#!/usr/bin/env python3
"""验证向量服务：pooling 对不对、语义相似度有没有意义。

用错 pooling 不会报错，只会让相似度变成噪声。所以这里直接量三件事：
  1. 同义（中英各一句）应该比 无关 明显高
  2. 完全相同的句子余弦必须 = 1
  3. 中文语义近的一对（猫/小猫）要高于语义远的一对（猫/汽车）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonsai.config import Settings                      # noqa: E402
from bonsai.embed import EmbedServer, embed_model_path  # noqa: E402


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def main() -> int:
    settings = Settings(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    model = embed_model_path(settings)
    print(f"向量模型: {model}")
    print(f"  存在: {model.exists()}  {model.stat().st_size if model.exists() else 0:,} B")

    server = EmbedServer(settings)
    try:
        server.start(model)
        print(f"服务已起，端口 {server.port}")

        ok, msg = server.selftest()
        print(f"\n[自检] {'通过' if ok else '不通过'} —— {msg}")

        pairs = [
            ("堆溢出利用", "heap overflow exploitation", "跨语言同义"),
            ("堆溢出利用", "堆溢出怎么利用", "中文近义"),
            ("小猫", "猫", "语义近"),
            ("猫", "汽车", "语义远"),
            ("CVE-2018-8120 提权", "win32k 本地提权漏洞", "领域相关"),
            ("CVE-2018-8120 提权", "红烧肉的做法", "领域无关"),
        ]
        vecs = server.embed([p[0] for p in pairs] + [p[1] for p in pairs])
        n = len(pairs)
        print(f"\n向量维度 = {len(vecs[0])}\n")
        print(f"{'类型':<12}{'句子 A':<26}{'句子 B':<26}{'余弦':>8}")
        for i, (a, b, kind) in enumerate(pairs):
            print(f"{kind:<12}{a:<26}{b:<26}{cosine(vecs[i], vecs[n + i]):>8.3f}")

        same = server.embed(["完全一样的句子", "完全一样的句子"])
        print(f"\n相同句子的余弦 = {cosine(same[0], same[1]):.6f}  （应为 1.000000）")
        print(f"模长 = {sum(x * x for x in same[0]) ** 0.5:.6f}  （应为 1.000000）")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
