#!/usr/bin/env python3
"""列出 gguf 包里每个模块真正导入的第三方包（用 AST，不看注释）。

起因：打包时 torch / tensorflow / transformers 整个被收进来，构建都失败了。
    gguf 是唯一新进入依赖图的包，所以先精确搞清它到底依赖什么。
第一次用正则扫，被一行注释里的 "transformers" 骗了。
"""

from __future__ import annotations

import ast
import pathlib
import sys

STDLIB_OK = {
    "gguf", "typing", "os", "sys", "json", "re", "math", "struct", "pathlib",
    "collections", "dataclasses", "enum", "functools", "itertools", "logging",
    "warnings", "io", "abc", "numpy", "tempfile", "shutil", "hashlib", "time",
    "textwrap", "contextlib", "copy", "uuid", "weakref", "types", "inspect",
    "__future__", "array", "base64", "zlib", "gzip", "codecs", "string",
}
HEAVY = {"torch", "transformers", "tensorflow", "sklearn", "cv2", "diffusers",
         "gradio", "timm", "librosa", "numba", "peft", "datasets", "triton",
         "unsloth", "sentencepiece", "scipy", "pandas", "PIL", "matplotlib"}


def main() -> int:
    import gguf
    base = pathlib.Path(gguf.__file__).parent
    print(f"gguf 包位置：{base}\n")
    all_heavy: set[str] = set()
    for f in sorted(base.glob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            print(f"  {f.name}: 解析失败 {e}")
            continue
        mods: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    mods.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
        third = sorted(m for m in mods if m not in STDLIB_OK)
        hit = sorted(mods & HEAVY)
        all_heavy |= set(hit)
        mark = "  ← 重型" if hit else ""
        print(f"  {f.name:<22} {third}{mark}")
    print(f"\n合计重型依赖：{sorted(all_heavy) if all_heavy else '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
