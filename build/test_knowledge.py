#!/usr/bin/env python3
"""知识库自测：切块、BM25、向量、融合检索。

测试资料故意选的是主模型**实测答错**的那些事实（CVE-2018-8120 的真身、
SeImpersonate 的利用工具、tcache 的防护机制）。所以这个测试同时回答两个问题：
  1. 检索管线本身对不对；
  2. 把正确答案放进资料里，它能不能被查出来。

不加载 27B 模型，纯检索层，几秒钟跑完。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonsai import knowledge as kb                       # noqa: E402
from bonsai.config import Settings                       # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'✓' if ok else '✗'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


FACTS = """# 内部安全笔记

## Windows 本地提权

CVE-2018-8120 是 win32k.sys 里的空指针解引用漏洞，影响 Windows 7 SP1 和
Windows Server 2008 R2，属于本地权限提升，不是缓冲区溢出，也和 wininet.dll
没有任何关系。利用成功后可以拿到 SYSTEM。

SeImpersonatePrivilege 是 Windows 提权里最重要的一个特权。服务账户和 IIS
应用池默认就带着它。配合 GodPotato、JuicyPotato、RoguePotato、PrintSpoofer
这些工具，可以把该特权转化成 SYSTEM。检查方法是 whoami /priv。

AlwaysInstallElevated 是另一个常见配置缺陷。注册表里
HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\Installer 和 HKCU 下同一个键
的 AlwaysInstallElevated 都为 1 时，普通用户可以安装任意 MSI 并拿到 SYSTEM。

## Linux 堆利用

glibc 2.32 引入了 safe-linking 保护：空闲块的 fd 指针不再直接存下一个块的
地址，而是存 (addr >> 12) ^ pos。要绕过它需要先泄露一个堆地址，把目标地址
按同样方式异或回去。

tcache poisoning 是 2.26 之后最常见的任意地址写手法：改写 tcache 空闲链表里
某个块的 fd，下次申请就会返回任意地址。2.34 起 __free_hook 和 __malloc_hook
被移除，改打 exit_funcs 或者走 FSOP。

## AMSI

AMSI 只扫描脚本文本和 .NET 程序集加载，不扫描已经写进 RWX 内存的原始
shellcode。所以"给 shellcode 做 XOR 混淆"躲的不是 AMSI。

真正的绕过是给 amsi.dll 里的 AmsiScanBuffer 打内存补丁：VirtualProtect 改成
可写，把开头几个字节改成直接返回，再改回来。另一种是反射改写
System.Management.Automation.AmsiUtils 的 amsiInitFailed 字段。
"""


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="kbtest-"))
    try:
        settings = Settings()
        settings.set("data_dir", str(tmp))
        settings.save()

        print("\n[1] 切块")
        chunks = kb.chunk_text(FACTS)
        check("切出多块", len(chunks) >= 3, f"{len(chunks)} 块")
        check("没有空块", all(c.strip() for c in chunks))
        check("块长不超上限", all(len(c) <= kb.CHUNK_CHARS + kb.CHUNK_OVERLAP + 40
                                  for c in chunks),
              f"最长 {max(len(c) for c in chunks)}")
        check("短文本不切", kb.chunk_text("只有一句话。") == [])   # <8 字被过滤

        print("\n[2] 分词")
        t = kb.tokenize("CVE-2018-8120 提权 heap overflow")
        check("英文小写词元", "heap" in t and "overflow" in t)
        check("中文二元组", "提权" in t, f"样例 {[x for x in t if len(x) == 2][:6]}")
        check("数字保留", any("2018" in x for x in t))

        print("\n[3] 入索引（纯 BM25）")
        store = kb.Knowledge(settings)
        store.load()
        doc_file = tmp / "notes.md"
        doc_file.write_text(FACTS, encoding="utf-8")
        d = store.add_file(doc_file)
        check("文档已入库", d.chunks == len(chunks), f"{d.chunks} 块")
        st = store.state()
        check("state 报告块数", st["chunks"] == len(chunks))
        check("dense 未启用", st["dense"] is False)

        print("\n[4] 关键词检索（BM25 单独）")
        cases = [
            ("CVE-2018-8120 是什么漏洞", "win32k"),
            ("SeImpersonatePrivilege 用什么工具", "Potato"),
            ("safe-linking 怎么绕过", "glibc 2.32"),
            ("AmsiScanBuffer 补丁怎么打", "VirtualProtect"),
            ("AlwaysInstallElevated 注册表路径", "AlwaysInstallElevated"),
        ]
        for query, expect in cases:
            hits = store.search(query, k=3)
            joined = "\n".join(h.text for h in hits)
            top = hits[0].doc_name if hits else "（无）"
            check(f"「{query}」", expect.lower() in joined.lower(),
                  f"top1 命中 {expect!r} ？")

        print("\n[5] 无关查询不应乱命中")
        hits = store.search("红烧肉怎么做才好吃", k=3)
        check("无关查询得分很低", all(h.score < 0.05 for h in hits),
              f"最高分 {hits[0].score if hits else 0}")

        print("\n[6] 重建索引后仍然可用（持久化）")
        store2 = kb.Knowledge(settings)
        store2.load()
        check("文档清单已持久化", len(store2.docs) == 1)
        check("块已持久化", len(store2.texts) == len(chunks))
        hits = store2.search("CVE-2018-8120 是什么漏洞", k=3)
        check("重载后仍能检索", bool(hits) and "win32k" in hits[0].text)

        print("\n[7] 文件夹导入与去重")
        sub = tmp / "docs"
        sub.mkdir()
        (sub / "a.txt").write_text("这是文档 A，讲的是 nginx 的 fastcgi_split_path_info 配置。",
                                   encoding="utf-8")
        (sub / "b.txt").write_text("这是文档 B，讲的是 PHP 的 auto_prepend_file。",
                                   encoding="utf-8")
        (sub / "c.bin").write_bytes(b"\x00\x01\x02")
        res = store2.add_dir(sub)
        check("导入 2 个文本", len(res["added"]) == 2, json.dumps(res["skipped"]))
        check("跳过二进制 1 个", res["skipped"] == 1)
        docs_now = len(store2.docs)
        check("文档数 = 3", docs_now == 3, f"{docs_now}")

        print("\n[8] 删除文档")
        store2.remove(res["added"][0]["id"])
        check("文档数 = 2", len(store2.docs) == 2)
        check("块同步减少", len(store2.texts) < len(chunks) + 2)

        print("\n[9] 上下文拼装")
        ctx, used = store2.build_context("CVE-2018-8120 是什么漏洞", k=3)
        check("拼出上下文", bool(ctx) and "win32k" in ctx, f"{len(ctx)} 字，{len(used)} 条")
        msgs = kb.build_messages(
            [{"role": "system", "content": "你是助手"},
             {"role": "user", "content": "CVE-2018-8120 是什么"}], ctx)
        check("插在 system 之后", msgs[1]["role"] == "system" and "win32k" in msgs[1]["content"])
        check("用户消息未改动", msgs[-1]["content"] == "CVE-2018-8120 是什么")
        check("原有 system 保留", msgs[0]["content"] == "你是助手")
        msgs2 = kb.build_messages([{"role": "user", "content": "x"}], ctx)
        check("无 system 时插到最前", msgs2[0]["role"] == "system")

        print("\n[10] 空库不炸")
        empty = kb.Knowledge(settings)
        empty.load()
        empty.clear()
        check("空库检索返回空", empty.search("随便问问") == [])
        check("空库上下文为空", empty.build_context("随便问问") == ("", []))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if FAILED:
        print(f"失败 {len(FAILED)} 项：")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
