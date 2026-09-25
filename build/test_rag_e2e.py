#!/usr/bin/env python3
"""端到端验证：资料到底有没有把知识补上。

做法是同一个问题问两遍 —— 先关掉知识库（对照），再打开（实验）。模型、风格包、
温度、种子全都一样，唯一变量是"有没有查资料"。这样得出的差异只可能来自检索。

选的问题都是主模型实测答错过的：
  CVE-2018-8120  它说成 wininet.dll 缓冲区溢出，真身是 win32k.sys 空指针提权
  SeImpersonate  五个适配器组合一次都没提到过 Potato 家族
  AmsiScanBuffer 它坚持"给 shellcode 做 XOR 就躲开了 AMSI"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

QUESTIONS = [
    "CVE-2018-8120 是哪个组件里的漏洞？属于哪一类？一句话回答。",
    "Windows 上拿到普通用户 shell 后，SeImpersonatePrivilege 这个特权可以配合哪些工具提权？列出工具名。",
    "通过给 AmsiScanBuffer 打内存补丁绕过 AMSI 的原理是什么？为什么给 shellcode 做 XOR 混淆没有用？",
]

DOC = """# 内部安全资料

CVE-2018-8120 位于 win32k.sys，是一个空指针解引用漏洞，属于本地权限提升，
影响 Windows 7 SP1 与 Windows Server 2008 R2。它和 wininet.dll 无关，也不是
缓冲区溢出。

SeImpersonatePrivilege 可以配合 GodPotato、JuicyPotato、RoguePotato、
PrintSpoofer、SweetPotato 这些工具把普通服务账户提升到 SYSTEM。

AMSI 只扫描脚本文本和 .NET 程序集加载，不检查已经写进 RWX 内存的原始
shellcode。所以对 shellcode 做 XOR 混淆并不会躲开 AMSI，因为它根本不经过
AMSI 扫描器。真正的绕过是给 amsi.dll 里的 AmsiScanBuffer 打内存补丁：先用
VirtualProtect 把该函数开头改成可写，写入直接返回的指令，再改回原来的保护属性。
"""


def req(url: str, payload=None, token: str = "", timeout: float = 900.0):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    r = urllib.request.Request(url, data=data,
                               method="POST" if data is not None else "GET")
    if data is not None:
        r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body) if body else None


def ask(port: int, token: str, question: str, n: int = 260) -> str:
    r = req(f"http://127.0.0.1:{port}/v1/chat/completions", {
        "messages": [{"role": "user", "content": question}],
        "max_tokens": n, "temperature": 0.1, "seed": 7,
        "chat_template_kwargs": {"enable_thinking": False},
    }, token)
    return r["choices"][0]["message"]["content"].strip()


def main() -> int:
    data_dir = ROOT / "devdata"
    port = int(os.environ.get("RAG_TEST_PORT", "8128"))
    token = ""      # 等应用起来再从 /app/state 取，避免和磁盘上的旧值不一致

    # 知识库先清空，保证是从零开始
    tmp_doc = data_dir / "ragtest.md"
    tmp_doc.write_text(DOC, encoding="utf-8")

    env = dict(os.environ)
    env["BONSAI_ENGINE_DIR"] = r"E:\src\llama-prism\build-cuda-multi\bin"
    env["BONSAI_CUDA_DIR"] = r"E:\cuda\bin"
    env["BONSAI_EMBED_MODEL"] = r"E:\models\bonsai\embed\Qwen3-Embedding-0.6B-Q8_0.gguf"
    env["PYTHONIOENCODING"] = "utf-8"

    log = (data_dir / "ragtest-app.log").open("w", encoding="utf-8", errors="replace")
    print(f"启动应用（端口 {port}）…")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "bonsai", "--no-window",
         "--data-dir", str(data_dir), "--port", str(port)],
        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
        env=env, creationflags=CREATE_NO_WINDOW)

    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            if proc.poll() is not None:
                print("应用提前退出，日志尾部：")
                print("\n".join((data_dir / "ragtest-app.log")
                                .read_text(encoding="utf-8", errors="replace")
                                .splitlines()[-20:]))
                return 1
            try:
                st = req(base + "/app/state")
                if st.get("engine", {}).get("running") and \
                        st.get("progress", {}).get("stage") == "ready":
                    break
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(2)
        else:
            print("等待就绪超时")
            return 1
        token = req(base + "/app/state")["token"]
        print(f"模型已就绪，令牌 {token[:12]}…\n")

        print("清空知识库 …")
        req(base + "/app/kb/clear", {}, token)

        print("── 对照组：关闭知识库 ──")
        req(base + "/app/kb/enabled", {"enabled": False}, token)
        baseline = {}
        for q in QUESTIONS:
            a = ask(port, token, q)
            baseline[q] = a
            print(f"\nQ: {q}\nA: {a[:400]}")

        print("\n" + "=" * 78)
        print("── 加入资料 ──")
        res = req(base + "/app/kb/add", {"path": str(tmp_doc)}, token)
        if not res.get("ok") or res.get("ok") is False:
            print("加入失败：", res)
            return 1
        st = req(base + "/app/kb", token=token)
        print(f"  文档 {len(st['docs'])} 份，{st['chunks']} 段，语义检索={st['dense']}")

        print("\n── 实验组：打开知识库 ──")
        req(base + "/app/kb/enabled", {"enabled": True}, token)
        withkb = {}
        for q in QUESTIONS:
            a = ask(port, token, q)
            withkb[q] = a
            print(f"\nQ: {q}")
            hits = req(base + "/app/kb", token=token).get("last_hits") or []
            for h in hits[:3]:
                print(f"   [检索命中] {h['doc']}  score={h['score']}  {h['preview'][:50]}…")
            print(f"A: {a[:400]}")

        print("\n" + "=" * 78)
        print("── 自动判定：资料里的关键答案有没有出现在回答里 ──")
        keys = [["win32k"], ["GodPotato", "JuicyPotato", "RoguePotato", "PrintSpoofer", "Potato"],
                ["VirtualProtect", "AmsiScanBuffer", "内存补丁"]]
        gained = 0
        for q, group in zip(QUESTIONS, keys):
            before = any(k.lower() in baseline[q].lower() for k in group)
            after = any(k.lower() in withkb[q].lower() for k in group)
            if after and not before:
                gained += 1
                mark = "✓ 由答不出变成答对"
            elif before and not after:
                mark = "✗ 反而变差了"
            elif after:
                mark = "= 两边都命中"
            else:
                mark = "✗ 两边都没命中"
            print(f"  {mark:<22} 关键词 {group[0]}")
        print(f"\n{gained}/3 题由「答不出」变成「答对」")

        req(base + "/app/kb/clear", {}, token)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        tmp_doc.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
