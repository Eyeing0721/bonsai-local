#!/usr/bin/env python3
"""同一提示跑多个 LoRA 组合，用来定预设的权重。

为什么要写成脚本而不是内联命令：PowerShell 的 Start-Process 在这台机器上
反复出问题（环境里有大小写重复的变量会直接抛异常），而且进程/端口管理容易
出现"健康检查过了但其实是残留进程"。Python 的 subprocess 一直很稳。

用法:
    python build/compare_loras.py --model <三值包> --core <破限.gguf> \
        --arm "权重0.4=E:\\...\\x0.4.gguf" --arm "权重0.7=..." --arm "权重1.0=..."
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 420.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"引擎提前退出，码 {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200 and json.loads(r.read()).get("status") == "ok":
                    return
        except Exception:                                       # noqa: BLE001
            pass
        time.sleep(0.6)
    raise RuntimeError("等待引擎就绪超时")


def ask(port: int, prompt: str, n: int) -> str:
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n, "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read())
    return j["choices"][0]["message"]["content"].strip()


def run(label: str, bin_dir: Path, model: Path, loras: list[Path], prompts: list[str],
        extra_path: list[Path], log_dir: Path, max_tokens: int) -> None:
    port = free_port()
    cmd = [str(bin_dir / "llama-server.exe"), "-m", str(model),
           "--host", "127.0.0.1", "--port", str(port), "-c", "4096", "-np", "1",
           "-ngl", "99", "--jinja", "--no-warmup", "--no-webui", "--alias", "bonsai"]
    for l in loras:
        cmd += ["--lora", str(l)]

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(bin_dir)] + [str(p) for p in extra_path]
                                  + [env.get("PATH", "")])
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"cmp-{label.replace('/', '_').replace(' ', '_')}.log"
    fh = log.open("w", encoding="utf-8", errors="replace")
    print(f"\n{'=' * 70}\n── {label} ──   （{len(loras)} 个 LoRA）")
    proc = subprocess.Popen(cmd, cwd=str(bin_dir), stdout=fh,
                            stderr=subprocess.STDOUT, env=env,
                            creationflags=CREATE_NO_WINDOW)
    try:
        wait_ready(port, proc)
        for p in prompts:
            print(f"  Q: {p}")
            answer = ask(port, p, max_tokens)
            print("  A: " + answer.replace("\n", "\n     "))
    except Exception as e:                                      # noqa: BLE001
        print(f"  失败: {e}")
        try:
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]:
                print(f"    | {line}")
        except Exception:                                       # noqa: BLE001
            pass
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        fh.close()
        time.sleep(3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--core", default="", help="始终挂载的核心适配器（如破限）")
    ap.add_argument("--arm", action="append", default=[],
                    help="标签=LoRA路径，可重复")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--prompt-file", default="",
                    help="UTF-8 文本文件，用一行 --- 分隔多段提示词")
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--bin", default=r"E:\src\llama-prism\build-cuda-multi\bin")
    ap.add_argument("--extra-path", action="append", default=[r"E:\cuda\bin"])
    ap.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "bench-logs"))
    args = ap.parse_args()

    model = Path(args.model)
    core = [Path(args.core)] if args.core else []
    prompts = list(args.prompt)
    if args.prompt_file:
        raw = Path(args.prompt_file).read_text(encoding="utf-8")
        prompts += [b.strip() for b in raw.split("\n---\n") if b.strip()]
    if not prompts:
        prompts = ["用中文写一段描写秋天黄昏的文字，两百字左右。"]
    bin_dir, log_dir = Path(args.bin), Path(args.log_dir)
    extra = [Path(p) for p in args.extra_path]

    for spec in args.arm:
        label, _, path = spec.partition("=")
        run(label, bin_dir, model, core + [Path(path)], prompts, extra, log_dir,
            args.max_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
