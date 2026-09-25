#!/usr/bin/env python3
"""验证第三方 LoRA 能不能直接挂在三值包上，以及多个 LoRA 能否叠加。

为什么这件事值得单独验证
------------------------
llama.cpp 的 LoRA 分支拿到的是**未旋转的原始激活**（build_lora_mm 里
`ggml_mul_mat(ctx0, lw->a, cur)`，用的是 cur 不是 cur_mm），输出也加在自然基上。
所以任何在未旋转底座上训练的 LoRA，理论上不需要任何转换就能挂到这个
Hadamard 旋转过的三值包上。理论归理论，这里跑一遍。

同时验证叠加：破限 LoRA + 人格/文风 LoRA 一起挂，这是产品化的前提。

用法:
    python build/test_lora_stack.py --model <三值包> --lora a.gguf --lora b.gguf
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


def ask(port: int, prompt: str, n: int = 160) -> str:
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
    return j["choices"][0]["message"]["content"].replace("\n", " ").strip()


def run(label: str, bin_dir: Path, model: Path, loras: list[Path],
        prompts: list[str], extra_path: list[Path], log: Path) -> None:
    port = free_port()
    cmd = [str(bin_dir / "llama-server.exe"), "-m", str(model),
           "--host", "127.0.0.1", "--port", str(port), "-c", "4096", "-np", "1",
           "-ngl", "99", "--jinja", "--no-warmup", "--alias", "bonsai"]
    for l in loras:
        cmd += ["--lora", str(l)]

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(bin_dir)] + [str(p) for p in extra_path]
                                  + [env.get("PATH", "")])
    print(f"\n{'=' * 66}\n=== {label}  ({len(loras)} 个 LoRA)  ===")
    for l in loras:
        print(f"    + {l.name}")
    fh = log.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, cwd=str(bin_dir), stdout=fh,
                            stderr=subprocess.STDOUT, env=env,
                            creationflags=CREATE_NO_WINDOW)
    try:
        wait_ready(port, proc)
        for p in prompts:
            out = ask(port, p)
            print(f"\n  Q: {p}")
            print(f"  A: {out[:260]}")
    except Exception as e:                                      # noqa: BLE001
        print(f"  失败: {e}")
        try:
            tail = Path(fh.name).read_text(encoding="utf-8", errors="replace").splitlines()[-10:]
            for line in tail:
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
    ap.add_argument("--lora", action="append", default=[])
    ap.add_argument("--bin", default=r"E:\src\llama-prism\build-cuda-multi\bin")
    ap.add_argument("--extra-path", action="append", default=[r"E:\cuda\bin"])
    ap.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "bench-logs"))
    args = ap.parse_args()

    model = Path(args.model)
    loras = [Path(x) for x in args.lora]
    bin_dir = Path(args.bin)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    prompts = [
        "用中文写一段描写秋天黄昏的文字。",
        "解释一下什么是浮点数精度损失。",
    ]

    run("只挂破限 LoRA（基线）", bin_dir, model, loras[:1], prompts,
        [Path(p) for p in args.extra_path], log_dir / "lora-base.log")
    if len(loras) > 1:
        run("叠加人格/文风 LoRA", bin_dir, model, loras,
            prompts, [Path(p) for p in args.extra_path], log_dir / "lora-stack.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
