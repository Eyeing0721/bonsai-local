#!/usr/bin/env python3
"""同一模型、同一提示、同一随机种子，只切换 LoRA 开关的对照实验。

为什么不一个适配器起一次服务：模型加载要 30-60 秒，而 llama.cpp 的
POST /lora-adapters 能在运行时把任意适配器的 scale 设成 0 或 1。所以把参战
的适配器一次性全挂上，之后靠开关做对照，差异就只来自适配器本身 —— 连加载
顺序、显存碎片、KV 状态都一样。

用法:
    python build/duel_loras.py --model <三值包> \
        --arm "无适配器=" --arm "破限=abl-lora-t.gguf" \
        --arm "攻击者3+破限=abl-lora-t.gguf,attacker-v3.gguf" \
        --prompt-file build/dev-prompts/redteam3.txt
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


def api(port: int, path: str, payload=None, timeout: float = 900.0):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else None


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"引擎退出，码 {proc.returncode}")
        try:
            if api(port, "/health", timeout=4).get("status") == "ok":
                return
        except Exception:                                       # noqa: BLE001
            pass
        time.sleep(0.7)
    raise RuntimeError("等待引擎就绪超时")


def dump(path: str, model: Path, prompts: list[str], answers: list[dict]) -> None:
    """每答完一题就落盘。

    之前是整轮跑完才写文件，结果中途被打断就什么都不剩 —— 一组要等好几分钟，
    重来一次代价太大。宁可多写几次磁盘。
    """
    if not path:
        return
    Path(path).write_text(
        json.dumps({"model": str(model), "prompts": prompts, "answers": answers},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", action="append", required=True,
                    help="标签=文件1,文件2,...（空表示不带任何适配器）")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=380)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--bin", default=r"E:\src\llama-prism\build-cuda-multi\bin")
    ap.add_argument("--extra-path", action="append", default=[r"E:\cuda\bin"])
    ap.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "bench-logs"))
    ap.add_argument("--json-out", default="",
                    help="把每个对照组的完整回答写成 JSON，交给 score_redteam.py 判分")
    args = ap.parse_args()

    model = Path(args.model)
    prompts = [b.strip() for b in
               Path(args.prompt_file).read_text(encoding="utf-8").split("\n---\n") if b.strip()]

    arms: list[tuple[str, list[Path]]] = []
    for spec in args.arm:
        label, _, files = spec.partition("=")
        arms.append((label, [Path(f) for f in files.split(",") if f.strip()]))

    # 所有参战适配器取并集，一次性挂载；之后靠 scale 开关选谁生效
    pool: list[Path] = []
    for _, files in arms:
        for f in files:
            if f not in pool:
                pool.append(f)

    port = free_port()
    bin_dir, log_dir = Path(args.bin), Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "duel.log"

    cmd = [str(bin_dir / "llama-server.exe"), "-m", str(model),
           "--host", "127.0.0.1", "--port", str(port),
           "-c", str(args.ctx), "-np", "1", "-ngl", "99",
           "--jinja", "--no-warmup", "--no-webui", "--alias", "bonsai"]
    for f in pool:
        cmd += ["--lora", str(f)]

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(bin_dir)] + [str(p) for p in args.extra_path]
                                  + [env.get("PATH", "")])

    print(f"参战适配器 {len(pool)} 个，对照组 {len(arms)} 个，提示 {len(prompts)} 条")
    for i, f in enumerate(pool):
        print(f"  [{i}] {f.name}")
    print(f"温度={args.temp}  种子={args.seed}  上限={args.max_tokens} tok\n")

    fh = log.open("w", encoding="utf-8", errors="replace")
    answers: list[dict] = []
    proc = subprocess.Popen(cmd, cwd=str(bin_dir), stdout=fh,
                            stderr=subprocess.STDOUT, env=env,
                            creationflags=CREATE_NO_WINDOW)
    try:
        wait_ready(port, proc)
        loaded = api(port, "/lora-adapters")
        print("引擎已就绪，挂载情况:")
        for a in loaded:
            print(f"  [{a['id']}] {Path(a['path']).name}  scale={a['scale']}")
        print()

        for label, files in arms:
            want = {pool.index(f) for f in files}
            api(port, "/lora-adapters",
                [{"id": i, "scale": 1.0 if i in want else 0.0}
                 for i in range(len(pool))])
            on = ", ".join(f.name for f in files) or "（裸模型）"
            print("=" * 78)
            print(f"── {label} ──   实际生效: {on}")
            print("=" * 78)
            for qi, p in enumerate(prompts):
                head = p.splitlines()[0][:60]
                print(f"\n  Q: {head}{' …' if len(p.splitlines()[0]) > 60 else ''}")
                try:
                    r = api(port, "/v1/chat/completions", {
                        "messages": [{"role": "user", "content": p}],
                        "max_tokens": args.max_tokens,
                        "temperature": args.temp, "seed": args.seed,
                        "chat_template_kwargs": {"enable_thinking": False},
                    })
                    out = r["choices"][0]["message"]["content"].strip()
                    finish = r["choices"][0].get("finish_reason")
                    rec_tokens = (r.get("usage") or {}).get("completion_tokens")
                    answers.append({"arm": label, "files": [f.name for f in files],
                                    "prompt_index": qi, "prompt": p, "answer": out,
                                    "finish_reason": finish,
                                    "completion_tokens": rec_tokens})
                    flag = "" if finish == "stop" else f"  [未写完: {finish}]"
                    print(f"  A:{flag} " + out.replace("\n", "\n     "))
                    print(f"     〔{len(out)} 字，{rec_tokens} tok〕", flush=True)
                except Exception as e:                          # noqa: BLE001
                    answers.append({"arm": label, "prompt_index": qi, "prompt": p,
                                    "answer": f"<失败 {e}>", "finish_reason": "error"})
                    print(f"  A: <失败 {e}>", flush=True)
                dump(args.json_out, model, prompts, answers)
            print(flush=True)
    except Exception as e:                                      # noqa: BLE001
        print(f"启动失败: {e}")
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]:
            print(f"  | {line}")
        return 1
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                proc.kill()
        fh.close()
        if args.json_out and answers:
            Path(args.json_out).write_text(
                json.dumps({"model": str(model), "prompts": prompts,
                            "answers": answers}, ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(f"完整回答已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
