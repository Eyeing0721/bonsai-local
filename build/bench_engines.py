#!/usr/bin/env python3
"""在同一台机器上对比两个引擎的 prefill 与 decode 速度。

为什么必须实测而不是推断：Vulkan 后端对 PTQ1_0 的处理和 CUDA 不是一回事 ——
CUDA 有 mmq-instance-ptq1_0 和 mmq-config-ampere 里手调的 MMA 分块，而
vulkan-shaders-gen.cpp 里写着一句注释：

    PTQ1_0 has no coopmat2 decoder: dequant_funcs_cm2.glsl carries no PTQ1_0 entry

也就是 PTQ1_0 被排除在 Vulkan 最快的张量核路径之外。所以"会不会掉性能"不能靠
猜，只能靠量。

用法：
    python build/bench_engines.py --model E:\\models\\bonsai\\...gguf \\
        --lora E:\\models\\bonsai\\abl-lora-t.gguf \\
        --engine cuda=E:\\src\\llama-prism\\build-cuda\\bin \\
        --engine vulkan=E:\\src\\llama-prism\\build-vulkan\\bin
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
FILLER = (
    "The quick brown fox jumps over the lazy dog while the engineer reviews a long "
    "document about distributed systems, memory bandwidth and cache behaviour. "
)


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 420.0) -> None:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"引擎提前退出，码 {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200 and json.loads(r.read()).get("status") == "ok":
                    return
        except Exception as e:                                  # noqa: BLE001
            last = type(e).__name__
        time.sleep(0.7)
    raise RuntimeError(f"等待引擎就绪超时（最后 {last}）")


def bench(name: str, engine_dir: Path, model: Path, loras: list[Path],
          ctx: int, prompt_tokens: int, predict: int, log_dir: Path,
          extra_path: list[Path] | None = None, ngl: int = 99) -> dict:
    exe = engine_dir / "llama-server.exe"
    if not exe.exists():
        return {"engine": name, "error": f"没有 {exe}"}

    port = free_port()
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / f"bench-{name}.log").open("w", encoding="utf-8", errors="replace")

    cmd = [str(exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
           "-c", str(ctx), "-np", "1", "-ngl", str(ngl), "--jinja", "--no-warmup",
           "--alias", "bonsai"]
    if loras:
        for l in loras:
            cmd += ["--lora", str(l)]

    env = dict(os.environ)
    # A CUDA build needs its runtime DLLs (cudart/cublas/cublasLt). Without them the
    # process dies instantly with 0xC0000135 and an empty log, which looks like
    # "the engine is broken" rather than "the path is wrong".
    parts = [str(engine_dir)] + [str(p) for p in (extra_path or [])]
    parts.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(parts)

    print(f"\n=== {name} ===")
    print(f"  {engine_dir}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(engine_dir), stdout=log,
                            stderr=subprocess.STDOUT, env=env,
                            creationflags=CREATE_NO_WINDOW)
    try:
        wait_ready(port, proc)
        load_s = time.time() - t0
        print(f"  加载耗时 {load_s:.1f} s")

        words = max(1, prompt_tokens // 2)
        filler = (FILLER * ((words * 2) // len(FILLER.split()) + 2))[: prompt_tokens * 5]
        # 走 chat 接口而不是裸 completion：裸提示在对话模型上会立刻吐 EOS，
        # 那样只能量到 prefill，量不到 decode。这里让长材料后面跟一个明确的
        # 指令，既能撑起 prefill，又能真的生成满 predict 个 token。
        prompt = (f"{filler}\n\n请把上面这段英文材料改写成中文，尽量详细，"
                  f"不少于 {max(200, predict * 2)} 字。")

        r = post(f"http://127.0.0.1:{port}/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": predict,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        t = r.get("timings") or {}
        content = ((r.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        out = {
            "engine": name,
            "load_s": round(load_s, 1),
            "prompt_tokens": t.get("prompt_n"),
            "prompt_tok_s": round(t.get("prompt_per_second") or 0, 1),
            "predicted_tokens": t.get("predicted_n"),
            "decode_tok_s": round(t.get("predicted_per_second") or 0, 1),
            "sample": content[:60].replace("\n", " "),
        }
        print(f"  prefill : {out['prompt_tokens']} tok @ {out['prompt_tok_s']} tok/s")
        print(f"  decode  : {out['predicted_tokens']} tok @ {out['decode_tok_s']} tok/s")
        print(f"  样例输出: {out['sample']}")
        return out
    except Exception as e:                                      # noqa: BLE001
        print(f"  失败: {e}")
        try:
            print("  引擎日志尾部:")
            for line in log.name and Path(log.name).read_text(
                    encoding="utf-8", errors="replace").splitlines()[-8:]:
                print(f"    {line}")
        except Exception:                                       # noqa: BLE001
            pass
        return {"engine": name, "error": str(e)}
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()
        time.sleep(3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", action="append", default=[],
                    help="挂载的 LoRA，可重复；用来量每个 LoRA 的运行时开销")
    ap.add_argument("--engine", action="append", required=True,
                    help="名字=引擎bin目录，可重复")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--prompt-tokens", type=int, default=1000)
    ap.add_argument("--predict", type=int, default=128)
    ap.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "bench-logs"))
    ap.add_argument("--extra-path", action="append", default=[],
                    help="额外加进子进程 PATH 的目录，CUDA 引擎需要指向 CUDA 运行时")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--ngl", type=int, default=99,
                    help="卸载到 GPU 的层数；设 0 可用来判断某引擎是否真的在用 GPU")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        sys.exit(f"模型不存在: {model}")
    loras = [Path(x) for x in args.lora if x]

    results = []
    for spec in args.engine:
        if "=" not in spec:
            sys.exit(f"--engine 需要 名字=目录 的形式，收到 {spec}")
        name, _, d = spec.partition("=")
        results.append(bench(name, Path(d), model, loras, args.ctx,
                             args.prompt_tokens, args.predict, Path(args.log_dir),
                             [Path(p) for p in args.extra_path], args.ngl))

    print("\n" + "=" * 62)
    print(f"{'引擎':<12}{'prefill tok/s':>16}{'decode tok/s':>15}{'加载 s':>10}")
    print("-" * 62)
    for r in results:
        if "error" in r:
            print(f"{r['engine']:<12}{'失败: ' + r['error'][:40]:>41}")
        else:
            print(f"{r['engine']:<12}{r['prompt_tok_s']:>16}{r['decode_tok_s']:>15}{r['load_s']:>10}")
    print("=" * 62)

    ok = [r for r in results if "error" not in r]
    if len(ok) >= 2:
        base = ok[0]
        for r in ok[1:]:
            if base["prompt_tok_s"] and r["prompt_tok_s"]:
                print(f"  {r['engine']} / {base['engine']}: "
                      f"prefill {r['prompt_tok_s'] / base['prompt_tok_s']:.2f}x, "
                      f"decode {r['decode_tok_s'] / base['decode_tok_s']:.2f}x")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"  结果已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
