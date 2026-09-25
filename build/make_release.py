#!/usr/bin/env python3
"""打出一个可以挂到 GitHub Releases 的版本。

产物：
    dist/BonsaiLocal.exe                单文件应用（含界面与方向适配器）
    dist/engines/engine-<variant>.zip   推理引擎包，首次运行时按硬件下载
    dist/engines.json                   引擎清单（应用读它决定下载哪个）
    dist/SHA256SUMS                     校验和

用法：
    python build/make_release.py                     # 只打 exe
    python build/make_release.py --with-engines      # 连引擎包一起打
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
ENGINES_OUT = DIST / "engines"

# 引擎构建产物的来源。发布时改这两个环境变量指向自己的构建即可。
CUDA_BIN = Path(os.environ.get("BONSAI_CUDA_BIN", r"E:\src\llama-prism\build-cuda\bin"))
CPU_BIN = Path(os.environ.get("BONSAI_CPU_BIN", r"E:\src\llama-prism\build-cpu\bin"))
CUDA_RUNTIME = Path(os.environ.get("BONSAI_CUDA_RUNTIME", r"E:\cuda\bin"))

# CUDA 构建依赖这些运行时 DLL；放进引擎包的 cuda/ 子目录，应用启动时会加进 PATH。
CUDA_DLLS = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "nvJitLink_120_0.dll"]

# 引擎包里要带的文件（引擎目录里的其它东西是调试工具，不分发）
ENGINE_FILES_COMMON = [
    "llama-server.exe", "llama-server-impl.dll", "llama-common.dll",
    "llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "mtmd.dll",
]
# CUDA 构建多一个后端 DLL。第一版漏了它，结果用户拿到的是
# 「退出码 0xC0000135（找不到 DLL）+ 日志完全空白」—— 所以现在打包完会真的试着启动一次。
ENGINE_FILES_CUDA = ["ggml-cuda.dll"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def build_exe(console: bool) -> Path:
    print("== 打包应用 ==")
    env = dict(os.environ)
    env["BONSAI_SRC"] = str(ROOT)
    env["BONSAI_CONSOLE"] = "1" if console else "0"
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
           "--distpath", str(DIST), "--workpath", str(ROOT / "build" / "work"),
           str(ROOT / "build" / "bonsai.spec")]
    r = subprocess.run(cmd, env=env, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit("PyInstaller 失败")
    exe = DIST / "BonsaiLocal.exe"
    if not exe.exists():
        raise SystemExit(f"没有生成 {exe}")
    print(f"   {exe}  {exe.stat().st_size / 2**20:.1f} MB")
    return exe


def verify_engine_zip(zip_path: Path) -> bool:
    """把包解到临时目录里，真的启动一次 llama-server。

    缺一个后端 DLL 的表现是「退出码 0xC0000135 + 日志空白」，
    从用户的反馈里几乎不可能定位。所以在构建阶段就把它挡住。
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="bonsai-verify-") as td:
        with zipfile.ZipFile(zip_path) as z:
            # 铺平解压：必须和应用运行时的做法一致（fetch.ensure_engine 也是铺平的），
            # 否则自检通过的布局和用户实际拿到的布局不是同一个东西。
            for member in z.infolist():
                name = Path(member.filename).name
                if not name:
                    continue
                with z.open(member) as src, (Path(td) / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
        exe = Path(td) / "llama-server.exe"
        if not exe.exists():
            print("   自检失败：包里没有 llama-server.exe")
            return False
        env = dict(os.environ)
        env["PATH"] = td + os.pathsep + env.get("PATH", "")
        try:
            r = subprocess.run([str(exe), "--help"], capture_output=True,
                               timeout=180, env=env)
        except Exception as e:                                  # noqa: BLE001
            print(f"   自检异常：{e}")
            return False
        if r.returncode == 0:
            return True
        code = r.returncode & 0xFFFFFFFF
        hint = "（找不到 DLL，多半漏了后端 DLL）" if code == 0xC0000135 else ""
        print(f"   自检失败：退出码 {r.returncode} / 0x{code:08X}{hint}")
        return False


def make_engine_zip(variant: str, src: Path, required: list[str],
                    extra_dlls: list[Path] | None, repo: str) -> dict:
    print(f"== 引擎包 {variant} ==")
    if not (src / "llama-server.exe").exists():
        print(f"   跳过：找不到 {src}\\llama-server.exe")
        return {}
    ENGINES_OUT.mkdir(parents=True, exist_ok=True)
    out = ENGINES_OUT / f"engine-{variant}.zip"

    missing = [f for f in required if not (src / f).exists()]
    if missing:
        print(f"   跳过：缺少 {', '.join(missing)}")
        return {}

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in required:
            z.write(src / name, name)
        for dll in extra_dlls or []:
            if dll.exists():
                # 平铺，不留 cuda/ 子目录：Windows 按可执行文件所在目录找 DLL，
                # 应用运行时也是铺平解压的，两边保持完全一致
                z.write(dll, dll.name)
            else:
                print(f"   警告：缺少 {dll}")

    if not verify_engine_zip(out):
        out.unlink(missing_ok=True)
        raise SystemExit(f"引擎包 {variant} 自检未通过，已放弃（不要发布这个包）")
    print("   自检通过：解压后能正常启动")

    info = {"url": f"https://github.com/{repo}/releases/latest/download/{out.name}",
            "bytes": out.stat().st_size, "sha256": sha256(out)}
    print(f"   {out.name}  {out.stat().st_size / 2**20:.1f} MB")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-engines", action="store_true", help="同时打引擎包")
    ap.add_argument("--console", action="store_true", help="保留控制台窗口（调试用）")
    ap.add_argument("--repo", default=os.environ.get("BONSAI_REPO", "OWNER/REPO"),
                    help="GitHub 仓库，用来生成引擎下载地址，例如 yourname/bonsai-local")
    args = ap.parse_args()

    DIST.mkdir(parents=True, exist_ok=True)

    # 引擎包必须先做：它们的地址和校验和要写进 engines.json，
    # 而 engines.json 会被打进 exe。顺序反了就会打出一个首次运行即崩的包。
    manifest: dict[str, dict] = {}
    if args.with_engines:
        if args.repo == "OWNER/REPO":
            print("警告：--repo 还是占位值，生成的下载地址不可用。")
        cuda = make_engine_zip("cuda-ada", CUDA_BIN,
                               ENGINE_FILES_COMMON + ENGINE_FILES_CUDA,
                               [CUDA_RUNTIME / n for n in CUDA_DLLS], args.repo)
        if cuda:
            manifest["cuda-ada"] = cuda
        cpu = make_engine_zip("cpu", CPU_BIN, ENGINE_FILES_COMMON, None, args.repo)
        if cpu:
            manifest["cpu"] = cpu
        if manifest:
            text = json.dumps(manifest, indent=2, ensure_ascii=False)
            (ROOT / "engines.json").write_text(text, encoding="utf-8")
            (DIST / "engines.json").write_text(text, encoding="utf-8")
            print(f"== 清单 ==\n   {ROOT / 'engines.json'}（会被打进 exe）")

    exe = build_exe(args.console)
    lines = [f"{sha256(exe)}  {exe.name}"]
    for name, info in manifest.items():
        lines.append(f"{info['sha256']}  engines/{Path(info['url']).name}")

    (DIST / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n== 完成 ==")
    for f in sorted(DIST.rglob("*")):
        if f.is_file():
            print(f"   {f.relative_to(DIST)}  {f.stat().st_size / 2**20:.1f} MB")
    if args.repo == "OWNER/REPO":
        print("\n提醒：engines.json 里的下载地址仍是占位值，"
              "请用 --repo 你的名字/仓库名 重新执行一次。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
