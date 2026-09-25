#!/usr/bin/env python3
"""用 Python 发 GitHub Release，而不是用 PowerShell。

存在的理由很具体：Windows PowerShell 5.1 的 `Invoke-RestMethod -Body <字符串>`
会按 ISO-8859-1 编码请求体，于是中文在 Release 页面上全部变成 `?` —— 而且是静默的，
API 返回 200。用 Python 发，编码是显式的，不会再有这个问题。

用法：
    set GH_TOKEN=ghp_xxx
    python build/gh_release.py --repo 名字/仓库 --tag v1.0.0 --notes CHANGELOG.md \
        --assets dist/BonsaiLocal.exe dist/engines.json dist/SHA256SUMS dist/engines/*.zip

已存在同 tag 的 Release 时会就地更新标题与正文，然后覆盖式上传资产。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"


def token() -> str:
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not tok:
        sys.exit("需要环境变量 GH_TOKEN")
    return tok


def call(method: str, url: str, tok: str, payload: dict | None = None,
         raw: bytes | None = None, ctype: str = "application/json") -> dict:
    """所有请求体都以 UTF-8 字节发送。这就是这个脚本存在的全部意义。"""
    if payload is not None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=raw, method=method)
    req.add_header("Authorization", f"Bearer {tok}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "bonsai-release")
    if raw is not None:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise SystemExit(f"{method} {url} -> HTTP {e.code}\n{detail}") from None


def section_for(text: str, tag: str) -> str:
    """从 CHANGELOG 里取出对应版本那一段，作为发布说明。"""
    marker = f"## {tag}"
    i = text.find(marker)
    if i < 0:
        return text
    rest = text[i:]
    # 去掉标题行本身
    nl = rest.find("\n")
    rest = rest[nl + 1:] if nl >= 0 else ""
    # 截到下一个版本标题为止
    j = rest.find("\n## ")
    return (rest[:j] if j >= 0 else rest).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="例如 yourname/bonsai-local")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--title", default=None, help="默认用 tag")
    ap.add_argument("--notes", default="CHANGELOG.md", help="changelog 文件")
    ap.add_argument("--assets", nargs="*", default=[])
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--prerelease", action="store_true")
    args = ap.parse_args()

    tok = token()
    title = args.title or args.tag
    notes = ""
    p = Path(args.notes)
    if p.exists():
        # 明确按 UTF-8 读，显式按 UTF-8 写出去
        notes = section_for(p.read_text(encoding="utf-8"), args.tag)
    if not notes:
        notes = f"{title}"

    existing = None
    try:
        existing = call("GET", f"{API}/repos/{args.repo}/releases/tags/{args.tag}", tok)
    except SystemExit:
        existing = None

    if existing:
        rel = call("PATCH", f"{API}/repos/{args.repo}/releases/{existing['id']}", tok,
                   {"name": title, "body": notes,
                    "draft": args.draft, "prerelease": args.prerelease})
        print(f"已更新 Release {args.tag}")
    else:
        rel = call("POST", f"{API}/repos/{args.repo}/releases", tok,
                   {"tag_name": args.tag, "name": title, "body": notes,
                    "draft": args.draft, "prerelease": args.prerelease})
        print(f"已创建 Release {args.tag}")

    print(f"  {rel['html_url']}")
    print(f"  标题: {rel['name']}")
    preview = rel["body"].replace("\n", " ")[:60]
    print(f"  正文: {preview}")

    have = {a["name"]: a for a in rel.get("assets", [])}
    for spec in args.assets:
        for path in sorted(Path().glob(spec)) if any(c in spec for c in "*?[") else [Path(spec)]:
            if not path.is_file():
                print(f"  跳过（不存在）: {path}")
                continue
            name = path.name
            if name in have:                       # 覆盖：先删旧的
                call("DELETE", f"{API}/repos/{args.repo}/releases/assets/{have[name]['id']}", tok)
            size = path.stat().st_size
            print(f"  上传 {name}  ({size / 2**20:.1f} MB)")
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            with path.open("rb") as fh:
                data = fh.read()
            call("POST", f"{UPLOADS}/repos/{args.repo}/releases/{rel['id']}/assets"
                         f"?name={urllib.parse.quote(name)}", tok, raw=data, ctype=ctype)

    final = call("GET", f"{API}/repos/{args.repo}/releases/{rel['id']}", tok)
    print("\n最终资产:")
    for a in final.get("assets", []):
        print(f"  {a['name']:<26} {a['size']:>12,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
