#!/usr/bin/env python3
"""扫一遍底座模型的 LoRA 生态，按用途分类、按下载量排序。

为什么按下载量排：名字和描述谁都会写，下载量是别人用脚投票的结果。
一个 0 下载的"神级 LoRA"基本等于不存在。

用法:
    python tools/survey_loras.py [--base Qwen/Qwen3.8-27B]
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

UA = {"User-Agent": "bonsai-research"}

# 分类靠名字和描述里的关键词。粗糙，但足够看出生态的形状。
BUCKETS: list[tuple[str, str]] = [
    ("去审查 / 破限", r"uncensor|abliter|heretic|crack|dealign|decensor|refusal|unfiltered|dolphin"),
    ("角色 / 人格",   r"persona|character|waifu|companion|roleplay|\brp\b|samantha|girl|bot-|chan\b"),
    ("语言 / 文风",   r"chinese|thai|japanese|korean|vietnam|spanish|style|taste|prose|writing|tone|文风|中文"),
    ("代码",          r"coder|\bcode\b|coding|programming|python|sql|dev"),
    ("推理 / 数学",   r"reason|math|think|r1|logic|cot|chain-of"),
    ("工具 / 智能体", r"tool|function|agent|json|rag|search|mcp|call"),
    ("领域专精",      r"medical|legal|law|financ|bio|chem|doctor|health"),
    ("长文 / 摘要",   r"summar|long|context|document|translat"),
    ("对齐 / 安全",   r"safety|guard|honest|sycophan|harmless|align|ethic"),
    ("蒸馏 / 效率",   r"distill|spec|draft|mtp|quant|efficien|tiny|mini"),
]


def bucket_of(text: str) -> str:
    low = text.lower()
    for name, pat in BUCKETS:
        if re.search(pat, low):
            return name
    return "其它"


def fetch(url: str) -> object:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    seen: dict[str, dict] = {}
    # 单一查询枚举不全：HF 的 search 只匹配仓库名，而很多 LoRA 名字里没有
    # "Qwen"。所以要同时用「base_model 标签」和「名字里带 LoRA」两类查询再合并。
    queries = [
        f"https://huggingface.co/api/models?filter=base_model:{urllib.parse.quote(args.base)}"
        f"&limit=100&full=true",
        "https://huggingface.co/api/models?filter=base_model:unsloth/Qwen3.8-27B"
        "&limit=100&full=true",
        "https://huggingface.co/api/models?search=Qwen3.8-27B-LoRA&limit=100&full=true",
        "https://huggingface.co/api/models?search=Qwen3.8-27b-LoRA&limit=100&full=true",
        "https://huggingface.co/api/models?search=Qwen3.8-27B&filter=lora&limit=100&full=true",
        "https://huggingface.co/api/models?search=Qwen3.8&filter=peft&limit=100&full=true",
    ]
    for q in queries:
        try:
            for m in fetch(q):                       # type: ignore[union-attr]
                seen[m["modelId"]] = m
        except urllib.error.HTTPError as e:
            print(f"  (查询失败 {e.code}: {q[:70]}…)")
        except Exception as e:                       # noqa: BLE001
            print(f"  (查询失败 {e}: {q[:70]}…)")

    print(f"扫到 {len(seen)} 个相关仓库\n")

    # 真正的适配器：标签里有 peft/lora，或名字里带
    adapters = []
    for mid, m in seen.items():
        tags = [str(t).lower() for t in (m.get("tags") or [])]
        low = mid.lower()
        if ("peft" in tags or "lora" in tags or "qlora" in tags
                or "lora" in low or "adapter" in low):
            adapters.append((mid, m, m.get("cardData") or {}))

    print(f"其中是 LoRA / 适配器的：{len(adapters)}\n")

    groups: dict[str, list] = defaultdict(list)
    for mid, m, card in adapters:
        name = (m.get("modelId") or "").split("/")[-1]
        desc = (card.get("model_name") or "") + " " + str(card.get("tags") or "")
        # 描述往往在 README 里，API 拿不到全文；就用名字 + tag 判断
        groups[bucket_of(name + " " + " ".join(m.get("tags") or []))].append(
            (m.get("downloads") or 0, mid, m.get("lastModified") or "", card.get("license")))

    for g in sorted(groups, key=lambda k: -sum(x[0] for x in groups[k])):
        items = sorted(groups[g], reverse=True)
        total = sum(x[0] for x in items)
        print(f"■ {g}   （{len(items)} 个，合计下载 {total:,}）")
        for dl, mid, when, lic in items[:6]:
            print(f"    {dl:>7,}  {mid:<58} {str(when)[:10]}  {lic or ''}")
        if len(items) > 6:
            print(f"    … 还有 {len(items) - 6} 个")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
