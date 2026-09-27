#!/usr/bin/env python3
"""按 rubrics-quality.json 给模型回答打分（确定性，可复现）。

评分规则：
  · 每个 must_mention 是一个「考点」，只要它的任一 regex 命中就算拿到分
  · 总分 = 所有考点的命中数（当前 8 题共 35 个考点）
  · suspect 是「答错的特征」，命中说明踩了坑，单独列出但不扣分
    （免得一个错误表述因为表述方式不同被重复扣分）

题目顺序与 dev-prompts/quality-probe.txt 按 `---` 切分的顺序一一对应
（评分表里只有 id/title，没有题干，靠序号对齐）。

用法:
    python build/score_quality.py --answers devdata/compare-models.json
    python build/score_quality.py --answers a.json --answers b.json --detail
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"


def load_rubric(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["prompts"]


def load_answers(paths: list[Path]) -> dict[str, dict[int, str]]:
    """-> {arm: {prompt_index: answer}}"""
    arms: dict[str, dict[int, str]] = {}
    for p in paths:
        if not p.exists():
            print(f"  ! 跳过不存在的 {p}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for a in data.get("answers", []):
            arm = a.get("arm") or data.get("label") or p.stem
            arms.setdefault(arm, {})[int(a["prompt_index"])] = a.get("answer") or ""
    return arms


def hit(text: str, patterns: list[str]) -> str | None:
    """返回命中的那个 pattern，没命中返回 None。"""
    for pat in patterns:
        try:
            if re.search(pat, text, re.IGNORECASE):
                return pat
        except re.error:
            if pat.lower() in text.lower():
                return pat
    return None


def score_arm(rubric: list[dict], answers: dict[int, str]) -> dict:
    total = 0
    possible = 0
    per_prompt = []
    suspects = []
    for i, r in enumerate(rubric):
        text = answers.get(i, "")
        got, want = 0, 0
        missing = []
        for m in r.get("must_mention", []):
            want += 1
            if text and hit(text, m["patterns"]):
                got += 1
            else:
                missing.append(m["name"])
        for s in r.get("suspect", []):
            if text and hit(text, s["patterns"]):
                suspects.append(f"{r['id']}: {s['name']}")
        total += got
        possible += want
        per_prompt.append({
            "id": r["id"], "title": r.get("title", ""), "got": got,
            "want": want, "missing": missing,
            "empty": not text.strip(),
        })
    return {"score": total, "possible": possible,
            "pct": (100.0 * total / possible) if possible else 0.0,
            "per_prompt": per_prompt, "suspects": suspects}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric", default=str(BUILD / "rubrics-quality.json"))
    ap.add_argument("--answers", action="append", required=True)
    ap.add_argument("--detail", action="store_true", help="逐题列出漏掉的考点")
    args = ap.parse_args()

    rubric = load_rubric(Path(args.rubric))
    arms = load_answers([Path(a) for a in args.answers])
    if not arms:
        print("没有可评分的回答")
        return 1

    results = {arm: score_arm(rubric, ans) for arm, ans in arms.items()}

    print("=" * 78)
    print(f"{'模型':<24}{'得分':>10}{'命中率':>10}{'空答':>8}")
    print("-" * 78)
    for arm, r in sorted(results.items(), key=lambda kv: -kv[1]["pct"]):
        empty = sum(1 for p in r["per_prompt"] if p["empty"])
        print(f"{arm:<24}{r['score']:>4}/{r['possible']:<5}{r['pct']:>9.1f}%{empty:>8}")

    # 逐考点对比（列 >=2 个模型时最有价值）
    if len(results) > 1:
        print("\n" + "=" * 78)
        print("逐题对比（每题命中的考点数）")
        print("-" * 78)
        names = list(results.keys())
        print(f"{'题目':<14}" + "".join(f"{n[:18]:>20}" for n in names))
        for i, r in enumerate(rubric):
            row = f"{r['id']:<14}"
            for n in names:
                pp = results[n]["per_prompt"][i]
                row += f"{pp['got']}/{pp['want']:>18}"
            print(row)

    if args.detail:
        for arm, r in results.items():
            print("\n" + "=" * 78)
            print(f"── {arm}  漏掉的考点 ──")
            for pp in r["per_prompt"]:
                mark = "空答!" if pp["empty"] else f"{pp['got']}/{pp['want']}"
                print(f"  [{mark:>5}] {pp['id']:<14} {pp['title']}")
                for m in pp["missing"]:
                    print(f"          ✗ {m}")
            if r["suspects"]:
                print("  踩坑：")
                for s in r["suspects"]:
                    print(f"          ! {s}")

    # 写一份机器可读结果，方便和速度表合并
    out = BUILD / "work" / "quality-scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
