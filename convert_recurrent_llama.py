#!/usr/bin/env python3
"""
poc-recurrent-llama/article_draft.md を Zenn 形式に変換する。
- frontmatter を Zenn スキーマで先頭追加
- 画像パス（プレーンファイル名）→ /images/recurrent-llama/ 下に書換え
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "poc-recurrent-llama" / "article_draft.md"
DST = ROOT / "articles" / "openmythos-slm-recurrent-llama-eval.md"

ZENN_TITLE = "OpenMythosはSLMの業務利用に道を開いたのか？"
ZENN_EMOJI = "🧠"
ZENN_TYPE = "tech"
ZENN_PUBLISHED = True  # 即公開（Qiita はすでに公開済み、Zenn 連携済み）
# Zenn の topics は小文字のみ。大文字混在トピックは不可
ZENN_TOPICS = ["llm", "slm", "machinelearning", "pytorch", "huggingface"]

# 画像ファイル名 → Zenn パス（/images/recurrent-llama/...）
PLOT_FILES = [
    "plot.png",
    "plot_marginal.png",
    "plot_three_studies.png",
    "plot_three_studies_normalized.png",
]


def main():
    body = SRC.read_text(encoding="utf-8")

    # 画像パス書換え：![alt](plot.png) → ![alt](/images/recurrent-llama/plot.png)
    for f in PLOT_FILES:
        body = body.replace(f"]({f})", f"](/images/recurrent-llama/{f})")

    topics_yaml = "[" + ", ".join([f'"{t}"' for t in ZENN_TOPICS]) + "]"
    zenn_fm = f"""---
title: "{ZENN_TITLE}"
emoji: "{ZENN_EMOJI}"
type: "{ZENN_TYPE}"
topics: {topics_yaml}
published: {str(ZENN_PUBLISHED).lower()}
---
"""
    out = zenn_fm + body
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(out, encoding="utf-8")
    print(f"✅ Wrote: {DST}")
    print(f"   Title: {ZENN_TITLE}")
    print(f"   Topics: {ZENN_TOPICS}")
    print(f"   Published: {ZENN_PUBLISHED}")
    print(f"   Body chars: {len(body):,}")
    print(f"   Image refs replaced: {sum(body.count(f'/images/recurrent-llama/{f}') for f in PLOT_FILES)}")


if __name__ == "__main__":
    main()
