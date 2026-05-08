#!/usr/bin/env python3
"""
Qiita 形式の記事を Zenn 形式に変換する。
- frontmatter を Zenn スキーマに変換
- 画像パスを `images/...` → `/images/...`（先頭スラッシュ）
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "qiita-agentic-quality-gate" / "public" / "agentic-quality-gate-karpathy.md"
DST = ROOT / "articles" / "agentic-quality-gate-karpathy.md"

# Zenn 設定（このファイル内で完結させる）
ZENN_EMOJI = "🛡️"        # 記事サムネ用 emoji（1 文字）
ZENN_TYPE = "tech"        # tech: 技術記事 / idea: アイデア
ZENN_PUBLISHED = False    # まずは下書き（GitHub push 後に true に切替）
# Zenn の topics は小文字のみ（大文字は不可）
ZENN_TOPICS = ["claudecode", "llm", "devops", "品質保証", "agenticengineering"]


def main():
    text = SRC.read_text(encoding="utf-8")

    # ---- frontmatter 抽出 ----
    if not text.startswith("---\n"):
        raise SystemExit("frontmatter not found")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise SystemExit("frontmatter end not found")
    fm_block = text[4:end]
    body = text[end + 5 :]

    # title だけ拾う
    m = re.search(r"^title:\s*(.+?)\s*$", fm_block, flags=re.M)
    if not m:
        raise SystemExit("title not found")
    title = m.group(1).strip().strip('"').strip("'")

    # ---- 画像パスを Zenn 形式に変換 ----
    # ![alt](images/...) → ![alt](/images/...)
    body = re.sub(r"!\[([^\]]*)\]\(images/", r"![\1](/images/", body)

    # ---- Zenn frontmatter を組み立て ----
    topics_yaml = "[" + ", ".join([f'"{t}"' for t in ZENN_TOPICS]) + "]"
    zenn_fm = f"""---
title: "{title}"
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
    print(f"   Title: {title}")
    print(f"   Topics: {ZENN_TOPICS}")
    print(f"   Published: {ZENN_PUBLISHED}")
    print(f"   Body chars: {len(body):,}")


if __name__ == "__main__":
    main()
