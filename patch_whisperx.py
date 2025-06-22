#!/usr/bin/env python
"""
WhisperX 0.6.x 〜 0.6.3 用ワンタイムパッチ
tokens.clamp(min=0).long() へ置換
"""
import site, pathlib, re, shutil, datetime, sys

# --- 1) site-packages 内の whisperx/alignment.py を探す
pkg_dir = next(p for p in site.getsitepackages() if p.endswith("site-packages"))
target = pathlib.Path(pkg_dir, "whisperx", "alignment.py")
if not target.exists():
    sys.exit("❌  alignment.py が見つかりません。WhisperX がインストールされていますか？")

text = target.read_text()

# --- 2) 既に .long() が入っていればスキップ
if ".long()" in text:
    print("✅  既にパッチ済みです → 何もしません")
    sys.exit(0)

# --- 3) バックアップを保存
bak = target.with_suffix(f".bak_{datetime.datetime.now():%Y%m%d_%H%M%S}")
shutil.copy2(target, bak)
print(f"🗂  バックアップ作成 → {bak.name}")

# --- 4) 置換して上書き
patched = re.sub(r"tokens\.clamp\(min=0\)",
                 "tokens.clamp(min=0).long()", text)
target.write_text(patched)
print(f"✅  パッチ完了 → {target}")
