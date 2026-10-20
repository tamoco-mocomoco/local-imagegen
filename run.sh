#!/usr/bin/env bash
# Mac / Linux 用の薄い包み。実体は run.py (OS 共通、Python 標準ライブラリのみ)。
# Mac では Docker から GPU (Metal) を使えないので、このネイティブ起動を使う。
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 が見つかりません。Python 3.8 以上を入れてから実行してください (macOS: brew install python / Ubuntu: sudo apt install python3)" >&2
  exit 1
fi
exec python3 "$(dirname "$0")/run.py" "$@"
