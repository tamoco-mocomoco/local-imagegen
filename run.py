#!/usr/bin/env python3
"""
OS 共通の起動スクリプト (Windows / macOS / Linux)。Python 標準ライブラリだけで動く。

  python3 run.py                 # Mac / Linux (./run.sh も同じ)
  python run.py                  # Windows
  python run.py --reload         # 引数はそのまま uvicorn に渡る

やること: uv の有無を確認 -> .env と PORT を読む -> doctor.py --brief の要約 -> uv run で uvicorn を起動。
環境変数の既定値 (HF_HOME, OUTPUT_DIR, HF_HUB_DISABLE_XET など) は server/app.py が自分で入れるので、
`uv run --extra rembg uvicorn server.app:app` を直接叩いても同じ動きになる。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UV_INSTALL = {
    "win32": 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
    "darwin": "curl -LsSf https://astral.sh/uv/install.sh | sh",
}


def read_env_file(path: Path) -> dict:
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def main(argv: list) -> int:
    uv = shutil.which("uv")
    if not uv:
        print("uv が見つかりません。インストール後にもう一度実行してください:", file=sys.stderr)
        print("  " + UV_INSTALL.get(sys.platform, UV_INSTALL["darwin"]), file=sys.stderr)
        print("  (https://docs.astral.sh/uv/)", file=sys.stderr)
        return 1

    env = read_env_file(ROOT / ".env")
    port = os.environ.get("PORT") or env.get("PORT") or "8000"
    preset = os.environ.get("PRESET") or env.get("PRESET") or "sdxl-turbo"
    print(f"preset={preset} port={port} project={ROOT}")

    # 環境診断の要約 (失敗しても起動は続ける)。詳細は python doctor.py
    try:
        subprocess.call([sys.executable, str(ROOT / "doctor.py"), "--brief"], cwd=str(ROOT), timeout=60)
    except Exception as e:  # noqa: BLE001
        print(f"doctor.py の実行に失敗しました (起動は続けます): {e}", file=sys.stderr)

    cmd = [uv, "run", "--project", str(ROOT), "--extra", "rembg",
           "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", str(port), *argv]
    try:
        return subprocess.call(cmd, cwd=str(ROOT))  # Ctrl+C は子にも届き、そのまま終了する
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
