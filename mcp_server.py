#!/usr/bin/env python3
"""
Claude Code / Claude Desktop から直接呼べる MCP サーバー (stdio)。
既存の HTTP API (server/app.py) を薄くラップするだけなので、先に ./run.sh (Windows は python run.py) か docker compose でサーバーを起動しておく。

  登録例 (Claude Code):
    claude mcp add imagegen -- /ABS/PATH/LocalLLMImages/.venv/bin/python /ABS/PATH/LocalLLMImages/mcp_server.py
  環境変数:
    IMAGEGEN_URL         生成サーバー (既定 http://localhost:8000)
    IMAGEGEN_OUTPUT_DIR  保存先 (既定 このファイルと同じ場所の outputs/)
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

try:  # mcp >= 2.0
    from mcp.server.mcpserver import Image, MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server, Image  # type: ignore
    from mcp.server.fastmcp.exceptions import ToolError  # type: ignore

IMAGEGEN_URL = os.environ.get("IMAGEGEN_URL", "http://localhost:8000").rstrip("/")
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("IMAGEGEN_OUTPUT_DIR", ROOT / "outputs"))
DOCTOR = ROOT / "doctor.py"
PREVIEW_MAX = int(os.environ.get("IMAGEGEN_PREVIEW_MAX", "512"))  # 返すプレビューの最大辺 (px)。0 で画像を返さない

STATIC_INSTRUCTIONS = (
    "ローカルの text-to-image サーバーで画像を生成する。generate_image の preset でモデルを切り替えられる "
    "(未ロードなら要求時に読み込んでキャッシュする)。目安: ラフ・構図確認は sdxl-turbo (数秒)、"
    "アニメ/ゲームキャラの仕上げは anime (Danbooru タグ形式、'solo' を入れる)、写実は flux-schnell。"
    "同じキャラの差分は reference_image_path に基準画像を渡す。透過 PNG が欲しければ remove_background=True。"
    "server_status で読み込み済みプリセットを確認でき、load_preset で事前に読み込める。"
    "サーバーが起動していない / 初期設定が済んでいないときは diagnose でこのマシンの環境診断 (推奨の起動方法・プリセット・.env) を得られる。"
)
TIER_JA = {"recommended": "推奨", "possible": "可 (ぎりぎり)", "unsupported": "不可"}


def _get(path: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(f"{IMAGEGEN_URL}{path}", timeout=timeout) as r:
        return json.load(r)


def _post(path: str, body: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(f"{IMAGEGEN_URL}{path}", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except Exception:
            pass
        raise ToolError(f"サーバーエラー (HTTP {e.code}): {detail}")


def _build_instructions() -> str:
    """起動時にサーバーへ問い合わせ、このマシンで選べるプリセットと使い分けを Claude に伝える。"""
    try:
        h = _get("/health", timeout=5)
        presets = _get("/presets", timeout=5)
    except Exception:
        return STATIC_INSTRUCTIONS
    lines = [
        f"ローカルの text-to-image サーバー (device={h.get('device')}, メモリ {h.get('memory_gb')}GB)。",
        f"generate_image の preset でモデルを切り替えられる。省略時は '{h.get('preset')}'。未ロードのプリセットは要求時に読み込んでキャッシュする "
        f"(初回は 1〜2 分、モデル未取得ならダウンロードでさらに数分)。同時に保持できるのは {h.get('max_loaded_models')} モデルで、超えると最も古いものを解放する。",
        "このマシンで選べるプリセット:",
    ]
    for name in h.get("allowed_presets", []):
        p = presets.get(name, {})
        state = "読み込み済み" if p.get("loaded") else "未読み込み"
        tier = TIER_JA.get(p.get("tier") or "", "")
        tag = f"{tier}, {state}" if tier else state
        lines.append(f"- {name} ({tag}): {p.get('description', '')}"
                     + (f" [{p['reason']}]" if p.get("tier") == "possible" and p.get("reason") else ""))
    env = h.get("environment") or {}
    if env.get("unsupported"):
        lines.append(f"このマシンのメモリでは使えないプリセット (選ばないこと): {', '.join(env['unsupported'])}")
    if env.get("default_preset_switched_from"):
        lines.append(f"既定の {env['default_preset_switched_from']} はこの環境では使えないため {h.get('preset')} に切り替えている。")
    lines += [
        "使い分けの目安: ラフ・構図の試行錯誤は sdxl-turbo (数秒)、アニメ/ゲームキャラの仕上げは anime "
        "(Danbooru タグ形式で 'solo' を入れる、ラフは steps=8)、写実・コンセプトアートは flux-schnell (あれば)。",
        "同じキャラの差分は reference_image_path に基準画像を渡す (SD/SDXL 系のみ)。透過 PNG は remove_background=True。",
        "一連の作業で使うプリセットが決まっているなら load_preset で先に読み込むと待ち時間を前倒しできる。",
        "環境の診断や推奨設定 (.env) が必要なら diagnose を呼ぶ。",
    ]
    return "\n".join(lines)


server = _Server("imagegen", instructions=_build_instructions())


def _wait_ready(timeout: float = 900) -> dict:
    deadline = time.time() + timeout
    while True:
        try:
            h = _get("/health")
        except (urllib.error.URLError, ConnectionError) as e:
            raise ToolError(f"生成サーバー {IMAGEGEN_URL} に接続できません ({e})。./run.sh (Windows は python run.py) か docker compose up で起動してください")
        if h.get("ok"):
            return h
        if h.get("error"):
            raise ToolError(f"生成サーバーのモデル読み込みに失敗しています: {h['error']}")
        if time.time() > deadline:
            raise ToolError("モデルの読み込み待ちがタイムアウトしました")
        time.sleep(3)


def _preview(png_bytes: bytes, transparent: bool) -> Optional[Image]:
    if PREVIEW_MAX <= 0:
        return None
    try:
        from PIL import Image as PILImage
    except ImportError:
        return Image(data=png_bytes, format="png")
    im = PILImage.open(io.BytesIO(png_bytes))
    if transparent or im.mode == "RGBA":
        bg = PILImage.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im.convert("RGBA"))
        im = bg
    im = im.convert("RGB")
    im.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return Image(data=buf.getvalue(), format="jpeg")


@server.tool()
def server_status() -> dict:
    """画像生成サーバーの状態 (モデル・デバイス・既定値・準備完了か) を返す。"""
    try:
        return _get("/health")
    except (urllib.error.URLError, ConnectionError) as e:
        return {"ok": False, "error": f"{IMAGEGEN_URL} に接続できません: {e}"}


@server.tool()
def list_presets() -> dict:
    """モデルプリセット一覧。各項目の description (用途)、tier (recommended / possible / unsupported とその reason)、allowed (このマシンで選べるか)、loaded (読み込み済みか) を返す。"""
    return _get("/presets")


@server.tool()
def diagnose(write_env: bool = False) -> dict:
    """このマシンの環境診断 (doctor.py)。生成サーバーが起動していなくても動く。

    OS / CPU / メモリ / GPU、docker・uv などのツール、ディスク空き、Hugging Face 到達性、キャッシュ済みモデル、
    サーバーの起動状態を調べ、推奨の起動方法 (run.sh / run.py / docker compose)、PRESET、MAX_LOADED_MODELS、
    ALLOWED_PRESETS、各プリセットの tier (recommended / possible / unsupported) と推奨 .env を返す。
    - write_env: True なら推奨 .env を書き込む (既存は .env.bak に退避)。既定は読むだけで何も変更しない
    戻り値の report が人向けの 1 画面レポート、recommendation.env が推奨 .env の本文。
    """
    if not DOCTOR.is_file():
        raise ToolError(f"doctor.py が見つかりません: {DOCTOR}")
    cmd = [sys.executable, str(DOCTOR), "--json", "--no-speed"]
    if write_env:
        cmd.append("--write-env")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, timeout=180, env=env)
    except subprocess.TimeoutExpired:
        raise ToolError("doctor.py がタイムアウトしました (ネットワーク確認で止まっている可能性)")
    out = p.stdout.decode("utf-8", errors="replace")
    if p.returncode != 0 or not out.strip():
        raise ToolError(f"doctor.py が失敗しました (exit {p.returncode}): {p.stderr.decode('utf-8', errors='replace')[-800:]}")
    try:
        d = json.loads(out)
    except json.JSONDecodeError as e:
        raise ToolError(f"doctor.py の出力を JSON として読めません: {e}: {out[:300]}")
    d.pop("brief", None)
    return d


@server.tool()
def load_preset(preset: str) -> dict:
    """プリセットを今すぐ読み込んでキャッシュする (生成はしない)。これから使うモデルの待ち時間を前倒しするために使う。"""
    _wait_ready()
    return _post("/load", {"preset": preset})


@server.tool()
def unload_preset(preset: str) -> dict:
    """プリセットをメモリから解放する。メモリを空けたいときに使う。次に使うときは再読み込みになる。"""
    return _post("/unload", {"preset": preset})


@server.tool()
def generate_image(
    prompt: str,
    preset: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: Optional[int] = None,
    n: int = 1,
    remove_background: bool = False,
    reference_image_path: Optional[str] = None,
    reference_scale: float = 0.6,
    raw_prompt: bool = False,
    out_path: Optional[str] = None,
) -> list:
    """テキストから画像を生成して PNG を保存し、保存パスとプレビューを返す。

    - prompt: anime プリセットではカンマ区切りの Danbooru タグ。1 人なら 'solo' を入れる
    - preset: モデル (sdxl-turbo=高速ラフ / anime=アニメ仕上げ / sdxl / flux-schnell ...)。省略時はサーバー既定。未ロードなら読み込んでキャッシュする
    - width/height/steps/guidance: 省略時はプリセット既定 (anime: 832x1216, 28 steps)
    - seed: 固定すると再現できる。省略でランダム
    - remove_background: True で背景を除去した透過 PNG にする
    - reference_image_path: 基準画像のパス。同じキャラの差分を作るときに渡す (IP-Adapter)
    - reference_scale: 参照の強さ 0.4 控えめ / 0.6 標準 / 0.8 強め
    - out_path: 保存先ファイル名。省略時は outputs/ にサーバー命名で保存。n>1 なら -1,-2.. を付与
    """
    body: dict = {"prompt": prompt, "n": max(1, min(int(n), 8)), "return_base64": True,
                  "remove_background": bool(remove_background), "raw_prompt": bool(raw_prompt)}
    if preset:
        body["preset"] = preset
    if negative_prompt is not None:
        body["negative_prompt"] = negative_prompt
    for k, v in (("width", width), ("height", height), ("steps", steps), ("guidance", guidance), ("seed", seed)):
        if v is not None:
            body[k] = v
    if reference_image_path:
        p = Path(reference_image_path).expanduser()
        if not p.is_file():
            raise ToolError(f"reference_image_path が見つかりません: {p}")
        body["reference_image"] = base64.b64encode(p.read_bytes()).decode()
        body["reference_scale"] = float(reference_scale)

    _wait_ready()
    res = _post("/generate", body)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out: list = []
    lines = []
    for i, im in enumerate(res["images"]):
        data = base64.b64decode(im["base64"])
        if out_path:
            root, ext = os.path.splitext(out_path)
            ext = ext or ".png"
            path = Path(out_path if len(res["images"]) == 1 else f"{root}-{i + 1}{ext}").expanduser()
            if not path.is_absolute():
                path = OUTPUT_DIR / path
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path = OUTPUT_DIR / im["file"]
        path.write_bytes(data)
        lines.append(
            f"saved: {path}\n  preset={im.get('preset', '-')} seed={im['seed']} size={im['width']}x{im['height']} steps={im['steps']} "
            f"guidance={im['guidance']} time={im['seconds']}s"
            + (" transparent" if im.get("transparent") else "")
            + (f" ref_scale={im['reference_scale']}" if im.get("reference_scale") is not None else "")
            + f"\n  model={im['model']}"
        )
        prev = _preview(data, bool(im.get("transparent")))
        if prev is not None:
            out.append(prev)
    out.insert(0, "\n".join(lines))
    return out


if __name__ == "__main__":
    server.run("stdio")
