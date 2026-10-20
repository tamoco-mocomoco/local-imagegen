#!/usr/bin/env python3
"""
依存ゼロの CLI クライアント。サーバー (Docker でもネイティブでも) に投げて PNG を保存する。

  python3 gen.py "a cat in a spacesuit" -o cat.png
  python3 gen.py "..." --n 4 --seed 42 --size 768x768 --steps 4
  IMAGEGEN_URL=http://gpu-box:8000 python3 gen.py "..."

出力: 保存したファイルパスを 1 行ずつ stdout に出す (Claude Code 等から扱いやすい)。
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser(description="Generate images via local imagegen server")
    p.add_argument("prompt", nargs="?", help="プロンプト (--load/--unload/--status のときは省略可)")
    p.add_argument("--preset", help="モデルプリセット (sdxl-turbo / anime / ...)。省略時はサーバー既定。未ロードなら要求時に読み込む")
    p.add_argument("--load", metavar="PRESET", help="生成せずにプリセットを読み込んでキャッシュする")
    p.add_argument("--unload", metavar="PRESET", help="プリセットをメモリから解放する")
    p.add_argument("--status", action="store_true", help="サーバー状態 (読み込み済みプリセットなど) を表示")
    p.add_argument("-o", "--out", help="出力ファイル名 (n>1 の時は -1,-2.. を付与)。省略時は outputs/ にサーバー命名で保存")
    p.add_argument("--negative", help='negative prompt。省略時はプリセット既定、"" で無効化')
    p.add_argument("--raw", action="store_true", help="プリセットの品質タグ (prompt_suffix) を付けない")
    p.add_argument("--rembg", action="store_true", help="背景を除去して透過 PNG にする")
    p.add_argument("--ref", help="参照画像 (PNG/JPEG)。IP-Adapter でキャラの見た目を引き継ぐ")
    p.add_argument("--ref-scale", type=float, default=0.6, help="参照の強さ 0〜1.5 (既定 0.6)")
    p.add_argument("--size", help="WxH 例: 1024x1024")
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--url", default=os.environ.get("IMAGEGEN_URL", "http://localhost:8000"))
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--json", action="store_true", help="サーバーの JSON 応答をそのまま出力")
    a = p.parse_args()

    if a.status or a.load or a.unload:
        try:
            if a.status:
                with urllib.request.urlopen(f"{a.url}/health", timeout=10) as r:
                    print(json.dumps(json.load(r), ensure_ascii=False, indent=2))
            for path, preset in (("/load", a.load), ("/unload", a.unload)):
                if preset:
                    req = urllib.request.Request(f"{a.url}{path}", data=json.dumps({"preset": preset}).encode(),
                                                 headers={"content-type": "application/json"})
                    with urllib.request.urlopen(req, timeout=a.timeout) as r:
                        print(json.dumps(json.load(r), ensure_ascii=False))
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read().decode(errors='replace')}", file=sys.stderr)
            return 1
        except (urllib.error.URLError, ConnectionError) as e:
            print(f"cannot reach {a.url} ({e}); is the server running?", file=sys.stderr)
            return 2
        return 0
    if not a.prompt:
        p.error("prompt が必要です")

    body = {"prompt": a.prompt, "n": a.n, "return_base64": True}
    if a.preset:
        body["preset"] = a.preset
    if a.negative is not None:
        body["negative_prompt"] = a.negative
    if a.raw:
        body["raw_prompt"] = True
    if a.rembg:
        body["remove_background"] = True
    if a.ref:
        with open(a.ref, "rb") as f:
            body["reference_image"] = base64.b64encode(f.read()).decode()
        body["reference_scale"] = a.ref_scale
    if a.size:
        w, h = a.size.lower().split("x")
        body["width"], body["height"] = int(w), int(h)
    for k in ("steps", "guidance", "seed"):
        if getattr(a, k) is not None:
            body[k] = getattr(a, k)

    # wait for model to be ready
    deadline = time.time() + a.timeout
    while True:
        try:
            with urllib.request.urlopen(f"{a.url}/health", timeout=10) as r:
                h = json.load(r)
            if h.get("ok"):
                break
            if h.get("error"):
                print(f"server error: {h['error']}", file=sys.stderr)
                return 2
            print("waiting for model to load...", file=sys.stderr)
        except (urllib.error.URLError, ConnectionError) as e:
            print(f"cannot reach {a.url} ({e}); is the server running?", file=sys.stderr)
            return 2
        if time.time() > deadline:
            print("timeout waiting for model", file=sys.stderr)
            return 2
        time.sleep(3)

    req = urllib.request.Request(
        f"{a.url}/generate", data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            res = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode(errors='replace')}", file=sys.stderr)
        return 1

    if a.json:
        for im in res["images"]:
            im.pop("base64", None)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    for i, im in enumerate(res["images"]):
        if a.out:
            root, ext = os.path.splitext(a.out)
            ext = ext or ".png"
            path = a.out if len(res["images"]) == 1 else f"{root}-{i + 1}{ext}"
        else:
            os.makedirs("outputs", exist_ok=True)
            path = os.path.join("outputs", im["file"])
        with open(path, "wb") as f:
            f.write(base64.b64decode(im["base64"]))
        print(path)
        extra = (" transparent" if im.get("transparent") else "") + (f" ref={im['reference_scale']}" if im.get("reference_scale") is not None else "")
        print(f"  [{im.get('preset', '-')}] seed={im['seed']} {im['width']}x{im['height']} steps={im['steps']} {im['seconds']}s{extra}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
