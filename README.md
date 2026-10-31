# Local ImageGen

ローカルで画像生成 (text-to-image) するための小さな環境。
[diffusers](https://github.com/huggingface/diffusers) ベースの HTTP API + Web UI + 依存ゼロの CLI で構成され、
**同じコードが Docker (Linux GPU / CPU) でも Mac ネイティブ (Apple GPU) でも動く**。

```
compose.yml / compose.gpu.yml   Docker 起動 (Linux + NVIDIA, または CPU)
run.py                          OS 共通のネイティブ起動 (uv を呼ぶ。Python 標準ライブラリのみ)。Windows は python run.py
run.sh                          Mac / Linux 用の薄い包み (python3 run.py と同じ)。Mac で GPU を使うにはこちら
doctor.py                       環境診断 (標準ライブラリのみ)。推奨の起動方法・プリセット・.env を出す
server/app.py                   FastAPI サーバー本体
server/presets.py               プリセット定義と環境要件 (min_memory_gb / cpu_ok)
gen.py                          CLI クライアント (Python 標準ライブラリのみ)
mcp_server.py                   Claude Code / Claude Desktop 用 MCP サーバー (HTTP API のラッパー)
models/                         Hugging Face モデルキャッシュ (Docker とネイティブで共有)
outputs/                        生成画像
```

## なぜ 2 通りあるのか

| 環境 | 推奨起動 | 理由 |
|---|---|---|
| Linux / WSL2 + NVIDIA GPU | `docker compose -f compose.yml -f compose.gpu.yml up -d --build` | CUDA で高速。要 NVIDIA Container Toolkit |
| Windows + NVIDIA GPU | `python run.py` (ネイティブ) または Docker Desktop で上の compose | ネイティブは uv だけで動く。Docker Desktop (WSL2) は GPU パススルー同梱 |
| Mac (Apple Silicon) | `./run.sh` (= `python3 run.py`) | **macOS の Docker は GPU (Metal) を使えない**。ネイティブなら MPS で加速 |
| GPU なし (どこでも) | `docker compose up -d --build` | CPU 生成。軽いモデル (sdxl-turbo / sd15) なら実用範囲 |

起動後は `http://localhost:8000` に Web UI、`http://localhost:8000/docs` に API ドキュメント。

## 動作実績

| 環境 | モデル | 解像度 | 1 枚あたり |
|---|---|---|---|
| M2 Max 64GB, `./run.sh` (MPS) | sdxl-turbo | 512x512 / 512x768 | 約 3 秒 (初回のみ約 7 秒) |
| 同上, `docker compose up` (CPU 6 コア) | sdxl-turbo | 512x512 | 約 67 秒 |
| M2 Max 64GB, `./run.sh` (MPS) | anime (Animagine XL 4.0) | 832x1216 / 1024x1024, 28 steps | 約 70 秒 (`--steps 20` で約 50 秒) |
| 同上 | anime + `--ref` (IP-Adapter) | 1024x1024, 28 steps | 約 70 秒 (+ 初回アダプタ読み込み 5 秒) |
| 同上 | anime + `--rembg` | 1024x1024 | 生成時間 + 約 2〜10 秒 |

## まず環境診断 (全 OS 共通)

最初に `doctor.py` を実行する。追加パッケージ不要 (Python 3.8+ の標準ライブラリだけ) で、
OS / CPU / メモリ / GPU、docker・uv などのツール、ディスク空き、Hugging Face への到達性と DL 速度、
キャッシュ済みモデル、サーバーの起動状態を調べ、**このマシンでの推奨 (起動方法・PRESET・.env)** を出す。

```bash
python3 doctor.py              # Mac / Linux
```

```powershell
python doctor.py               # Windows (py -3 doctor.py でも可)
```

出力例 (M2 Max 64GB):

```
== Local ImageGen 環境診断 2026-10-20 11:08 ==
システム   macOS 25.3.0 (arm64)  CPU: Apple M2 Max (12 cores)  RAM: 64GB  Python 3.9.6
GPU        Apple Silicon (統合メモリ 64GB) -> device=mps
ツール     uv [OK] 0.8.14  python [OK] 3.9.6  docker [OK] 29.1.3  compose v2 [OK] v2.40.3-desktop.1  NVIDIA toolkit [n/a]
           - python: サーバーは 3.11+ が必要だが uv が自動で取得するので問題なし (doctor / gen.py はこのままで可)
ディスク   /Users/you/LocalLLMImages/models の空き 328GB / 926GB [OK]
ネット     https://huggingface.co [OK] 0.27s  DL 速度 約 8.8MB/s (8.3MB で計測)
キャッシュ sdxl-turbo 7.1GB, anime 6.5GB, ip-adapter 3.1GB  (計 16.7GB, .../models/hub)
サーバー   http://localhost:8000/health 起動中 (device=mps, preset=anime, loaded: anime, sdxl-turbo)

プリセット (device=mps, メモリ 64GB)
  [推奨] sdxl-turbo    統合メモリ 64GB >= 目安 8GB  [cached 7.1GB]
  [推奨] sd15          統合メモリ 64GB >= 目安 6GB
  [推奨] sdxl          統合メモリ 64GB >= 目安 12GB
  [推奨] anime         統合メモリ 64GB >= 目安 12GB  [cached 6.5GB]
  [推奨] flux-schnell  統合メモリ 64GB >= 目安 24GB
  [推奨] zimage-turbo  統合メモリ 64GB >= 目安 20GB

推奨
  起動:   ./run.sh    (macOS の Docker は GPU (Metal) を使えないのでネイティブ起動)
  設定:   PRESET=anime  MAX_LOADED_MODELS=2  ALLOWED_PRESETS=sdxl-turbo,sd15,sdxl,anime,flux-schnell,zimage-turbo
  初回 DL: 不要 (anime はキャッシュ済み) (別途: rembg 0.2GB (--rembg 初回のみ))

推奨 .env (python3 doctor.py --write-env で書き込み。既存は .env.bak に退避):
  PRESET=anime
  MAX_LOADED_MODELS=2
  ALLOWED_PRESETS=sdxl-turbo,sd15,sdxl,anime,flux-schnell,zimage-turbo
  PORT=8000
  #HF_TOKEN=hf_xxx  # gated モデル (FLUX.1-dev など) を使うときだけ
```

| オプション | 動作 |
|---|---|
| `--write-env` | 推奨 .env を書き込む (既存は `.env.bak` に退避し、`HF_TOKEN` などの既存キーは引き継ぐ)。**これを付けない限り何も変更しない** |
| `--brief` | 数行の要約。`run.py` (`run.sh`) / Docker が起動時に自動で表示する |
| `--json` | 機械可読 (MCP の `diagnose` ツールが使う) |
| `--no-net` / `--no-speed` | Hugging Face への接続確認 / 速度計測を省く |

不足しているツール (uv, Docker, NVIDIA Container Toolkit など) があれば OS 別の導入コマンドを「要対応」として出す。
GPU が無い、またはメモリが足りないときは、各プリセットを **推奨 / 可 / 不可** に分けて理由を付ける (サーバーも同じ判定を使う。下の「環境別の推奨」参照)。

### 環境別の推奨

`server/presets.py` の `min_memory_gb` (fp16 で快適に動く VRAM / 統合メモリの目安) と `cpu_ok` から、
device とメモリで各プリセットを 3 段階に分類する。メモリが目安の 70% 以上なら「可」(解像度を下げる / `CPU_OFFLOAD=1`)。

| 環境 | 起動 | 推奨 PRESET | 推奨 | 可 (ぎりぎり) | 不可 |
|---|---|---|---|---|---|
| CPU のみ (RAM 16GB) | `./run.sh` / `docker compose up` | `sdxl-turbo` | sdxl-turbo, sd15 | - | sdxl, anime, flux-schnell, zimage-turbo |
| CPU のみ (RAM 32GB+) | 同上 | `sdxl-turbo` | sdxl-turbo, sd15 | anime, sdxl (動くが非常に遅い) | flux-schnell, zimage-turbo (RAM が目安の 2 倍あれば「可」) |
| NVIDIA 8GB (Windows `python run.py` / Linux) | ネイティブ or `compose.gpu` | `sdxl-turbo` | sdxl-turbo, sd15 | - | sdxl, anime, flux-schnell, zimage-turbo |
| NVIDIA 12〜16GB | 同上 | `anime` | sdxl-turbo, sd15, sdxl, anime | zimage-turbo (16GB) | flux-schnell |
| NVIDIA 24GB+ | 同上 | `anime` | 全部 | - | - |
| Apple Silicon 16GB | `./run.sh` | `anime` | sdxl-turbo, sd15, sdxl, anime | zimage-turbo | flux-schnell |
| Apple Silicon 32GB | `./run.sh` | `anime` | 全部 | - | - |
| Apple Silicon 64GB | `./run.sh` (常駐 2 モデル) | `anime` | 全部 | - | - |

- `MAX_LOADED_MODELS` の推奨はメモリ 32GB 以上で 2、未満で 1 (サーバーの既定と同じ)
- サーバー側も起動時に同じ分類を行い、`ALLOWED_PRESETS` 未指定なら **不可以外** を許可し、既定の `PRESET` が不可ならログで警告して最良の推奨に切り替える (`.env` で `PRESET` を明示した場合は警告だけ出してそのまま使う)。分類は `GET /presets` の `tier` / `reason` と `GET /health` の `environment` で確認できる
- 検証用に `FAKE_DEVICE=cpu|cuda|mps` と `FAKE_MEMORY_GB=8` で環境を偽装できる (doctor.py はさらに `FAKE_OS=Windows` `FAKE_GPU=...` `FAKE_WSL=1`)

## クイックスタート (Mac / Linux)

```bash
# uv が無ければ: curl -LsSf https://astral.sh/uv/install.sh | sh
python3 doctor.py --write-env   # 推奨の .env を書く (省略すると既定の sdxl-turbo)
./run.sh                        # = python3 run.py。初回は依存の取得 + モデル DL (sdxl-turbo / anime で約 7GB)
```

起動スクリプトは OS 共通の [run.py](run.py) (Python 標準ライブラリのみ) で、uv の有無を確認し、`.env` と `PORT` を読み、
`doctor.py --brief` の要約 (device、推奨プリセット、注意点) を出してから `uv run --extra rembg uvicorn server.app:app` を起動する。
引数はそのまま uvicorn に渡る (`./run.sh --reload` など)。`run.sh` は `python3 run.py` を呼ぶだけの包み。
環境変数の既定値 (`HF_HOME=./models`, `OUTPUT_DIR=./outputs`, `HF_HUB_DISABLE_XET=1` など) と `.env` の読み込みは
`server/app.py` 自身が行うので、`uv run --extra rembg uvicorn server.app:app --host 0.0.0.0 --port 8000` を直接叩いても同じ動きになる。

別ターミナルで:

```bash
python3 gen.py "a cute shiba inu wearing a kimono, ukiyo-e style" -o shiba.png
```

## クイックスタート (Windows)

```powershell
# 診断 (GPU / VRAM に合った PRESET と .env を提案。--write-env で書き込み)
python doctor.py --write-env
# uv が無ければ
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# 起動 (初回は依存の取得 + モデル DL)。Mac / Linux と同じ run.py を使う
python run.py
```

Python が無い場合は `winget install Python.Python.3.12` か [python.org](https://www.python.org/downloads/windows/) から入れる
(サーバー本体の Python 3.12 は uv が自動で取得するので、`doctor.py` / `gen.py` 用に 3.8+ があれば十分)。
VRAM 8GB の GPU では `anime` が「不可」になるので `sdxl-turbo` が推奨される。

別ウィンドウで `python gen.py "..." -o out.png`。Claude Code への MCP 登録は Python のパスが変わる:

```powershell
claude mcp add imagegen -- "$PWD\.venv\Scripts\python.exe" "$PWD\mcp_server.py"
```

Docker Desktop を使う場合は WSL2 バックエンドにして、下の Docker の手順をそのまま使う (NVIDIA GPU は Windows 用ドライバだけで `compose.gpu.yml` が効く)。

## クイックスタート (Docker)

ビルド済みイメージを [GitHub Container Registry](https://github.com/tamoco-mocomoco/local-imagegen/pkgs/container/local-imagegen) で公開している。
`linux/amd64` は CUDA 同梱の torch (NVIDIA GPU / CPU 両用)、`linux/arm64` は CPU 版。`docker compose` が自動で合うほうを取得する。

```bash
git clone https://github.com/tamoco-mocomoco/local-imagegen.git && cd local-imagegen
python3 doctor.py --write-env   # 推奨 .env を書く (cp .env.example .env でも可)
docker compose up -d                                    # CPU
docker compose -f compose.yml -f compose.gpu.yml up -d  # NVIDIA GPU
docker compose logs -f                                  # モデル DL の進捗
```

- 取得するタグは `IMAGE_TAG` で変えられる (既定 `latest`。リリースタグ `v0.1.0` や `sha-xxxxxxx` も公開される)
- 自分でビルドしたいときは `--build` を付ける (`docker compose up -d --build`)
- イメージにはモデルの重みは入っていない。初回起動時に `./models` へダウンロードされる

x86_64 の CPU 専用マシンでイメージを軽くしたい場合:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu docker compose build
```

## モデルの切替

`.env` (Docker / run.py 共通) の `PRESET` で選ぶ。`MODEL_ID` に任意の Hugging Face ID を入れると上書きできる。

| PRESET | モデル | 目安 | 向き |
|---|---|---|---|
| `sdxl-turbo` (既定) | stabilityai/sdxl-turbo | 512px, 4 steps | 速い。CPU でも可。試作・下書き向け。目安 8GB |
| `sd15` | stable-diffusion-v1-5 | 512px, 25 steps | 一番軽い定番。CPU 向け。目安 6GB |
| `sdxl` | stable-diffusion-xl-base-1.0 | 1024px, 25 steps | 高画質。目安 12GB (8GB は「可」) |
| `anime` | cagliostrolab/animagine-xl-4.0 | 832x1216, 28 steps | **アニメ・ゲームイラスト特化** (OpenRAIL++)。キャラ立ち絵向け。目安 12GB |
| `flux-schnell` | black-forest-labs/FLUX.1-schnell | 1024px, 4 steps | 最高クラス (Apache-2.0)。目安 24GB (初回 DL 約 34GB) |
| `zimage-turbo` | Tongyi-MAI/Z-Image-Turbo | 1024px, 8 steps | 日本語・漢字の文字描画が得意 (Apache-2.0)。目安 20GB。要最新 diffusers |

「目安」は fp16 で快適に動く VRAM (NVIDIA) / 統合メモリ (Apple Silicon) で、`python3 doctor.py` がこの表に沿って推奨 / 可 / 不可を判定する。

VRAM が足りない GPU で大きいモデルを動かすときは `CPU_OFFLOAD=1` (CUDA のみ)。

### 実行時のプリセット切替 (再起動不要)

`PRESET` は「既定」で、リクエストごとに `preset` を指定して別のモデルを使える。未ロードのプリセットはその場で読み込んでキャッシュし、
`MAX_LOADED_MODELS` (既定: メモリ 32GB 以上なら 2、それ未満なら 1) を超えたら最も長く使っていないものを解放する。

```bash
python3 gen.py "..." --preset sdxl-turbo --steps 4     # ラフを数秒で
python3 gen.py "..." --preset anime                     # 同じサーバーで仕上げ
python3 gen.py --load anime                             # 生成せずに先に読み込んでおく
python3 gen.py --unload sdxl-turbo                      # メモリから解放
python3 gen.py --status                                 # 読み込み済み一覧など
```

- API: `POST /generate` の `preset`、`POST /load` / `POST /unload` (`{"preset": "..."}`)、`GET /presets` (各プリセットの用途・`tier` (recommended / possible / unsupported) と `reason`・許可・読み込み状態)。`GET /health` の `environment` に device / メモリ / tier 別一覧が入る
- MCP: `generate_image(preset=...)`、`load_preset` / `unload_preset`。MCP サーバーは起動時に `/health` と `/presets` を見て「このマシンで選べるプリセットと使い分けの目安」を Claude に伝えるので、Claude 側が用途に応じて選べる
- `ALLOWED_PRESETS=sdxl-turbo,anime` のようにカンマ区切りで、その環境で選べるプリセットを制限できる。未指定なら環境判定で「不可」になったものを除いた全部
- 切替後の初回だけ読み込み時間 (キャッシュ済みなら 1〜2 分、未取得なら DL 時間が加算) がかかる。2 回目以降は常駐分なら即座

### `anime` プリセットの使い方 (ゲームキャラ向け)

Animagine は Danbooru タグ形式のプロンプトが効く。英文より「カンマ区切りのタグ」で書く。

```bash
python3 gen.py "1girl, solo, knight, silver armor, red cape, long orange hair, full body, standing, simple background, white background" -o knight.png
python3 gen.py "1boy, wizard, pointy hat, chibi, holding staff, full body, white background" --size 1024x1024
```

- 品質タグ (`masterpiece, high score, great score, absurdres`) はプリセットが自動で末尾に付ける。付けたくなければ `--raw`
- negative prompt はプリセット既定 (`lowres, bad anatomy, bad hands, ...`) が入る。`--negative ""` で無効化、`--negative "..."` で差し替え
- 1 人だけ描かせたいときは `solo` を入れる (入れないと背景に別キャラが混ざることがある)
- 立ち絵は縦長 (既定 832x1216)、アイコンや顔は `--size 1024x1024`
- 下書き段階は `--steps 20` で時短、仕上げで 28 に戻す
- 同じキャラの差分は `--seed` を固定し、髪色・服装などの特徴タグを毎回同じ順で書くと揃いやすい
- 透過スプライトが欲しいときは `--rembg` (下記)。`simple background, white background` も入れておくと切り抜きがきれい

### 背景除去で透過 PNG にする (`--rembg`)

拡散モデルは透過画像を直接出せないので、生成後に [rembg](https://github.com/danielgatis/rembg) で背景を抜いて RGBA PNG として保存する。
anime プリセットでは `isnet-anime`、それ以外では `isnet-general-use` モデルを使う (`REMBG_MODEL` で変更可)。初回のみモデル (約 170MB) を `models/rembg/` に取得する。

```bash
# ネイティブ起動は run.py が自動で --extra rembg を付ける。Docker イメージには同梱。
python3 gen.py "1girl, solo, knight, full body, standing, white background" --rembg -o knight.png
```

API では `POST /generate` に `"remove_background": true`、GET では `&remove_background=true`。
足元の影まで残したくないときはプロンプトに `no shadow` を足す (影も半透明で残る)。

### 同一キャラの差分を作る (`--ref` 参照画像)

[IP-Adapter](https://huggingface.co/h94/IP-Adapter) (`ip-adapter-plus_sdxl_vit-h`) で参照画像の見た目 (髪色・衣装・画風) をプロンプトに足す。
SD / SDXL 系プリセット (anime, sdxl, sdxl-turbo, sd15) で使える。初回のみ重み + 画像エンコーダ (約 3GB) を取得する。

```bash
# 1. まず基準になる 1 枚を作る
python3 gen.py "1girl, solo, knight, silver armor, red cape, long orange hair, full body, standing, white background" --seed 7 -o base.png
# 2. それを参照にして別ポーズ・別表情を作る (特徴タグは同じ順で書き続ける)
python3 gen.py "1girl, solo, knight, silver armor, red cape, long orange hair, upper body, smile, waving, white background" --ref base.png -o smile.png
python3 gen.py "1girl, solo, knight, silver armor, red cape, long orange hair, full body, running, side view, white background" --ref base.png --ref-scale 0.8 -o run.png
```

- `--ref-scale`: 0.4 控えめ (プロンプト優先) / 0.6 標準 / 0.8 以上 強め (構図まで寄る)。0.6 でも小物の色が少しずれることがあるので、色を厳密に揃えたいときは 0.7〜0.8 にするか、色をタグで明示する (`red scarf` など)
- マスコット系 (動物キャラ) は `no humans, solo, mascot character, chibi` から始めると安定する
- 参照画像は正方形に白パディングしてから渡すので、縦長の立ち絵をそのまま使える
- 参照あり/なしを切り替えるたびにアダプタを付け外しするため、数秒のオーバーヘッドがある。まとめて作ると速い
- 顔だけ揃えたいときは `IP_ADAPTER_WEIGHT=ip-adapter-plus-face_sdxl_vit-h.safetensors` で起動する
gated モデル (FLUX.1-dev, SD3.5 等) は `HF_TOKEN` を設定する。

## CLI (`gen.py`)

Python 3 さえあれば動く (追加パッケージ不要)。Claude Code などから Bash で呼ぶ想定。

```bash
python3 gen.py "prompt" -o out.png                 # 1 枚保存
python3 gen.py "prompt" --n 4 --seed 42            # 4 枚 (out-1.png ...) / seed 固定
python3 gen.py "prompt" --size 768x768 --steps 4 --guidance 0
python3 gen.py "prompt" --negative "blurry, low quality"   # 対応モデルのみ。--negative "" で既定を無効化
python3 gen.py "prompt" --raw                      # プリセットの品質タグを付けない
python3 gen.py "prompt" --rembg                    # 背景除去して透過 PNG
python3 gen.py "prompt" --ref base.png --ref-scale 0.6   # 参照画像で見た目を引き継ぐ (IP-Adapter)
python3 gen.py "prompt" --json                     # メタ情報を JSON で
IMAGEGEN_URL=http://gpu-box:8000 python3 gen.py "prompt"   # 別マシンのサーバーへ
```

保存先のパスを stdout に 1 行ずつ出力する。モデル読み込み中なら完了まで待つ。

## Claude から直接呼ぶ (MCP サーバー)

[mcp_server.py](mcp_server.py) は HTTP API を MCP ツールにラップした stdio サーバー。登録すると Claude Code / Claude Desktop が
`generate_image` / `server_status` / `list_presets` / `load_preset` / `unload_preset` / `diagnose` を直接呼べる (Bash で gen.py を叩かなくてよい)。生成サーバー本体は別途 `./run.sh` (Windows は `python run.py`) で起動しておく。
`diagnose` は `doctor.py --json` を実行して環境診断と推奨 .env を返すツールで、生成サーバーが起動していなくても動く (`diagnose(write_env=True)` で .env を書く)。
MCP サーバーの instructions には各プリセットの tier (推奨 / 可 / 不可) も入るので、Claude はそのマシンで無理なモデルを選ばない。

```bash
uv sync --extra rembg --extra mcp      # 依存を入れる (run.py が入れるものに mcp を足す)
```

Claude Code に登録 (パスは絶対パスにする):

```bash
claude mcp add imagegen -- "$PWD/.venv/bin/python" "$PWD/mcp_server.py"
```

Claude Desktop は `claude_desktop_config.json` に追記:

```json
{
  "mcpServers": {
    "imagegen": {
      "command": "/ABS/PATH/LocalLLMImages/.venv/bin/python",
      "args": ["/ABS/PATH/LocalLLMImages/mcp_server.py"],
      "env": { "IMAGEGEN_URL": "http://localhost:8000" }
    }
  }
}
```

`generate_image` の引数は CLI と同じ (prompt, preset, negative_prompt, width, height, steps, guidance, seed, n, remove_background, reference_image_path, reference_scale, raw_prompt, out_path)。
保存先パスとメタ情報のテキストに加えて、最大 512px の JPEG プレビューを返す (`IMAGEGEN_PREVIEW_MAX=0` でプレビューを止められる)。
別マシンのサーバーを使うときは `IMAGEGEN_URL` を変える。

## HTTP API

```bash
# JSON
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt":"a red bicycle on the moon","seed":1}'
# => {"images":[{"file":"...png","path":"/outputs/...png","url":"/outputs/...png","seed":1,...}]}

# ブラウザで直接 PNG
open "http://localhost:8000/generate?prompt=a%20red%20bicycle"

# 状態
curl -s localhost:8000/health
```

`POST /generate` のパラメータ: `prompt`, `preset`, `negative_prompt`, `width`, `height`, `steps`, `guidance`, `seed`, `n` (1-8), `raw_prompt`, `remove_background`, `reference_image` (base64), `reference_scale`, `return_base64`。
省略した値はプリセットの既定値になる (`negative_prompt` は省略でプリセット既定、`""` で無効)。生成画像の PNG には prompt / seed / steps / model がメタデータとして埋め込まれる。

## 環境変数

| 変数 | 既定 | 説明 |
|---|---|---|
| `PRESET` | `sdxl-turbo` | 既定のプリセット名 (リクエストの `preset` で上書き可)。環境で「不可」なら警告 (未指定時は推奨に自動切替) |
| `ALLOWED_PRESETS` | 「不可」以外の全部 | この環境で選べるプリセット (カンマ区切り) |
| `MAX_LOADED_MODELS` | メモリ 32GB 以上で 2、未満で 1 | 同時に常駐させるモデル数 |
| `MODEL_ID` | (preset) | 任意の HF モデル ID で上書き |
| `DEVICE` | 自動 | `cuda` / `mps` / `cpu` を強制 |
| `HF_TOKEN` | - | gated モデル用 |
| `CPU_OFFLOAD` | `0` | CUDA で VRAM 節約 |
| `PRELOAD` | `1` | `0` で初回リクエスト時にロード |
| `PORT` | `8000` | 公開ポート |
| `REMBG_MODEL` | `isnet-anime` (anime) / `isnet-general-use` | 背景除去モデル |
| `IP_ADAPTER_WEIGHT` | `ip-adapter-plus_sdxl_vit-h.safetensors` | 参照画像用アダプタ。顔だけなら `ip-adapter-plus-face_sdxl_vit-h.safetensors` |
| `HF_HOME` | `./models` (Docker は `/models`) | モデルキャッシュ |
| `U2NET_HOME` | `./models/rembg` (Docker は `/models/rembg`) | rembg モデルの置き場 |
| `OUTPUT_DIR` | `./outputs` (Docker は `/outputs`) | 出力先 |
| `FAKE_DEVICE` / `FAKE_MEMORY_GB` | - | 検証用。環境判定 (tier) を偽装する。doctor.py は `FAKE_OS` / `FAKE_GPU` / `FAKE_WSL` も見る |

## 関連リポジトリ

同じ作りのローカル生成サービス。

| リポジトリ | 内容 |
|---|---|
| [local-musicgen](https://github.com/tamoco-mocomoco/local-musicgen) | BGM・歌入りの曲の生成 (ACE-Step 1.5)。HTTP API・CLI・MCP サーバー |

## ライセンスと免責

このリポジトリのコード (サーバー・CLI・MCP サーバー・Docker 設定) は [MIT License](LICENSE) です。

**モデルの重みは同梱していません。** 初回起動時に各ユーザーが Hugging Face から自分で取得します。
したがって各モデルのライセンス条件を確認し遵守する責任、および生成した画像の利用に関する責任は、このソフトウェアを使う人にあります。
作者はモデルの提供者ではなく、生成物についていかなる保証もしません。

| プリセット / 部品 | 提供元 | ライセンス | 商用利用 |
|---|---|---|---|
| `anime` (Animagine XL 4.0) | cagliostrolab | [CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | 可 (使用制限あり) |
| `sdxl` (SDXL base 1.0) | Stability AI | CreativeML Open RAIL++-M | 可 (使用制限あり) |
| `sd15` | - | CreativeML Open RAIL-M | 可 (使用制限あり) |
| `sdxl-turbo` | Stability AI | [Stability AI Community License](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md) | **年商 100 万ドル未満のみ無償**。超える場合は Stability AI と別途契約 |
| `flux-schnell` | Black Forest Labs | Apache-2.0 | 可 |
| `zimage-turbo` | Tongyi-MAI | Apache-2.0 | 可 |
| IP-Adapter (`--ref`) | h94 | Apache-2.0 | 可 |
| rembg / isnet-anime (`--rembg`) | danielgatis / SkyTNT | MIT / Apache-2.0 | 可 |

- OpenRAIL 系は商用可だが、違法行為・他者を害する用途・誤情報の生成などの禁止用途 (ライセンス Attachment A) を守る必要がある。
- 学習データ由来で既存キャラクターに酷似した絵が出ることがある。商用素材にするなら作品名・キャラ名をプロンプトに使わない運用を推奨する。
- gated モデル (FLUX.1-dev, SD3.5 など) を `MODEL_ID` で使う場合は、Hugging Face 上でそのモデルの条件に自分で同意する必要がある。
- 上の表は 2026-10-20 時点の確認内容。最終的には各モデルページのライセンス本文を参照すること。

## トラブルシュート

まず `python3 doctor.py` (Windows は `python doctor.py`) で、不足ツール・メモリ不足・ネットワーク・ディスクのどれが原因かを切り分ける。

- **Mac で真っ黒な画像になる**: fp16 の VAE オーバーフロー。`DEVICE=cpu` で切り分け、再現するなら `MODEL_ID` に fp16 修正版 VAE 同梱モデルを指定するか、プリセットを変える。
- **Docker で `could not select device driver "nvidia"`**: NVIDIA Container Toolkit が未インストール。`nvidia-smi` が Docker 内で動くか確認。
- **モデル DL が遅い / 0 B/s で止まる**: Hugging Face の xet 転送が不調なことがある。server/app.py / Dockerfile は既定で `HF_HUB_DISABLE_XET=1` (通常 HTTPS) にしてある。`models/` にキャッシュされるので再起動すれば続きから。
- **メモリ不足**: 小さいプリセットにする、`width/height` を下げる、CUDA なら `CPU_OFFLOAD=1`。
