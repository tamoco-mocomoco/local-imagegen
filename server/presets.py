"""
プリセットの定義と、環境 (device / メモリ) に応じた推奨度の分類。

server/app.py (サーバー) と doctor.py (環境診断) の両方から読む。
doctor.py は torch なしの素の Python 3.9+ で import できる必要があるので、
このモジュールは標準ライブラリ以外に依存しない。

各プリセットの環境要件:
  min_memory_gb  fp16 で快適に動く VRAM (CUDA) / 統合メモリ (Apple Silicon) の目安
  cpu_ok         CPU 生成でも実用的な速さか
  download_gb    初回取得サイズの目安 (概算)
"""
from __future__ import annotations

from typing import Optional

TIER_RECOMMENDED = "recommended"
TIER_POSSIBLE = "possible"
TIER_UNSUPPORTED = "unsupported"
TIERS = (TIER_RECOMMENDED, TIER_POSSIBLE, TIER_UNSUPPORTED)

# possible とみなすメモリの下限 (min_memory_gb に対する割合)
POSSIBLE_RATIO = 0.7

# PRESET 未指定時や既定が使えないときに選ぶ優先順 (先頭ほど優先)
DEFAULT_PRIORITY = ("anime", "sdxl-turbo", "sdxl", "sd15", "flux-schnell", "zimage-turbo")

# steps/guidance/size are defaults; every request can override them.
PRESETS: dict[str, dict] = {
    "sdxl-turbo": dict(
        model_id="stabilityai/sdxl-turbo", steps=4, guidance=0.0,
        width=512, height=512, variant="fp16",
        min_memory_gb=8, cpu_ok=True, download_gb=7,
        description="高速な下書き用 (512px, 4 steps, 数秒)。CPU でも実用。構図やプロンプトの試行錯誤向け。商用は年商 100 万ドル未満のみ無償",
    ),
    "sd15": dict(
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", steps=25, guidance=7.5,
        width=512, height=512, variant="fp16",
        min_memory_gb=6, cpu_ok=True, download_gb=3,
        description="一番軽い定番 SD1.5 (512px)。CPU 向け。品質は控えめ",
    ),
    "sdxl": dict(
        model_id="stabilityai/stable-diffusion-xl-base-1.0", steps=25, guidance=7.0,
        width=1024, height=1024, variant="fp16",
        min_memory_gb=12, cpu_ok=False, download_gb=7,
        description="汎用の高画質 1024px (SDXL)。写実からイラストまで。GPU 8GB+ / Mac 16GB+",
    ),
    "anime": dict(
        model_id="cagliostrolab/animagine-xl-4.0", steps=28, guidance=6.0,
        width=832, height=1216, scheduler="EulerAncestralDiscreteScheduler", rembg="isnet-anime",
        prompt_suffix="masterpiece, high score, great score, absurdres",
        negative_prompt=("lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, "
                         "fewer digits, cropped, worst quality, low quality, low score, bad score, "
                         "average score, signature, watermark, jpeg artifacts, blurry"),
        min_memory_gb=12, cpu_ok=False, download_gb=7,
        description="アニメ・ゲームキャラの仕上げ用 (Animagine XL 4.0, 1024px, 約 1 分)。Danbooru タグ形式で 'solo' を入れる。参照画像・背景除去と相性良",
    ),
    "flux-schnell": dict(
        model_id="black-forest-labs/FLUX.1-schnell", steps=4, guidance=0.0,
        width=1024, height=1024, dtype="bf16",
        min_memory_gb=24, cpu_ok=False, download_gb=34,
        description="写実・コンセプトアート向け最高クラス (FLUX.1, Apache-2.0)。自然文プロンプト。GPU 24GB+ / Mac 32GB+。参照画像は非対応",
    ),
    "zimage-turbo": dict(
        model_id="Tongyi-MAI/Z-Image-Turbo", steps=8, guidance=0.0,
        width=1024, height=1024, dtype="bf16",
        min_memory_gb=20, cpu_ok=False, download_gb=20,
        description="日本語・漢字の文字入り画像が得意 (Z-Image-Turbo, Apache-2.0)。要 diffusers 最新。参照画像は非対応",
    ),
}

# プリセット以外に初回取得が必要になる部品 (GB, 概算)
EXTRA_DOWNLOAD_GB = {"ip-adapter": 3, "rembg": 0.2}
EXTRA_MODEL_IDS = {"ip-adapter": "h94/IP-Adapter"}


def classify_preset(preset: dict, device: str, memory_gb: float) -> tuple[str, str]:
    """1 プリセットを (tier, 理由) に分類する。

    device:    "cuda" / "mps" / "cpu"
    memory_gb: cuda なら VRAM、mps / cpu なら物理メモリ。0 以下は不明扱い
    """
    need = preset.get("min_memory_gb")
    cpu_ok = bool(preset.get("cpu_ok", False))
    if need is None:
        return TIER_POSSIBLE, "カスタムモデルのため要件不明"
    need = float(need)
    mem = float(memory_gb or 0)

    if device == "cpu":
        if mem <= 0:
            tier = TIER_POSSIBLE if cpu_ok else TIER_UNSUPPORTED
            return tier, "メモリ容量を取得できなかった"
        if cpu_ok:
            if mem >= need:
                return TIER_RECOMMENDED, f"CPU 生成でも実用的 (RAM {mem:.0f}GB)"
            if mem >= need * POSSIBLE_RATIO:
                return TIER_POSSIBLE, f"RAM {mem:.0f}GB は目安 {need:.0f}GB にやや足りない。小さい解像度で"
            return TIER_UNSUPPORTED, f"RAM {mem:.0f}GB では目安 {need:.0f}GB に足りない"
        # CPU 向きではないが、RAM が十分なら動きはする (fp32 で非常に遅い)
        if mem >= need * 2:
            return TIER_POSSIBLE, f"CPU では動くが非常に遅い (1 枚数十分)。GPU 推奨"
        return TIER_UNSUPPORTED, f"CPU では実用にならず、RAM {mem:.0f}GB では fp32 の読み込みも厳しい (目安 {need * 2:.0f}GB)"

    kind = "VRAM" if device == "cuda" else "統合メモリ"
    if mem <= 0:
        return TIER_POSSIBLE, "メモリ容量を取得できなかった"
    if mem >= need:
        return TIER_RECOMMENDED, f"{kind} {mem:.0f}GB >= 目安 {need:.0f}GB"
    if mem >= need * POSSIBLE_RATIO:
        hint = "CPU_OFFLOAD=1 か小さい解像度で" if device == "cuda" else "小さい解像度で、他のアプリを閉じて"
        return TIER_POSSIBLE, f"{kind} {mem:.0f}GB は目安 {need:.0f}GB にやや足りない。{hint}"
    return TIER_UNSUPPORTED, f"{kind} {mem:.0f}GB では目安 {need:.0f}GB に足りない"


def classify_presets(device: str, memory_gb: float, presets: Optional[dict] = None) -> dict[str, dict]:
    """全プリセットを {name: {"tier": ..., "reason": ...}} に分類する。"""
    presets = PRESETS if presets is None else presets
    out: dict[str, dict] = {}
    for name, p in presets.items():
        tier, reason = classify_preset(p, device, memory_gb)
        out[name] = {"tier": tier, "reason": reason}
    return out


def by_tier(classified: dict[str, dict]) -> dict[str, list[str]]:
    """{tier: [name, ...]} にまとめる (全 tier のキーを必ず持つ)。"""
    out: dict[str, list[str]] = {t: [] for t in TIERS}
    for name, c in classified.items():
        out.setdefault(c["tier"], []).append(name)
    return out


def best_recommended(classified: dict[str, dict], allow_possible: bool = True) -> Optional[str]:
    """既定にするのに最も良いプリセット。DEFAULT_PRIORITY 順に recommended を探し、無ければ possible。"""
    names = list(DEFAULT_PRIORITY) + [n for n in classified if n not in DEFAULT_PRIORITY]
    for want in (TIER_RECOMMENDED, TIER_POSSIBLE) if allow_possible else (TIER_RECOMMENDED,):
        for n in names:
            if classified.get(n, {}).get("tier") == want:
                return n
    return None


def default_max_loaded_models(memory_gb: float) -> int:
    """同時常駐モデル数の既定。メモリ 32GB 以上なら 2、未満なら 1。"""
    return 2 if float(memory_gb or 0) >= 32 else 1
