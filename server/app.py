"""
Local text-to-image server.

- Device auto-detect: CUDA (NVIDIA) / MPS (Apple Silicon, native only) / CPU
- Presets (PRESETS) are switchable per request: `preset` on /generate loads on demand and caches
  up to MAX_LOADED_MODELS pipelines (LRU eviction). ALLOWED_PRESETS restricts what this machine offers.
- HTTP API:  POST /generate (JSON)  /  GET /generate?prompt=...  (PNG)  /  POST /load  /  POST /unload
- Web UI:    GET /
- Outputs saved under OUTPUT_DIR and served at /outputs/<file>
"""
from __future__ import annotations

import base64
import gc
import inspect
import io
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- environment defaults
# torch / diffusers / huggingface_hub を import する前に決める。どの OS でも
#   uv run --extra rembg uvicorn server.app:app --host 0.0.0.0 --port 8000
# だけで同じ動きになるよう、起動スクリプトに頼らず既定値をここで入れる。
PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """プロジェクト直下の .env (KEY=VALUE、# コメント、クォート除去) を読み、未設定の変数だけ入れる。"""
    if not path.is_file():
        return
    import re
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))


_dotenv = os.environ.get("IMAGEGEN_DOTENV")  # 別の .env を使う / 空文字で読まない (テスト用)
if _dotenv is None:
    _load_dotenv(PROJECT_DIR / ".env")
elif _dotenv:
    _load_dotenv(Path(_dotenv))
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / "models"))
os.environ.setdefault("U2NET_HOME", str(Path(os.environ["HF_HOME"]) / "rembg"))  # rembg のモデル置き場
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # hf-xet 経由の転送が不安定なことがあるので通常 HTTPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, PngImagePlugin
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("imagegen")

# ---------------------------------------------------------------- presets
# 定義は server/presets.py (doctor.py と共有)。steps/guidance/size は既定値で、リクエストごとに上書きできる。
from .presets import (  # noqa: E402
    PRESETS, TIER_RECOMMENDED, TIER_UNSUPPORTED, best_recommended, by_tier, classify_presets,
    default_max_loaded_models,
)

_PRESET_ENV = os.environ.get("PRESET")  # 明示指定かどうかを後で見る
DEFAULT_PRESET = os.environ.get("PRESET", "sdxl-turbo")
if os.environ.get("MODEL_ID"):
    # 任意の HF モデル。既知プリセット名なら model_id だけ差し替え、未知なら汎用既定値で新規登録
    entry = dict(PRESETS.get(DEFAULT_PRESET) or dict(steps=25, guidance=7.0, width=1024, height=1024,
                                                     description="MODEL_ID で指定したカスタムモデル"))
    entry["model_id"] = os.environ["MODEL_ID"]
    entry.pop("variant", None)
    PRESETS[DEFAULT_PRESET] = entry
if DEFAULT_PRESET not in PRESETS:
    log.warning("unknown PRESET=%s; falling back to sdxl-turbo", DEFAULT_PRESET)
    DEFAULT_PRESET = "sdxl-turbo"

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR") or ("/outputs" if Path("/outputs").is_dir() else PROJECT_DIR / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD", "0") == "1"
PRELOAD = os.environ.get("PRELOAD", "1") == "1"
REMBG_MODEL_OVERRIDE = os.environ.get("REMBG_MODEL")  # 指定すると全プリセットでこれを使う

# 参照画像 (IP-Adapter)。パイプラインの種類ごとに重みが違う。
IP_ADAPTER_REPO = os.environ.get("IP_ADAPTER_REPO", "h94/IP-Adapter")
IP_ADAPTER_CONFIGS = {
    "sdxl": dict(subfolder=os.environ.get("IP_ADAPTER_SUBFOLDER", "sdxl_models"),
                 weight=os.environ.get("IP_ADAPTER_WEIGHT", "ip-adapter-plus_sdxl_vit-h.safetensors"),
                 encoder=os.environ.get("IP_ADAPTER_ENCODER", "models/image_encoder")),
    "sd15": dict(subfolder="models", weight="ip-adapter-plus_sd15.safetensors", encoder="models/image_encoder"),
}


# ---------------------------------------------------------------- device
def pick_device() -> str:
    forced = os.environ.get("FAKE_DEVICE") or os.environ.get("DEVICE")
    if forced:
        if os.environ.get("FAKE_DEVICE"):
            log.warning("FAKE_DEVICE=%s (検証用。実際の計算もこのデバイスで行う)", forced)
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


def pick_dtype(preset: dict) -> torch.dtype:
    if DEVICE == "cpu":
        return torch.float32
    if preset.get("dtype") == "bf16":
        return torch.bfloat16
    return torch.float16


def total_memory_gb() -> float:
    """CUDA なら VRAM、それ以外は物理メモリ (macOS / Linux は sysconf、Windows は kernel32)。単位は GiB。"""
    try:
        if DEVICE == "cuda":
            return torch.cuda.get_device_properties(0).total_memory / 2**30
        if hasattr(os, "sysconf"):
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        import ctypes  # Windows

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
        return st.ullTotalPhys / 2**30
    except Exception:
        return 0.0


MEMORY_GB = round(float(os.environ.get("FAKE_MEMORY_GB") or total_memory_gb()), 1)
FAKE_ENVIRONMENT = bool(os.environ.get("FAKE_DEVICE") or os.environ.get("FAKE_MEMORY_GB"))  # 検証用の偽装
MAX_LOADED_MODELS = int(os.environ.get("MAX_LOADED_MODELS") or default_max_loaded_models(MEMORY_GB))

# ---------------------------------------------------------------- environment-based tiers
# 各プリセットを recommended / possible / unsupported に分類し、ALLOWED_PRESETS の既定と既定プリセットを決める。
PRESET_TIERS: dict[str, dict] = classify_presets(DEVICE, MEMORY_GB)
_tiers = by_tier(PRESET_TIERS)
DEFAULT_PRESET_SWITCHED_FROM: Optional[str] = None
if PRESET_TIERS.get(DEFAULT_PRESET, {}).get("tier") == TIER_UNSUPPORTED:
    reason = PRESET_TIERS[DEFAULT_PRESET]["reason"]
    if _PRESET_ENV:
        log.warning("PRESET=%s はこの環境 (%s, %sGB) では unsupported: %s。明示指定なのでそのまま使う",
                    DEFAULT_PRESET, DEVICE, MEMORY_GB, reason)
    else:
        better = best_recommended(PRESET_TIERS)
        if better and better != DEFAULT_PRESET:
            log.warning("既定の PRESET=%s はこの環境 (%s, %sGB) では unsupported: %s。%s に切り替える",
                        DEFAULT_PRESET, DEVICE, MEMORY_GB, reason, better)
            DEFAULT_PRESET_SWITCHED_FROM, DEFAULT_PRESET = DEFAULT_PRESET, better
        else:
            log.warning("既定の PRESET=%s はこの環境では unsupported (%s) だが代替が無いのでそのまま使う", DEFAULT_PRESET, reason)

_allowed_env = [p.strip() for p in os.environ.get("ALLOWED_PRESETS", "").split(",") if p.strip()]
if _allowed_env:
    ALLOWED_PRESETS: list[str] = [p for p in _allowed_env if p in PRESETS]
else:  # 未指定なら unsupported 以外
    ALLOWED_PRESETS = [p for p in PRESETS if PRESET_TIERS[p]["tier"] != TIER_UNSUPPORTED]
if DEFAULT_PRESET not in ALLOWED_PRESETS:
    ALLOWED_PRESETS.insert(0, DEFAULT_PRESET)


def environment_info() -> dict:
    """/health の environment。doctor.py の判定と同じ材料 (device とメモリ) で付けた tier を返す。"""
    import platform
    return dict(
        os=platform.system(), os_version=platform.release(), machine=platform.machine(),
        python=platform.python_version(),
        device=DEVICE, memory_gb=MEMORY_GB,
        memory_kind="vram" if DEVICE == "cuda" else "ram",
        fake=FAKE_ENVIRONMENT,
        recommended=_tiers[TIER_RECOMMENDED], possible=_tiers["possible"], unsupported=_tiers[TIER_UNSUPPORTED],
        default_preset=DEFAULT_PRESET, default_preset_switched_from=DEFAULT_PRESET_SWITCHED_FROM,
        default_preset_tier=PRESET_TIERS.get(DEFAULT_PRESET, {}).get("tier"),
        allowed_presets_from_env=bool(_allowed_env),
    )


def empty_cache() -> None:
    gc.collect()
    try:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------- model
@dataclass
class LoadedModel:
    name: str
    preset: dict
    pipe: object
    dtype: torch.dtype
    accepts_negative: bool
    loaded_in: float
    last_used: float = field(default_factory=time.time)
    ip_adapter_loaded: bool = False

    @property
    def ip_adapter_kind(self) -> Optional[str]:
        cls = self.pipe.__class__.__name__
        if not hasattr(self.pipe, "load_ip_adapter"):
            return None
        if cls.startswith("StableDiffusionXL"):
            return "sdxl"
        if cls == "StableDiffusionPipeline":
            return "sd15"
        return None

    def rembg_model(self) -> str:
        return REMBG_MODEL_OVERRIDE or self.preset.get("rembg") or "isnet-general-use"

    def info(self) -> dict:
        return dict(model=self.preset["model_id"], loaded_in_seconds=self.loaded_in,
                    last_used=time.strftime("%H:%M:%S", time.localtime(self.last_used)),
                    ip_adapter=self.ip_adapter_kind is not None, ip_adapter_loaded=self.ip_adapter_loaded,
                    dtype=str(self.dtype).replace("torch.", ""))


class Engine:
    def __init__(self) -> None:
        self.models: dict[str, LoadedModel] = {}
        self.lock = threading.Lock()  # covers loading, eviction and generation (single accelerator)
        self.loading: set[str] = set()
        self.load_errors: dict[str, str] = {}
        self.rembg_sessions: dict[str, object] = {}

    # ---- preset resolution
    @staticmethod
    def resolve(name: Optional[str]) -> str:
        name = name or DEFAULT_PRESET
        if name not in PRESETS:
            raise HTTPException(400, f"不明なプリセット '{name}'。選べるもの: {', '.join(ALLOWED_PRESETS)}")
        if name not in ALLOWED_PRESETS:
            raise HTTPException(400, f"プリセット '{name}' はこのサーバーでは許可されていません (ALLOWED_PRESETS)。選べるもの: {', '.join(ALLOWED_PRESETS)}")
        return name

    # ---- load / evict (call with self.lock held)
    def get(self, name: str) -> LoadedModel:
        m = self.models.get(name)
        if m is not None:
            m.last_used = time.time()
            return m
        self._evict_for(name)
        return self._load(name)

    def _evict_for(self, incoming: str) -> None:
        while len(self.models) >= MAX_LOADED_MODELS and self.models:
            victim = min(self.models.values(), key=lambda m: m.last_used)
            self._unload(victim.name)

    def _unload(self, name: str) -> None:
        m = self.models.pop(name, None)
        if m is None:
            return
        try:
            if m.ip_adapter_loaded:
                m.pipe.unload_ip_adapter()
        except Exception:
            pass
        del m.pipe, m
        empty_cache()
        log.info("unloaded %s (loaded now: %s)", name, list(self.models) or "-")

    def _load(self, name: str) -> LoadedModel:
        from diffusers import AutoPipelineForText2Image

        preset = PRESETS[name]
        dtype = pick_dtype(preset)
        model_id = preset["model_id"]
        log.info("loading preset %s (%s) on %s (%s) ...", name, model_id, DEVICE, dtype)
        self.loading.add(name)
        t0 = time.time()
        try:
            kwargs: dict = dict(torch_dtype=dtype)
            if os.environ.get("HF_TOKEN"):
                kwargs["token"] = os.environ["HF_TOKEN"]
            try:
                if preset.get("variant"):
                    pipe = AutoPipelineForText2Image.from_pretrained(model_id, variant=preset["variant"], **kwargs)
                else:
                    pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
            except Exception as e:  # e.g. no fp16 variant files
                if not preset.get("variant"):
                    raise
                log.warning("variant=%s failed (%s); retrying without variant", preset["variant"], e)
                pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)

            if preset.get("scheduler"):
                import diffusers as _d
                pipe.scheduler = getattr(_d, preset["scheduler"]).from_config(pipe.scheduler.config)

            if DEVICE == "cuda" and CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(DEVICE)
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            m = LoadedModel(
                name=name, preset=preset, pipe=pipe, dtype=dtype,
                accepts_negative="negative_prompt" in inspect.signature(pipe.__call__).parameters,
                loaded_in=round(time.time() - t0, 1),
            )
            self._set_attention_slicing(m, True)
            self.models[name] = m
            self.load_errors.pop(name, None)
            log.info("preset %s ready in %ss (loaded now: %s)", name, m.loaded_in, list(self.models))
            return m
        except Exception as e:
            self.load_errors[name] = str(e)
            empty_cache()
            raise
        finally:
            self.loading.discard(name)

    def load_preset(self, name: str) -> LoadedModel:
        with self.lock:
            return self.get(name)

    def unload_preset(self, name: str) -> bool:
        with self.lock:
            existed = name in self.models
            self._unload(name)
            return existed

    # ---- 背景除去
    def remove_background(self, m: LoadedModel, img: Image.Image) -> Image.Image:
        try:
            from rembg import new_session, remove
        except ImportError:
            raise HTTPException(400, "rembg が未インストールです: uv sync --extra rembg (Docker イメージには同梱)")
        model = m.rembg_model()
        if model not in self.rembg_sessions:
            log.info("loading rembg model %s ...", model)
            self.rembg_sessions[model] = new_session(model)
        return remove(img, session=self.rembg_sessions[model]).convert("RGBA")

    # ---- 参照画像 (IP-Adapter)
    def set_ip_adapter(self, m: LoadedModel, enabled: bool) -> None:
        """参照画像あり/なしで IP-Adapter を付け外しする (付けたままだと参照なし生成ができないため)。"""
        if enabled and not m.ip_adapter_loaded:
            kind = m.ip_adapter_kind
            if kind is None:
                raise HTTPException(400, f"プリセット '{m.name}' は参照画像 (IP-Adapter) に対応していません。SD/SDXL 系 (anime, sdxl, sdxl-turbo, sd15) を使ってください")
            cfg = IP_ADAPTER_CONFIGS[kind]
            log.info("loading IP-Adapter %s/%s/%s for %s ...", IP_ADAPTER_REPO, cfg["subfolder"], cfg["weight"], m.name)
            t0 = time.time()
            kwargs = {}
            if os.environ.get("HF_TOKEN"):
                kwargs["token"] = os.environ["HF_TOKEN"]
            # attention slicing が有効だと IP-Adapter の attention processor 差し替えに失敗する (diffusers の制約)
            self._set_attention_slicing(m, False)
            try:
                m.pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=cfg["subfolder"], weight_name=cfg["weight"],
                                       image_encoder_folder=cfg["encoder"], **kwargs)
                if DEVICE != "cuda" or not CPU_OFFLOAD:
                    m.pipe.image_encoder.to(DEVICE, dtype=m.dtype)
            except Exception:
                try:
                    m.pipe.unload_ip_adapter()
                except Exception:
                    pass
                self._set_attention_slicing(m, True)
                raise
            m.ip_adapter_loaded = True
            log.info("IP-Adapter ready in %.1fs", time.time() - t0)
        elif not enabled and m.ip_adapter_loaded:
            m.pipe.unload_ip_adapter()
            m.ip_adapter_loaded = False
            self._set_attention_slicing(m, True)
            log.info("IP-Adapter unloaded from %s", m.name)

    @staticmethod
    def _set_attention_slicing(m: LoadedModel, on: bool) -> None:
        """MPS / CPU では省メモリのため attention slicing を使う (IP-Adapter 使用中は切る)。"""
        if DEVICE not in ("mps", "cpu") or not hasattr(m.pipe, "enable_attention_slicing"):
            return
        if on:
            m.pipe.enable_attention_slicing()
        else:
            m.pipe.disable_attention_slicing()

    @staticmethod
    def decode_reference(data: str) -> Image.Image:
        if "," in data[:64] and data.lstrip().startswith("data:"):
            data = data.split(",", 1)[1]
        try:
            img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"reference_image を画像として読めません: {e}")
        # CLIP は中央 224px 正方形で見るので、縦長の立ち絵は白でパディングして全身が入るようにする
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas

    # ---- generation
    def generate(self, req: "GenerateRequest") -> list[dict]:
        name = self.resolve(req.preset)
        preset = PRESETS[name]
        ref_img = self.decode_reference(req.reference_image) if req.reference_image else None
        seed = req.seed if req.seed is not None else int.from_bytes(os.urandom(4), "little")
        width = req.width or preset["width"]
        height = req.height or preset["height"]
        steps = req.steps or preset["steps"]
        guidance = preset["guidance"] if req.guidance is None else req.guidance

        prompt = req.prompt
        if preset.get("prompt_suffix") and not req.raw_prompt:
            prompt = f"{prompt}, {preset['prompt_suffix']}"
        # negative_prompt: None -> プリセット既定 / "" -> なし / 文字列 -> そのまま
        negative = preset.get("negative_prompt") if req.negative_prompt is None else req.negative_prompt

        call: dict = dict(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
                          width=width, height=height, num_images_per_prompt=req.n)

        with self.lock:  # one load/generation at a time
            m = self.get(name)
            if negative and m.accepts_negative:
                call["negative_prompt"] = negative
            self.set_ip_adapter(m, ref_img is not None)
            if ref_img is not None:
                m.pipe.set_ip_adapter_scale(req.reference_scale)
                call["ip_adapter_image"] = ref_img
            gen = torch.Generator(device="cpu").manual_seed(seed)
            t0 = time.time()
            images = m.pipe(generator=gen, **call).images
            elapsed = round(time.time() - t0, 2)
            m.last_used = time.time()
            if req.remove_background:
                t1 = time.time()
                images = [self.remove_background(m, im) for im in images]
                log.info("background removed (%s) in %.1fs", m.rembg_model(), time.time() - t1)

        results = []
        for i, img in enumerate(images):
            fname = f"{time.strftime('%Y%m%d-%H%M%S')}-{seed}-{uuid.uuid4().hex[:6]}.png"
            path = OUTPUT_DIR / fname
            meta = PngImagePlugin.PngInfo()
            meta.add_text("prompt", prompt)
            meta.add_text("negative_prompt", negative or "")
            meta.add_text("seed", str(seed))
            meta.add_text("steps", str(steps))
            meta.add_text("guidance", str(guidance))
            meta.add_text("model", preset["model_id"])
            meta.add_text("preset", name)
            if ref_img is not None:
                meta.add_text("reference_scale", str(req.reference_scale))
            img.save(path, pnginfo=meta)
            item = dict(
                file=fname, path=str(path.resolve()), url=f"/outputs/{fname}",
                seed=seed, index=i, width=img.width, height=img.height,
                steps=steps, guidance=guidance, preset=name, model=preset["model_id"], seconds=elapsed,
                prompt=prompt, negative_prompt=negative or "",
                transparent=req.remove_background, reference_scale=req.reference_scale if ref_img is not None else None,
            )
            if req.return_base64:
                buf = io.BytesIO()
                img.save(buf, format="PNG", pnginfo=meta)
                item["base64"] = base64.b64encode(buf.getvalue()).decode()
            results.append(item)
        log.info("[%s] generated %d image(s) in %ss seed=%s %r", name, len(images), elapsed, seed, req.prompt[:60])
        return results


engine = Engine()


# ---------------------------------------------------------------- API
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    preset: Optional[str] = None  # 省略時はサーバー既定 (PRESET)。未ロードなら要求時に読み込む
    negative_prompt: Optional[str] = None
    width: Optional[int] = Field(None, ge=64, le=2048)
    height: Optional[int] = Field(None, ge=64, le=2048)
    steps: Optional[int] = Field(None, ge=1, le=150)
    guidance: Optional[float] = Field(None, ge=0, le=30)
    seed: Optional[int] = None
    n: int = Field(1, ge=1, le=8)
    raw_prompt: bool = False  # True ならプリセットの prompt_suffix を付けない
    remove_background: bool = False  # rembg で背景を透過にする
    reference_image: Optional[str] = None  # base64 PNG/JPEG。IP-Adapter でキャラの見た目を引き継ぐ
    reference_scale: float = Field(0.6, ge=0, le=1.5)  # 参照の強さ。0.4 控えめ / 0.6 標準 / 0.8 強め
    return_base64: bool = False


class PresetRequest(BaseModel):
    preset: str


app = FastAPI(title="Local ImageGen", version="0.2.0")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.on_event("startup")
def _startup() -> None:
    log.info("device=%s memory=%sGB default=%s allowed=%s max_loaded=%s outputs=%s",
             DEVICE, MEMORY_GB, DEFAULT_PRESET, ALLOWED_PRESETS, MAX_LOADED_MODELS, OUTPUT_DIR)
    log.info("preset tiers: recommended=%s possible=%s unsupported=%s",
             _tiers[TIER_RECOMMENDED], _tiers["possible"], _tiers[TIER_UNSUPPORTED])
    if PRELOAD:
        threading.Thread(target=_safe_preload, daemon=True).start()


def _safe_preload() -> None:
    try:
        engine.load_preset(DEFAULT_PRESET)
    except Exception:
        log.exception("preload of %s failed", DEFAULT_PRESET)


def _preset_view(name: str) -> dict:
    p = PRESETS[name]
    d = {k: v for k, v in p.items()}
    t = PRESET_TIERS.get(name, {})
    d.update(allowed=name in ALLOWED_PRESETS, loaded=name in engine.models, default=name == DEFAULT_PRESET,
             loading=name in engine.loading, error=engine.load_errors.get(name),
             tier=t.get("tier"), reason=t.get("reason"))
    return d


@app.get("/health")
def health() -> dict:
    default_loaded = DEFAULT_PRESET in engine.models
    err = engine.load_errors.get(DEFAULT_PRESET)
    p = PRESETS[DEFAULT_PRESET]
    return dict(
        ok=default_loaded or not PRELOAD,
        loading=PRELOAD and not default_loaded and err is None,
        error=err,
        device=DEVICE,
        memory_gb=MEMORY_GB,
        dtype=str(pick_dtype(p)).replace("torch.", ""),
        preset=DEFAULT_PRESET,
        model=p["model_id"],
        defaults={k: p.get(k) for k in ("steps", "guidance", "width", "height", "prompt_suffix", "negative_prompt")},
        loaded_in_seconds=engine.models[DEFAULT_PRESET].loaded_in if default_loaded else None,
        loaded_presets={n: m.info() for n, m in engine.models.items()},
        loading_presets=sorted(engine.loading),
        allowed_presets=ALLOWED_PRESETS,
        max_loaded_models=MAX_LOADED_MODELS,
        load_errors=engine.load_errors,
        environment=environment_info(),
    )


@app.get("/presets")
def presets() -> dict:
    return {name: _preset_view(name) for name in PRESETS}


@app.post("/load")
def load(req: PresetRequest) -> dict:
    name = engine.resolve(req.preset)
    try:
        m = engine.load_preset(name)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("load failed")
        raise HTTPException(500, f"プリセット '{name}' の読み込みに失敗: {e}")
    return dict(preset=name, loaded_in_seconds=m.loaded_in, loaded_presets=list(engine.models))


@app.post("/unload")
def unload(req: PresetRequest) -> dict:
    if req.preset not in PRESETS:
        raise HTTPException(400, f"不明なプリセット '{req.preset}'")
    existed = engine.unload_preset(req.preset)
    return dict(preset=req.preset, unloaded=existed, loaded_presets=list(engine.models))


@app.post("/generate")
def generate(req: GenerateRequest) -> JSONResponse:
    try:
        return JSONResponse(dict(images=engine.generate(req)))
    except HTTPException:
        raise
    except Exception as e:
        log.exception("generation failed")
        raise HTTPException(500, str(e))


@app.get("/generate")
def generate_get(
    prompt: str = Query(..., min_length=1),
    preset: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: Optional[int] = None,
    raw_prompt: bool = False,
    remove_background: bool = False,
) -> Response:
    """Browser-friendly: returns the PNG directly."""
    req = GenerateRequest(prompt=prompt, preset=preset, negative_prompt=negative_prompt, width=width, height=height,
                          steps=steps, guidance=guidance, seed=seed, raw_prompt=raw_prompt,
                          remove_background=remove_background)
    try:
        item = engine.generate(req)[0]
    except HTTPException:
        raise
    except Exception as e:
        log.exception("generation failed")
        raise HTTPException(500, str(e))
    data = (OUTPUT_DIR / item["file"]).read_bytes()
    return Response(data, media_type="image/png",
                    headers={"X-Seed": str(item["seed"]), "X-File": item["file"], "X-Preset": item["preset"]})


INDEX_HTML = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>Local ImageGen</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}
 textarea,input,select{width:100%;box-sizing:border-box;padding:.5rem;font-size:1rem}
 .row{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.5rem;margin:.5rem 0}
 button{padding:.6rem 1.2rem;font-size:1rem;cursor:pointer}
 #out img{max-width:100%;margin:.5rem 0;border:1px solid #ddd}
 #status,#pdesc{color:#666;margin:.5rem 0}
 small{color:#888}
</style></head><body>
<h1>Local ImageGen</h1>
<div id="info"><small>loading…</small></div>
<div class="row">
 <label>preset<select id="preset"></select></label>
</div>
<div id="pdesc"></div>
<textarea id="prompt" rows="3" placeholder="prompt (英語推奨)">a cute shiba inu wearing a kimono, ukiyo-e style, highly detailed</textarea>
<input id="neg" placeholder="negative prompt (任意・対応モデルのみ)">
<div class="row">
 <label>W<input id="w" type="number" step="64"></label>
 <label>H<input id="h" type="number" step="64"></label>
 <label>steps<input id="steps" type="number"></label>
 <label>guidance<input id="g" type="number" step="0.5"></label>
 <label>seed<input id="seed" type="number" placeholder="random"></label>
 <label>n<input id="n" type="number" value="1" min="1" max="8"></label>
</div>
<div class="row">
 <label><input id="rembg" type="checkbox" style="width:auto"> 背景を透過 (rembg)</label>
 <label>参照画像 (IP-Adapter)<input id="ref" type="file" accept="image/*"></label>
 <label>参照の強さ<input id="refscale" type="number" step="0.1" value="0.6" min="0" max="1.5"></label>
</div>
<button id="go">Generate</button> <span id="status"></span>
<div id="out"></div>
<script>
const $=id=>document.getElementById(id);
let PRESETS={};
function applyPreset(name){
  const p=PRESETS[name]; if(!p) return;
  $('w').value=p.width;$('h').value=p.height;$('steps').value=p.steps;$('g').value=p.guidance;
  $('neg').placeholder=p.negative_prompt?'negative prompt (空欄ならプリセット既定: '+p.negative_prompt.slice(0,60)+'…)':'negative prompt (任意・対応モデルのみ)';
  $('prompt').placeholder=p.prompt_suffix?'prompt (末尾に自動付与: '+p.prompt_suffix+')':'prompt (英語推奨)';
  $('pdesc').innerHTML=`<small>${p.description||''} · ${p.loaded?'読み込み済み':'未読み込み (初回は 1〜2 分、初回 DL はさらに)'}</small>`;
}
async function refresh(){
  const h=await (await fetch('/health')).json(); PRESETS=await (await fetch('/presets')).json();
  const loaded=Object.keys(h.loaded_presets).join(', ')||'-';
  $('info').innerHTML=`<small>device: ${h.device} (${h.memory_gb}GB) · 読み込み済み: ${loaded} (最大 ${h.max_loaded_models}) · ${h.ok?'ready':(h.error?'ERROR: '+h.error:'loading '+h.preset+'…')}</small>`;
  const sel=$('preset'); const cur=sel.value||h.preset; sel.innerHTML='';
  for(const n of h.allowed_presets){const o=document.createElement('option');o.value=n;o.textContent=n+(PRESETS[n].loaded?' ●':'');sel.appendChild(o);}
  sel.value=cur; applyPreset(cur);
  if(!h.ok&&!h.error) setTimeout(refresh,5000);
}
$('preset').onchange=()=>applyPreset($('preset').value);
refresh();
$('go').onclick=async()=>{
  const body={prompt:$('prompt').value,preset:$('preset').value,negative_prompt:$('neg').value||null,width:+$('w').value,height:+$('h').value,
    steps:+$('steps').value,guidance:+$('g').value,seed:$('seed').value?+$('seed').value:null,n:+$('n').value,
    remove_background:$('rembg').checked,reference_scale:+$('refscale').value};
  const f=$('ref').files[0]; if(f){body.reference_image=await new Promise(r=>{const fr=new FileReader();fr.onload=()=>r(fr.result);fr.readAsDataURL(f);});}
  $('status').textContent='generating…';$('go').disabled=true;const t0=Date.now();
  try{
    const r=await fetch('/generate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json(); if(!r.ok) throw new Error(j.detail||r.statusText);
    for(const im of j.images){const d=document.createElement('div');
      d.innerHTML=`<a href="${im.url}" target="_blank"><img src="${im.url}" style="${im.transparent?'background:repeating-conic-gradient(#ccc 0 25%,#fff 0 50%) 0 0/20px 20px':''}"></a><br><small>${im.preset} · seed ${im.seed} · ${im.seconds}s · ${im.file}${im.transparent?' · 透過':''}${im.reference_scale!=null?' · ref '+im.reference_scale:''}</small>`;
      $('out').prepend(d);}
    $('status').textContent=`done (${((Date.now()-t0)/1000).toFixed(1)}s)`;
  }catch(e){$('status').textContent='error: '+e.message}
  $('go').disabled=false; refresh();
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
