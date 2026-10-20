#!/usr/bin/env python3
"""
環境診断 (doctor)。このマシンで Local ImageGen をどう動かすのが良いかを調べて、推奨設定 (.env) を出す。

  python3 doctor.py                 # 1 画面のレポート + 推奨 .env   (Windows は python doctor.py)
  python3 doctor.py --write-env     # 推奨 .env を書き込む (既存は .env.bak に退避)
  python3 doctor.py --brief         # 起動スクリプト用の要約 (数行、ネットワーク確認なし)
  python3 doctor.py --json          # 機械可読 (MCP の diagnose ツールが使う)
  python3 doctor.py --no-net        # Hugging Face への接続確認を省く

標準ライブラリだけで動く (torch 不要、Python 3.8+)。Windows / macOS / Linux (WSL2 含む) 対応。
検証用: FAKE_OS / FAKE_MACHINE / FAKE_DEVICE / FAKE_MEMORY_GB / FAKE_GPU / FAKE_WSL で環境を偽装できる。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

if sys.version_info < (3, 8):
    sys.exit("Python 3.8 以上が必要です")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from server.presets import (  # noqa: E402
    DEFAULT_PRIORITY, EXTRA_DOWNLOAD_GB, EXTRA_MODEL_IDS, PRESETS, TIER_POSSIBLE, TIER_RECOMMENDED,
    TIER_UNSUPPORTED, best_recommended, by_tier, classify_presets, default_max_loaded_models,
)

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
SPEED_TEST_URL = f"{HF_ENDPOINT}/stabilityai/sdxl-turbo/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors"
SPEED_TEST_BYTES = 8 * 1024 * 1024
SPEED_TEST_SECONDS = 6.0
REACH_URL = f"{HF_ENDPOINT}/api/models/stabilityai/sdxl-turbo"

IS_WINDOWS = sys.platform == "win32"


# ---------------------------------------------------------------- helpers
def run(cmd: list[str], timeout: float = 10) -> tuple[int, str]:
    """コマンドを実行して (returncode, 出力) を返す。無い / タイムアウトは (-1, 理由)。"""
    try:
        kw: dict = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        if IS_WINDOWS:
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = subprocess.run(cmd, **kw)
        return p.returncode, p.stdout.decode("utf-8", errors="replace").strip()
    except FileNotFoundError:
        return -1, "not found"
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:  # noqa: BLE001
        return -1, str(e)


def which(name: str, extra: tuple[str, ...] = ()) -> Optional[str]:
    p = shutil.which(name)
    if p:
        return p
    for cand in extra:
        if Path(cand).is_file():
            return cand
    return None


def gb(n: float) -> float:
    """bytes -> GiB (表示は 64GB のような慣用値に合わせる)。"""
    return round(n / 2**30, 1)


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE 形式の .env を読む (# はコメント、クォートは外す)。"""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m and not line.lstrip().startswith("#"):
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def fake(name: str) -> Optional[str]:
    v = os.environ.get(name)
    return v if v else None


# ---------------------------------------------------------------- system
def ram_gb() -> float:
    try:
        if IS_WINDOWS:
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            return gb(st.ullTotalPhys)
        if hasattr(os, "sysconf"):
            return gb(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            rc, out = run(["sysctl", "-n", "machdep.cpu.brand_string"], 3)
            if rc == 0 and out:
                return out
        elif sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    return line.split(":", 1)[1].strip()
        elif IS_WINDOWS:
            import winreg  # type: ignore[import-not-found]
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine() or "unknown"


def is_wsl() -> bool:
    if fake("FAKE_WSL"):
        return fake("FAKE_WSL") not in ("0", "false")
    if not sys.platform.startswith("linux"):
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(errors="replace").lower()
    except Exception:  # noqa: BLE001
        return False


def in_container() -> bool:
    return Path("/.dockerenv").exists() or bool(os.environ.get("KUBERNETES_SERVICE_HOST"))


def detect_system() -> dict:
    os_name = fake("FAKE_OS") or platform.system()  # Darwin / Linux / Windows
    label = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}.get(os_name, os_name)
    machine = fake("FAKE_MACHINE") or platform.machine()
    return dict(
        os=os_name, os_label=label, os_version=platform.release(), machine=machine,
        cpu=cpu_name(), cores=os.cpu_count() or 0,
        ram_gb=float(fake("FAKE_MEMORY_GB") or 0) if fake("FAKE_DEVICE") in ("cpu", "mps") else ram_gb(),
        python=platform.python_version(), python_exe=sys.executable,
        wsl=is_wsl(), container=in_container(),
        fake=any(fake(k) for k in ("FAKE_OS", "FAKE_MACHINE", "FAKE_DEVICE", "FAKE_MEMORY_GB", "FAKE_GPU", "FAKE_WSL")),
    )


# ---------------------------------------------------------------- gpu
NVIDIA_SMI_EXTRA = (
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    "/usr/lib/wsl/lib/nvidia-smi",
)


def detect_gpu(sysinfo: dict) -> dict:
    """kind: nvidia / apple / none。nvidia なら gpus=[{name, vram_gb, driver}]。"""
    if fake("FAKE_DEVICE"):
        dev = fake("FAKE_DEVICE")
        mem = float(fake("FAKE_MEMORY_GB") or 0)
        if dev == "cuda":
            return dict(kind="nvidia", name=fake("FAKE_GPU") or "NVIDIA GPU (fake)", vram_gb=mem,
                        gpus=[dict(name=fake("FAKE_GPU") or "NVIDIA GPU (fake)", vram_gb=mem, driver="fake")],
                        nvidia_smi=True, note=None)
        if dev == "mps":
            return dict(kind="apple", name=fake("FAKE_GPU") or "Apple Silicon (fake)", vram_gb=mem, gpus=[],
                        nvidia_smi=False, note=None)
        return dict(kind="none", name=None, vram_gb=0.0, gpus=[], nvidia_smi=False, note=None)

    smi = which("nvidia-smi", NVIDIA_SMI_EXTRA)
    if smi:
        rc, out = run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], 15)
        gpus = []
        if rc == 0:
            for line in out.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2:
                    try:
                        vram = round(float(parts[1]) / 1024, 1)
                    except ValueError:
                        vram = 0.0
                    gpus.append(dict(name=parts[0], vram_gb=vram, driver=parts[2] if len(parts) > 2 else None))
        if gpus:
            g = max(gpus, key=lambda x: x["vram_gb"])
            return dict(kind="nvidia", name=g["name"], vram_gb=g["vram_gb"], gpus=gpus, nvidia_smi=True, note=None)
        return dict(kind="nvidia-broken", name=None, vram_gb=0.0, gpus=[], nvidia_smi=True,
                    note=f"nvidia-smi はあるが応答が不正: {out[:120]}")

    if sysinfo["os"] == "Darwin" and sysinfo["machine"] in ("arm64", "aarch64"):
        return dict(kind="apple", name=sysinfo["cpu"], vram_gb=sysinfo["ram_gb"], gpus=[], nvidia_smi=False, note=None)

    # nvidia-smi は無いが NVIDIA デバイスらしきものがある
    note = None
    if sysinfo["os"] == "Linux":
        if Path("/proc/driver/nvidia/version").exists():
            note = "NVIDIA ドライバはあるが nvidia-smi が PATH に無い"
        else:
            rc, out = run(["lspci"], 5)
            if rc == 0 and "nvidia" in out.lower():
                note = "NVIDIA デバイスを検出したがドライバ (nvidia-smi) が無い"
    return dict(kind="none", name=None, vram_gb=0.0, gpus=[], nvidia_smi=False, note=note)


def pick_device(sysinfo: dict, gpu: dict) -> tuple[str, float, str]:
    """(device, 分類に使うメモリ GB, メモリの種類) を返す。"""
    if fake("FAKE_DEVICE"):
        dev = fake("FAKE_DEVICE")
        mem = float(fake("FAKE_MEMORY_GB") or 0)
        return dev, mem, "vram" if dev == "cuda" else "ram"
    if gpu["kind"] == "nvidia" and sysinfo["os"] != "Darwin":
        return "cuda", gpu["vram_gb"], "vram"
    if gpu["kind"] == "apple":
        return "mps", sysinfo["ram_gb"], "ram"
    return "cpu", sysinfo["ram_gb"], "ram"


# ---------------------------------------------------------------- tools
def detect_tools(sysinfo: dict, gpu: dict) -> dict:
    """NVIDIA Container Toolkit は NVIDIA GPU がある環境でだけ判定し、それ以外は該当なし (applicable=False)。"""
    tools: dict[str, dict] = {}
    nvidia = gpu["kind"] == "nvidia" and sysinfo["os"] != "Darwin"

    def add(name: str, found: bool, version: Optional[str] = None, note: Optional[str] = None) -> None:
        tools[name] = dict(found=found, version=version, note=note)

    rc, out = run(["uv", "--version"], 5)
    add("uv", rc == 0, out.split("\n")[0] if rc == 0 else None)

    add("python", True, sysinfo["python"],
        None if tuple(int(x) for x in sysinfo["python"].split(".")[:2]) >= (3, 11)
        else "サーバーは 3.11+ が必要だが uv が自動で取得するので問題なし (doctor / gen.py はこのままで可)")

    rc, out = run(["docker", "--version"], 5)
    docker_found = rc == 0
    add("docker", docker_found, out if docker_found else None)
    daemon_ok = False
    runtimes = ""
    if docker_found:
        rc, out = run(["docker", "info", "--format", "{{json .Runtimes}}"], 15)
        daemon_ok = rc == 0 and out.startswith("{")
        runtimes = out if daemon_ok else ""
        if not daemon_ok:
            tools["docker"]["note"] = "docker コマンドはあるがデーモンに接続できない (Docker Desktop を起動する)"
        rc, out = run(["docker", "compose", "version"], 10)
        add("compose", rc == 0, out if rc == 0 else None,
            None if rc == 0 else "compose v2 (docker compose) が無い")
        if nvidia:
            add("nvidia-container-toolkit", "nvidia" in runtimes.lower(),
                None, None if daemon_ok else "docker デーモン未接続のため未確認")
        else:
            add("nvidia-container-toolkit", False, None, None)
        if nvidia and not tools["nvidia-container-toolkit"]["found"] and daemon_ok:
            rc, out = run(["nvidia-ctk", "--version"], 5)
            if rc == 0:
                tools["nvidia-container-toolkit"].update(
                    found=True, version=out.split("\n")[0],
                    note="nvidia-ctk はあるが docker の Runtimes に nvidia が無い (sudo nvidia-ctk runtime configure --runtime=docker 後に docker 再起動)")
    else:
        add("compose", False, None, "docker が無い")
        add("nvidia-container-toolkit", False, None, "docker が無い" if nvidia else None)
    tools["docker"]["daemon"] = daemon_ok
    tools["nvidia-container-toolkit"]["applicable"] = nvidia
    return tools


# ---------------------------------------------------------------- disk / cache / net / server
def models_dir(envfile: dict) -> Path:
    hf_home = os.environ.get("HF_HOME") or envfile.get("HF_HOME")
    return Path(hf_home).expanduser() if hf_home else ROOT / "models"


def detect_disk(mdir: Path) -> dict:
    target = mdir if mdir.exists() else (mdir.parent if mdir.parent.exists() else ROOT)
    try:
        u = shutil.disk_usage(str(target))
        return dict(path=str(mdir), free_gb=gb(u.free), total_gb=gb(u.total), ok=True)
    except Exception as e:  # noqa: BLE001
        return dict(path=str(mdir), free_gb=0.0, total_gb=0.0, ok=False, error=str(e))


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def detect_cache(mdir: Path) -> dict:
    """HF キャッシュ (models/hub) と rembg のモデルを見る。"""
    hub = mdir / "hub"
    found: dict[str, dict] = {}

    def look(label: str, model_id: str) -> None:
        d = hub / ("models--" + model_id.replace("/", "--"))
        if not d.is_dir():
            return
        blobs = d / "blobs"
        size = _dir_size(blobs) if blobs.is_dir() else 0
        if size == 0:  # Windows で symlink が使えない場合は snapshots に実体がある
            size = _dir_size(d / "snapshots")
        # snapshots/ 配下のファイル (symlink なら参照先) が全部揃っていれば完了。
        # ダウンロード中断の残骸 (blobs/*.incomplete) があっても、揃っていれば使える
        snap = d / "snapshots"
        files = [p for p in snap.rglob("*") if p.is_file() or p.is_symlink()] if snap.is_dir() else []
        complete = bool(files) and all(p.exists() for p in files)
        found[label] = dict(model_id=model_id, size_gb=gb(size), partial=not complete, complete=complete,
                            leftovers=any(blobs.glob("*.incomplete")) if blobs.is_dir() else False)

    for name, p in PRESETS.items():
        look(name, p["model_id"])
    for label, mid in EXTRA_MODEL_IDS.items():
        look(label, mid)
    rembg_dir = mdir / "rembg"
    onnx = list(rembg_dir.glob("*.onnx")) if rembg_dir.is_dir() else []
    if onnx:
        found["rembg"] = dict(model_id=", ".join(p.name for p in onnx), size_gb=gb(sum(p.stat().st_size for p in onnx)),
                              partial=False, complete=True)
    return dict(dir=str(hub), models=found, total_gb=round(sum(m["size_gb"] for m in found.values()), 1))


def check_net(speed: bool = True) -> dict:
    out: dict = dict(endpoint=HF_ENDPOINT, reachable=False, latency_s=None, speed_mbps=None, error=None,
                     hf_token=bool(os.environ.get("HF_TOKEN")))
    headers = {"User-Agent": "local-imagegen-doctor"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(REACH_URL, headers=headers), timeout=6) as r:
            r.read(2048)
        out["reachable"] = True
        out["latency_s"] = round(time.time() - t0, 2)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    if not speed:
        return out
    try:
        req = urllib.request.Request(SPEED_TEST_URL, headers=dict(headers, Range=f"bytes=0-{SPEED_TEST_BYTES - 1}"))
        got = 0
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=10) as r:
            first = r.read(64 * 1024)  # 最初のチャンクまでの待ち (接続・リダイレクト) は速度に含めない
            t0 = time.time()
            while first and got < SPEED_TEST_BYTES and time.time() - t0 < SPEED_TEST_SECONDS:
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                got += len(chunk)
        dt = max(time.time() - t0, 0.05)
        out["speed_mbps"] = round(got / dt / 1e6, 1)  # MB/s
        out["speed_sample_mb"] = round(got / 1e6, 1)
    except Exception as e:  # noqa: BLE001
        out["speed_error"] = f"{type(e).__name__}: {e}"
    return out


def check_server(port: int) -> dict:
    url = f"http://localhost:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            h = json.load(r)
        return dict(url=url, running=True, ok=h.get("ok"), device=h.get("device"), preset=h.get("preset"),
                    loaded=list(h.get("loaded_presets", {})), memory_gb=h.get("memory_gb"),
                    environment=h.get("environment"))
    except Exception as e:  # noqa: BLE001
        return dict(url=url, running=False, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- recommendations
INSTALL = {
    "uv": {
        "Windows": 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
        "Darwin": "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "Linux": "curl -LsSf https://astral.sh/uv/install.sh | sh",
    },
    "docker": {
        "Windows": "Docker Desktop (WSL2 バックエンド): https://docs.docker.com/desktop/setup/install/windows-install/",
        "Darwin": "brew install --cask docker  (または https://docs.docker.com/desktop/setup/install/mac-install/)",
        "Linux": "curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER",
    },
    "nvidia-container-toolkit": {
        "Linux": ("curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && "
                  "curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | "
                  "sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
                  "sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && "
                  "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit && "
                  "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"),
        "Windows": "Docker Desktop (WSL2) なら Windows 用 NVIDIA ドライバだけで GPU が使える (toolkit の個別導入は不要)",
    },
    "nvidia-driver": {
        "Windows": "https://www.nvidia.com/drivers (WSL2 で使う場合も Windows 側のドライバだけで良い)",
        "Linux": "sudo ubuntu-drivers install  (Ubuntu) / ディストリのドライバパッケージ",
    },
}

LAUNCH = {
    "run.sh": "./run.sh",
    "run.py": "python run.py",
    "compose.gpu": "docker compose -f compose.yml -f compose.gpu.yml up -d --build",
    "compose.cpu": "docker compose up -d --build",
}


def recommend(sysinfo: dict, gpu: dict, device: str, mem: float, tools: dict, disk: dict, cache: dict,
              net: Optional[dict], envfile: dict, port: int) -> dict:
    tiers = classify_presets(device, mem)
    grouped = by_tier(tiers)
    preset = best_recommended(tiers) or "sdxl-turbo"
    max_loaded = default_max_loaded_models(mem)
    allowed = [n for n in PRESETS if tiers[n]["tier"] != TIER_UNSUPPORTED]
    warnings: list[str] = []
    actions: list[str] = []  # 不足ツールの導入など
    osn = sysinfo["os"]
    has = lambda t: bool(tools.get(t, {}).get("found"))  # noqa: E731
    docker_ready = has("docker") and tools["docker"].get("daemon") and has("compose")

    # ---- 起動経路
    if sysinfo["container"]:
        launch, why = None, "コンテナ内で実行中"
    elif osn == "Darwin":
        launch, why = "run.sh", "macOS の Docker は GPU (Metal) を使えないのでネイティブ起動"
        if not has("uv"):
            actions.append("uv を入れる: " + INSTALL["uv"]["Darwin"])
        if gpu["kind"] != "apple":
            warnings.append("Intel Mac は GPU 加速なし (CPU 生成)。軽いプリセットのみ実用")
    elif osn == "Windows":
        if gpu["kind"] == "nvidia":
            launch, why = "run.py", "NVIDIA GPU を CUDA で使う。uv だけで動く (Docker Desktop + compose.gpu でも可)"
        else:
            launch, why = "run.py", "GPU なし。CPU 生成 (docker compose up でも同じ)"
            if gpu["kind"] == "none" and gpu.get("note"):
                actions.append(gpu["note"] + ": " + INSTALL["nvidia-driver"]["Windows"])
        if not has("uv"):
            actions.append("uv を入れる: " + INSTALL["uv"]["Windows"])
    else:  # Linux / WSL2
        if gpu["kind"] == "nvidia":
            if docker_ready and has("nvidia-container-toolkit"):
                launch, why = "compose.gpu", "Docker + NVIDIA Container Toolkit が揃っている"
            elif has("uv"):
                launch, why = "run.sh", "ネイティブで CUDA を使う (Docker で使うなら NVIDIA Container Toolkit が必要)"
                if docker_ready:
                    actions.append("Docker で GPU を使うなら: " + INSTALL["nvidia-container-toolkit"]["Linux"])
            else:
                launch, why = "compose.gpu", "NVIDIA GPU あり。Docker + Toolkit で起動 (uv を入れれば run.sh でも可)"
                if not docker_ready:
                    actions.append("docker を入れる: " + INSTALL["docker"]["Linux"])
                if not has("nvidia-container-toolkit"):
                    actions.append("NVIDIA Container Toolkit を入れる: " + INSTALL["nvidia-container-toolkit"]["Linux"])
                actions.append("または uv を入れて ./run.sh: " + INSTALL["uv"]["Linux"])
        else:
            if gpu.get("note"):
                warnings.append(gpu["note"] + " -> " + INSTALL["nvidia-driver"]["Linux"])
            if has("uv"):
                launch, why = "run.sh", "GPU なし。CPU 生成をネイティブで"
            elif docker_ready:
                launch, why = "compose.cpu", "GPU なし。CPU 生成を Docker で"
            else:
                launch, why = "run.sh", "GPU なし。uv を入れてネイティブ起動 (または docker compose up)"
                actions.append("uv を入れる: " + INSTALL["uv"]["Linux"])
        if sysinfo["wsl"] and gpu["kind"] != "nvidia":
            warnings.append("WSL2 で GPU を使うには Windows 側に NVIDIA ドライバを入れる (nvidia-smi が WSL 内で動けば OK)")

    # ---- プリセット / .env
    cur_preset = envfile.get("PRESET")
    if cur_preset and cur_preset in tiers and tiers[cur_preset]["tier"] == TIER_UNSUPPORTED:
        warnings.append(f".env の PRESET={cur_preset} はこの環境では unsupported ({tiers[cur_preset]['reason']})。"
                        f"推奨は {preset}")
    if device == "cuda" and tiers[preset]["tier"] == TIER_POSSIBLE:
        warnings.append(f"{preset} は VRAM ぎりぎり。CPU_OFFLOAD=1 を推奨")
    if device == "cpu":
        warnings.append("GPU なし (CPU 生成)。sdxl-turbo 512px で 1 枚 1〜2 分が目安。anime 等の SDXL 系は非常に遅い")
    if not grouped[TIER_RECOMMENDED]:
        warnings.append("快適に動くプリセットがない。メモリ的に厳しいので小さい解像度 / CPU_OFFLOAD を使う")

    # ---- ダウンロード見込み
    need_gb = 0.0
    need_items: list[str] = []
    cached = cache["models"]
    if not cached.get(preset, {}).get("complete"):
        need_gb += PRESETS[preset]["download_gb"]
        need_items.append(f"{preset} {PRESETS[preset]['download_gb']}GB")
    if preset in ("anime", "sdxl", "sdxl-turbo", "sd15") and not cached.get("ip-adapter", {}).get("complete"):
        need_items.append(f"IP-Adapter {EXTRA_DOWNLOAD_GB['ip-adapter']}GB (--ref 初回のみ)")
    if not cached.get("rembg"):
        need_items.append(f"rembg {EXTRA_DOWNLOAD_GB['rembg']}GB (--rembg 初回のみ)")
    dl_minutes = None
    if net and net.get("speed_mbps"):
        dl_minutes = round(need_gb * 1000 / net["speed_mbps"] / 60, 1)
    if disk["ok"] and disk["free_gb"] < need_gb + 10:
        warnings.append(f"ディスク空き {disk['free_gb']}GB は不足気味 (初回 DL {need_gb:.0f}GB + 余裕 10GB)")
    if net is not None and not net.get("reachable") and need_gb > 0:
        warnings.append(f"{HF_ENDPOINT} に到達できない (モデル DL が必要: {need_gb:.0f}GB)。プロキシ / HF_ENDPOINT を確認")

    # ---- .env テキスト
    stamp = time.strftime("%Y-%m-%d %H:%M")
    envlines = [
        f"# doctor.py が生成 ({stamp}): {sysinfo['os_label']} {sysinfo['machine']} / device={device} / "
        f"{'VRAM' if device == 'cuda' else 'RAM'} {mem:.0f}GB",
        f"# 推奨: {', '.join(grouped[TIER_RECOMMENDED]) or '-'} / 可: {', '.join(grouped[TIER_POSSIBLE]) or '-'} / "
        f"不可: {', '.join(grouped[TIER_UNSUPPORTED]) or '-'}",
        f"PRESET={preset}",
        f"MAX_LOADED_MODELS={max_loaded}",
        f"ALLOWED_PRESETS={','.join(allowed)}",
        f"PORT={port}",
    ]
    if device == "cuda":
        envlines.append(("CPU_OFFLOAD=1" if tiers[preset]["tier"] == TIER_POSSIBLE else "#CPU_OFFLOAD=1")
                        + "  # VRAM が足りないときに有効化 (CUDA のみ)")
    envlines.append("#HF_TOKEN=hf_xxx  # gated モデル (FLUX.1-dev など) を使うときだけ")
    env_text = "\n".join(envlines) + "\n"

    return dict(
        launch=launch, launch_command=LAUNCH.get(launch) if launch else None, launch_reason=why,
        preset=preset, max_loaded_models=max_loaded, allowed_presets=allowed,
        tiers=tiers, by_tier=grouped,
        download_gb=round(need_gb, 1), download_items=need_items, download_minutes=dl_minutes,
        warnings=warnings, actions=actions, env=env_text,
    )


# ---------------------------------------------------------------- diagnose (all-in-one)
def diagnose(no_net: bool = False, speed: bool = True, port: Optional[int] = None,
             env_path: Optional[Path] = None) -> dict:
    env_path = env_path or ROOT / ".env"
    envfile = read_env_file(env_path)
    port = port or int(os.environ.get("PORT") or envfile.get("PORT") or 8000)
    sysinfo = detect_system()
    gpu = detect_gpu(sysinfo)
    device, mem, mem_kind = pick_device(sysinfo, gpu)
    tools = detect_tools(sysinfo, gpu)
    mdir = models_dir(envfile)
    disk = detect_disk(mdir)
    cache = detect_cache(mdir)
    net = None if no_net else check_net(speed=speed)
    server = check_server(port)
    rec = recommend(sysinfo, gpu, device, mem, tools, disk, cache, net, envfile, port)
    return dict(
        version=1, generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"), root=str(ROOT),
        system=sysinfo, gpu=gpu, device=device, memory_gb=mem, memory_kind=mem_kind,
        tools=tools, disk=disk, cache=cache, net=net, server=server,
        env_file=dict(path=str(env_path), exists=env_path.is_file(), values=envfile),
        recommendation=rec,
    )


# ---------------------------------------------------------------- rendering
OK, NG, NA = "[OK]", "[!!]", "[n/a]"
TIER_LABEL = {TIER_RECOMMENDED: "[推奨]", TIER_POSSIBLE: "[可]  ", TIER_UNSUPPORTED: "[不可]"}


def render_report(d: dict) -> str:
    s, g, t, disk, cache, net, srv, rec = (d["system"], d["gpu"], d["tools"], d["disk"], d["cache"], d["net"],
                                           d["server"], d["recommendation"])
    L: list[str] = []
    fake_note = " (FAKE_* による偽装あり)" if s["fake"] else ""
    L.append(f"== Local ImageGen 環境診断 {d['generated_at'][:16].replace('T', ' ')}{fake_note} ==")
    wsl = " / WSL2" if s["wsl"] else ""
    cont = " / コンテナ内" if s["container"] else ""
    L.append(f"システム   {s['os_label']} {s['os_version']} ({s['machine']}){wsl}{cont}  CPU: {s['cpu']} ({s['cores']} cores)  "
             f"RAM: {s['ram_gb']:.0f}GB  Python {s['python']}")
    if g["kind"] == "nvidia":
        names = ", ".join(f"{x['name']} {x['vram_gb']:.0f}GB" for x in g["gpus"])
        drv = g["gpus"][0].get("driver") if g["gpus"] else None
        L.append(f"GPU        NVIDIA: {names}" + (f" (driver {drv})" if drv else "") + f" -> device={d['device']}")
    elif g["kind"] == "apple":
        L.append(f"GPU        Apple Silicon (統合メモリ {g['vram_gb']:.0f}GB) -> device={d['device']}")
    else:
        L.append(f"GPU        なし -> device={d['device']}" + (f"  ({g['note']})" if g.get("note") else ""))

    def tool(name: str, label: str) -> str:
        x = t.get(name, {})
        if x.get("applicable") is False:
            return f"{label} {NA}"
        mark = OK if x.get("found") else NG
        ver = (x.get("version") or "").replace("Docker version ", "").replace("Docker Compose version ", "")
        ver = ver.split(",")[0].split(" (")[0]
        if ver.startswith(name + " "):
            ver = ver[len(name) + 1:]
        return f"{label} {mark}" + (f" {ver}" if ver and x.get("found") else "")

    L.append("ツール     " + "  ".join([tool("uv", "uv"), tool("python", "python"), tool("docker", "docker"),
                                        tool("compose", "compose v2"), tool("nvidia-container-toolkit", "NVIDIA toolkit")]))
    notes = [f"{k}: {v['note']}" for k, v in t.items() if v.get("note")]
    for n in notes:
        L.append(f"           - {n}")
    L.append(f"ディスク   {disk['path']} の空き {disk['free_gb']:.0f}GB / {disk['total_gb']:.0f}GB "
             + (OK if disk["free_gb"] >= 20 else NG))
    if net is None:
        L.append(f"ネット     (--no-net のため未確認)")
    elif net["reachable"]:
        sp = f"  DL 速度 約 {net['speed_mbps']}MB/s ({net.get('speed_sample_mb', 0)}MB で計測)" if net.get("speed_mbps") else ""
        L.append(f"ネット     {net['endpoint']} {OK} {net['latency_s']}s{sp}" + ("  HF_TOKEN あり" if net["hf_token"] else ""))
    else:
        L.append(f"ネット     {net['endpoint']} {NG} {net['error']}")
    if cache["models"]:
        items = ", ".join(f"{k} {v['size_gb']}GB" + (" (未完)" if v["partial"] else "") for k, v in cache["models"].items())
        L.append(f"キャッシュ {items}  (計 {cache['total_gb']}GB, {cache['dir']})")
    else:
        L.append(f"キャッシュ なし ({cache['dir']})")
    if srv["running"]:
        L.append(f"サーバー   {srv['url']} 起動中 (device={srv.get('device')}, preset={srv.get('preset')}, "
                 f"loaded: {', '.join(srv.get('loaded') or []) or '-'})")
    else:
        L.append(f"サーバー   {srv['url']} 未起動")

    L.append("")
    L.append(f"プリセット (device={d['device']}, {'VRAM' if d['memory_kind'] == 'vram' else 'メモリ'} {d['memory_gb']:.0f}GB)")
    for name in PRESETS:
        tier = rec["tiers"][name]
        c = cache["models"].get(name)
        cached = f"  [cached {c['size_gb']}GB]" if c else ""
        L.append(f"  {TIER_LABEL[tier['tier']]} {name:<13} {tier['reason']}{cached}")

    L.append("")
    L.append("推奨")
    if rec["launch_command"]:
        L.append(f"  起動:   {rec['launch_command']}    ({rec['launch_reason']})")
    else:
        L.append(f"  起動:   {rec['launch_reason']}")
    L.append(f"  設定:   PRESET={rec['preset']}  MAX_LOADED_MODELS={rec['max_loaded_models']}  "
             f"ALLOWED_PRESETS={','.join(rec['allowed_presets'])}")
    if rec["download_gb"] > 0:
        mins = f"、約 {rec['download_minutes']} 分" if rec["download_minutes"] is not None else ""
        L.append(f"  初回 DL: 約 {rec['download_gb']}GB ({', '.join(rec['download_items'])}{mins})")
    else:
        extra = f" (別途: {', '.join(rec['download_items'])})" if rec["download_items"] else ""
        L.append(f"  初回 DL: 不要 ({rec['preset']} はキャッシュ済み){extra}")
    for w in rec["warnings"]:
        L.append(f"  注意:   {w}")
    for a in rec["actions"]:
        L.append(f"  要対応: {a}")

    L.append("")
    py = "python" if s["os"] == "Windows" else "python3"
    L.append(f"推奨 .env ({py} doctor.py --write-env で書き込み。既存は .env.bak に退避):")
    for line in rec["env"].rstrip("\n").splitlines():
        L.append("  " + line)
    return "\n".join(L)


def render_brief(d: dict) -> str:
    s, g, rec, cache = d["system"], d["gpu"], d["recommendation"], d["cache"]
    gpu = (f"NVIDIA {g['name']} {g['vram_gb']:.0f}GB" if g["kind"] == "nvidia"
           else f"Apple Silicon 統合メモリ {g['vram_gb']:.0f}GB" if g["kind"] == "apple" else "GPU なし")
    grouped = rec["by_tier"]
    L = [
        f"[doctor] {s['os_label']} {s['machine']}{' (WSL2)' if s['wsl'] else ''} / {s['cpu']} / RAM {s['ram_gb']:.0f}GB / "
        f"{gpu} -> device={d['device']}",
        f"[doctor] 推奨: {rec['launch_command'] or rec['launch_reason']}  PRESET={rec['preset']}  "
        f"MAX_LOADED_MODELS={rec['max_loaded_models']}",
        f"[doctor] プリセット 推奨: {', '.join(grouped[TIER_RECOMMENDED]) or '-'} / 可: "
        f"{', '.join(grouped[TIER_POSSIBLE]) or '-'} / 不可: {', '.join(grouped[TIER_UNSUPPORTED]) or '-'}",
    ]
    if rec["download_gb"] > 0:
        L.append(f"[doctor] 初回 DL 約 {rec['download_gb']}GB ({', '.join(rec['download_items'])})")
    for w in rec["warnings"]:
        L.append(f"[doctor] 注意: {w}")
    py = "python" if s["os"] == "Windows" else "python3"
    if rec["actions"]:
        L.append(f"[doctor] 要対応 {len(rec['actions'])} 件 (詳細: {py} doctor.py): " + rec["actions"][0][:90])
    else:
        L.append(f"[doctor] 詳細: {py} doctor.py")
    return "\n".join(L)


# ---------------------------------------------------------------- .env writing
MANAGED_KEYS = ("PRESET", "MAX_LOADED_MODELS", "ALLOWED_PRESETS", "PORT", "CPU_OFFLOAD", "HF_TOKEN")


def write_env(d: dict, env_path: Path) -> dict:
    """推奨 .env を書く。既存ファイルは .env.bak に退避し、推奨に含まれないキーは引き継ぐ。"""
    rec = d["recommendation"]
    existing = read_env_file(env_path)
    text = rec["env"]
    carried = {k: v for k, v in existing.items() if k not in MANAGED_KEYS}
    if existing.get("HF_TOKEN"):
        text = text.replace("#HF_TOKEN=hf_xxx  # gated モデル (FLUX.1-dev など) を使うときだけ", f"HF_TOKEN={existing['HF_TOKEN']}")
    if carried:
        text += "# 既存の .env から引き継ぎ\n" + "".join(f"{k}={v}\n" for k, v in carried.items())
    backup = None
    if env_path.is_file():
        backup = env_path.with_name(env_path.name + ".bak")
        shutil.copy2(str(env_path), str(backup))
    env_path.write_text(text, encoding="utf-8")
    return dict(path=str(env_path), backup=str(backup) if backup else None, carried=sorted(carried))


# ---------------------------------------------------------------- main
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Local ImageGen の環境診断と推奨設定")
    ap.add_argument("--json", action="store_true", help="JSON で出力 (report / env テキストも含む)")
    ap.add_argument("--brief", action="store_true", help="起動時用の数行要約 (ネットワーク確認は省く)")
    ap.add_argument("--write-env", action="store_true", help="推奨 .env を書き込む (既存は .env.bak に退避)")
    ap.add_argument("--no-net", action="store_true", help="Hugging Face への接続・速度確認を省く")
    ap.add_argument("--no-speed", action="store_true", help="到達確認だけして速度計測は省く")
    ap.add_argument("--port", type=int, default=None, help="サーバーのポート (既定: .env の PORT か 8000)")
    ap.add_argument("--env", dest="env_path", default=None, help=".env のパス (既定: このファイルと同じ場所)")
    a = ap.parse_args(argv)

    try:  # Windows の cp932 コンソールでも落ちないように
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    env_path = Path(a.env_path).expanduser() if a.env_path else ROOT / ".env"
    no_net = a.no_net or a.brief
    d = diagnose(no_net=no_net, speed=not a.no_speed, port=a.port, env_path=env_path)
    if a.write_env:
        d["written"] = write_env(d, env_path)
    if a.json:
        d["report"] = render_report(d)
        d["brief"] = render_brief(d)
        print(json.dumps(d, ensure_ascii=False, indent=1))
    elif a.brief:
        print(render_brief(d))
    else:
        print(render_report(d))
        if d.get("written"):
            w = d["written"]
            print(f"\n.env を書き込みました: {w['path']}" + (f" (旧ファイル: {w['backup']})" if w["backup"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
