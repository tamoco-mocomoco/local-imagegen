"""
環境診断と環境別プリセット分類のテスト。

  .venv/bin/python -m unittest discover -s tests -t .      (pytest でも可)

サーバーのテストは FAKE_DEVICE / FAKE_MEMORY_GB で環境を偽装し、PRELOAD=0 でモデルは読まない (torch の import はする)。
doctor.py のテストは subprocess で --json を実行し、.env は一時ファイルを指す (リポジトリの .env は触らない)。
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import presets as P  # noqa: E402

FAKE_KEYS = ("FAKE_OS", "FAKE_MACHINE", "FAKE_DEVICE", "FAKE_MEMORY_GB", "FAKE_GPU", "FAKE_WSL",
             "PRESET", "ALLOWED_PRESETS", "MAX_LOADED_MODELS", "MODEL_ID", "DEVICE", "PRELOAD", "OUTPUT_DIR")


def clean_env(**over: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in FAKE_KEYS}
    env["IMAGEGEN_DOTENV"] = ""  # リポジトリの .env (PRESET=anime など) をテストに持ち込まない
    env.update(over)
    return env


# ---------------------------------------------------------------- presets.py (pure)
class ClassifyTests(unittest.TestCase):
    def tiers(self, device: str, mem: float) -> dict:
        return {k: v["tier"] for k, v in P.classify_presets(device, mem).items()}

    def test_cuda_8gb(self):
        t = self.tiers("cuda", 8)
        self.assertEqual(t["sdxl-turbo"], "recommended")
        self.assertEqual(t["sd15"], "recommended")
        self.assertEqual(t["anime"], "unsupported")
        self.assertEqual(t["flux-schnell"], "unsupported")

    def test_cuda_10gb_is_possible_for_sdxl(self):
        t = self.tiers("cuda", 10)  # 12 * 0.7 = 8.4 <= 10 < 12
        self.assertEqual(t["anime"], "possible")
        self.assertEqual(t["sdxl"], "possible")
        self.assertIn("CPU_OFFLOAD", P.classify_presets("cuda", 10)["anime"]["reason"])

    def test_cuda_12_and_24gb(self):
        self.assertEqual(self.tiers("cuda", 12)["anime"], "recommended")
        self.assertEqual(self.tiers("cuda", 12)["flux-schnell"], "unsupported")
        self.assertEqual(self.tiers("cuda", 24)["flux-schnell"], "recommended")

    def test_mps_16gb(self):
        t = self.tiers("mps", 16)
        self.assertEqual(t["anime"], "recommended")
        self.assertEqual(t["zimage-turbo"], "possible")  # 20 * 0.7 = 14
        self.assertEqual(t["flux-schnell"], "unsupported")  # 24 * 0.7 = 16.8

    def test_mps_64gb_all_recommended(self):
        self.assertTrue(all(v == "recommended" for v in self.tiers("mps", 64).values()))

    def test_cpu(self):
        t = self.tiers("cpu", 16)
        self.assertEqual(t["sdxl-turbo"], "recommended")
        self.assertEqual(t["anime"], "unsupported")
        t = self.tiers("cpu", 64)
        self.assertEqual(t["anime"], "possible")  # 動くが非常に遅い
        t = self.tiers("cpu", 6)
        self.assertEqual(t["sdxl-turbo"], "possible")  # 8 * 0.7 = 5.6
        self.assertEqual(t["sd15"], "recommended")

    def test_unknown_memory(self):
        self.assertEqual(self.tiers("cuda", 0)["anime"], "possible")

    def test_custom_model_without_requirements(self):
        tier, reason = P.classify_preset({"model_id": "x/y"}, "cuda", 4)
        self.assertEqual(tier, "possible")

    def test_best_recommended_priority(self):
        self.assertEqual(P.best_recommended(P.classify_presets("mps", 64)), "anime")
        self.assertEqual(P.best_recommended(P.classify_presets("cuda", 8)), "sdxl-turbo")
        self.assertEqual(P.best_recommended(P.classify_presets("cuda", 5)), "sd15")  # sdxl-turbo は 5.6 未満

    def test_max_loaded(self):
        self.assertEqual(P.default_max_loaded_models(64), 2)
        self.assertEqual(P.default_max_loaded_models(31.9), 1)


# ---------------------------------------------------------------- server/app.py (TestClient)
def load_app(**env: str):
    """環境変数を差し替えて server.app を読み直し、TestClient を返す。"""
    from fastapi.testclient import TestClient
    env.setdefault("PRELOAD", "0")
    env.setdefault("OUTPUT_DIR", tempfile.mkdtemp(prefix="imagegen-test-"))
    with mock.patch.dict(os.environ, clean_env(**env), clear=True):
        sys.modules.pop("server.app", None)
        importlib.reload(P)  # MODEL_ID などで PRESETS を書き換えた状態を引きずらない
        app_mod = importlib.import_module("server.app")
    return TestClient(app_mod.app)


class ServerTierTests(unittest.TestCase):
    def test_windows_nvidia_8gb(self):
        """Windows + 8GB GPU 相当: 既定 sdxl-turbo のまま、ALLOWED は unsupported 以外、常駐 1。"""
        c = load_app(FAKE_DEVICE="cuda", FAKE_MEMORY_GB="8")
        h = c.get("/health").json()
        self.assertEqual(h["device"], "cuda")
        self.assertEqual(h["memory_gb"], 8.0)
        self.assertEqual(h["preset"], "sdxl-turbo")
        self.assertEqual(h["allowed_presets"], ["sdxl-turbo", "sd15"])
        self.assertEqual(h["max_loaded_models"], 1)
        env = h["environment"]
        self.assertEqual(env["unsupported"], ["sdxl", "anime", "flux-schnell", "zimage-turbo"])
        self.assertTrue(env["fake"])
        p = c.get("/presets").json()
        self.assertEqual(p["anime"]["tier"], "unsupported")
        self.assertFalse(p["anime"]["allowed"])
        self.assertIn("VRAM", p["anime"]["reason"])
        self.assertEqual(p["sdxl-turbo"]["tier"], "recommended")
        self.assertTrue(p["sdxl-turbo"]["allowed"])
        r = c.post("/load", json={"preset": "anime"})
        self.assertEqual(r.status_code, 400)

    def test_default_switches_when_unsupported(self):
        """PRESET 未指定で既定が unsupported なら最良の recommended に切り替える。"""
        c = load_app(FAKE_DEVICE="cuda", FAKE_MEMORY_GB="5")
        h = c.get("/health").json()
        self.assertEqual(h["preset"], "sd15")
        self.assertEqual(h["environment"]["default_preset_switched_from"], "sdxl-turbo")

    def test_explicit_preset_is_kept_on_cpu(self):
        """CPU 16GB で PRESET=anime を明示: 警告のみで尊重し、allowed にも入る。"""
        c = load_app(FAKE_DEVICE="cpu", FAKE_MEMORY_GB="16", PRESET="anime")
        h = c.get("/health").json()
        self.assertEqual(h["preset"], "anime")
        self.assertEqual(h["environment"]["default_preset_tier"], "unsupported")
        self.assertIsNone(h["environment"]["default_preset_switched_from"])
        self.assertEqual(h["allowed_presets"][0], "anime")
        self.assertIn("sdxl-turbo", h["allowed_presets"])
        self.assertNotIn("flux-schnell", h["allowed_presets"])

    def test_allowed_presets_env_wins(self):
        c = load_app(FAKE_DEVICE="cuda", FAKE_MEMORY_GB="24", ALLOWED_PRESETS="sdxl-turbo,flux-schnell")
        h = c.get("/health").json()
        self.assertEqual(h["allowed_presets"], ["sdxl-turbo", "flux-schnell"])
        self.assertTrue(h["environment"]["allowed_presets_from_env"])

    def test_mac_64gb_unchanged(self):
        """この Mac の既存挙動: anime 既定、全プリセット許可、常駐 2。"""
        c = load_app(FAKE_DEVICE="mps", FAKE_MEMORY_GB="64", PRESET="anime")
        h = c.get("/health").json()
        self.assertEqual(h["preset"], "anime")
        self.assertEqual(h["max_loaded_models"], 2)
        self.assertEqual(h["allowed_presets"], list(P.PRESETS))
        self.assertEqual(h["environment"]["unsupported"], [])
        self.assertTrue(all(v["tier"] == "recommended" for v in c.get("/presets").json().values()))


# ---------------------------------------------------------------- doctor.py (subprocess, --json)
def run_doctor(*args: str, **env: str) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="imagegen-doctor-"))
    env_path = tmp / ".env"
    cmd = [sys.executable, str(ROOT / "doctor.py"), "--json", "--no-net", "--port", "1", "--env", str(env_path), *args]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, timeout=120,
                       env=dict(clean_env(**env), PYTHONIOENCODING="utf-8"))
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    d = json.loads(p.stdout.decode("utf-8"))
    d["_env_path"] = env_path
    return d


class DoctorTests(unittest.TestCase):
    def test_windows_nvidia_8gb(self):
        d = run_doctor(FAKE_OS="Windows", FAKE_MACHINE="AMD64", FAKE_DEVICE="cuda", FAKE_MEMORY_GB="8",
                       FAKE_GPU="NVIDIA GeForce RTX 3070")
        r = d["recommendation"]
        self.assertEqual(d["system"]["os"], "Windows")
        self.assertEqual(d["device"], "cuda")
        self.assertEqual(d["gpu"]["name"], "NVIDIA GeForce RTX 3070")
        self.assertEqual(r["launch"], "run.py")
        self.assertEqual(r["launch_command"], "python run.py")
        self.assertEqual(r["preset"], "sdxl-turbo")
        self.assertEqual(r["max_loaded_models"], 1)
        self.assertEqual(r["allowed_presets"], ["sdxl-turbo", "sd15"])
        self.assertEqual(r["by_tier"]["unsupported"], ["sdxl", "anime", "flux-schnell", "zimage-turbo"])
        self.assertIn("PRESET=sdxl-turbo\n", r["env"])
        self.assertIn("ALLOWED_PRESETS=sdxl-turbo,sd15\n", r["env"])
        self.assertIn("python doctor.py", d["report"])  # Windows では python3 ではなく python
        self.assertFalse(d["server"]["running"])
        self.assertIsNone(d["net"])
        self.assertFalse(d["_env_path"].exists())  # --write-env なしでは書かない

    def test_cpu_only(self):
        d = run_doctor(FAKE_OS="Linux", FAKE_DEVICE="cpu", FAKE_MEMORY_GB="16")
        r = d["recommendation"]
        self.assertEqual(d["device"], "cpu")
        self.assertIn(r["launch"], ("run.sh", "compose.cpu"))
        self.assertEqual(r["preset"], "sdxl-turbo")
        self.assertEqual(r["tiers"]["anime"]["tier"], "unsupported")
        self.assertTrue(any("CPU 生成" in w for w in r["warnings"]))

    def test_mac_apple_silicon(self):
        d = run_doctor(FAKE_OS="Darwin", FAKE_MACHINE="arm64", FAKE_DEVICE="mps", FAKE_MEMORY_GB="64")
        r = d["recommendation"]
        self.assertEqual(r["launch"], "run.sh")
        self.assertEqual(r["preset"], "anime")
        self.assertEqual(r["max_loaded_models"], 2)
        self.assertEqual(r["by_tier"]["unsupported"], [])

    def test_write_env_and_backup(self):
        tmp = Path(tempfile.mkdtemp(prefix="imagegen-doctor-"))
        env_path = tmp / ".env"
        env_path.write_text("PRESET=anime\nHF_TOKEN=hf_secret\nOUTPUT_DIR=/tmp/x\n", encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "doctor.py"), "--json", "--no-net", "--port", "1", "--env", str(env_path),
               "--write-env"]
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, timeout=120,
                           env=dict(clean_env(FAKE_DEVICE="cuda", FAKE_MEMORY_GB="12", FAKE_OS="Linux"),
                                    PYTHONIOENCODING="utf-8"))
        self.assertEqual(p.returncode, 0, p.stderr)
        d = json.loads(p.stdout.decode("utf-8"))
        self.assertEqual(d["written"]["path"], str(env_path))
        self.assertTrue((tmp / ".env.bak").is_file())
        self.assertEqual((tmp / ".env.bak").read_text(encoding="utf-8").splitlines()[0], "PRESET=anime")
        new = env_path.read_text(encoding="utf-8")
        self.assertIn("PRESET=anime\n", new)  # 12GB なら anime が推奨
        self.assertIn("MAX_LOADED_MODELS=1\n", new)
        self.assertIn("HF_TOKEN=hf_secret\n", new)  # 既存の値は引き継ぐ
        self.assertIn("OUTPUT_DIR=/tmp/x\n", new)


if __name__ == "__main__":
    unittest.main()
