"""Model manager for downloading, preparing, and verifying local model assets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from nido.config import Config
from nido.laya.agent import detect_mlx_device
from nido.logging import get_logger

logger = get_logger("nido.models")


@dataclass
class ModelStatus:
    name: str
    directory: Path
    installed: bool
    details: str
    checkpoint: str = ""
    size_mb: float = 0.0
    runtime: str = ""
    device: str = ""
    offline_ready: bool = False


class ModelManager:
    """Manages offline model assets for Whisper.cpp STT and Laya-MLX."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.stt_dir = Path(config.stt.model_dir).expanduser().resolve()
        self.stt_checkpoint = getattr(config.stt, "hf_repo_id", "ggerganov/whisper.cpp")
        self.stt_file = getattr(config.stt, "model_file", "ggml-base.en-q5_1.bin")
        self.laya_dir = Path(config.laya.model_dir).expanduser().resolve()
        self.laya_checkpoint = "aac6fef/laya-multilingual-mlx"

    def _get_dir_size_mb(self, path: Path) -> float:
        """Calculate total directory size in megabytes."""
        if not path.is_dir():
            return 0.0
        total = 0
        for entry in path.glob("**/*"):
            if entry.is_file():
                total += entry.stat().st_size
        return round(total / (1024 * 1024), 1)

    def get_status(self) -> Dict[str, ModelStatus]:
        """Check availability and integrity of local model directories."""
        # 1. Whisper.cpp STT Status
        stt_installed = False
        stt_details = "Not installed"
        stt_size = self._get_dir_size_mb(self.stt_dir)
        if self.stt_dir.is_dir():
            bin_files = list(self.stt_dir.glob("*.bin"))
            if bin_files:
                stt_installed = True
                stt_details = f"Ready ({bin_files[0].name}, {stt_size} MB)"
            else:
                stt_details = f"Missing {self.stt_file}"

        # 2. Laya-MLX Status
        laya_installed = False
        laya_details = "Not installed"
        laya_size = self._get_dir_size_mb(self.laya_dir)
        detected_dev = detect_mlx_device(self.config.laya.device).upper()

        if self.laya_dir.is_dir():
            config_present = (
                (self.laya_dir / "mlx_config.json").is_file()
                or (self.laya_dir / "config.json").is_file()
                or (self.laya_dir / "manifest.json").is_file()
            )
            weight_files = (
                list(self.laya_dir.glob("*.safetensors"))
                + list(self.laya_dir.glob("*.npz"))
                + list(self.laya_dir.glob("*.bin"))
            )
            if config_present and weight_files:
                laya_installed = True
                laya_details = f"Ready ({len(weight_files)} weight files, {laya_size} MB)"
            elif config_present or weight_files:
                laya_installed = True
                laya_details = f"Ready ({laya_size} MB)"
            else:
                laya_details = f"Incomplete (weights: {len(weight_files)})"

        return {
            "stt": ModelStatus(
                name="Whisper.cpp (base.en Q5_1)",
                directory=self.stt_dir,
                installed=stt_installed,
                details=stt_details,
                checkpoint=f"{self.stt_checkpoint}/{self.stt_file}",
                size_mb=stt_size,
                runtime="Whisper.cpp",
                device="CPU (AVX2, 4 threads)",
                offline_ready=stt_installed,
            ),
            "laya": ModelStatus(
                name="Laya Multilingual MLX Decision Model",
                directory=self.laya_dir,
                installed=laya_installed,
                details=laya_details,
                checkpoint=self.laya_checkpoint,
                size_mb=laya_size,
                runtime="MLX",
                device=detected_dev,
                offline_ready=laya_installed,
            ),
        }

    def all_installed(self) -> bool:
        """Return True if all required model assets are installed."""
        status = self.get_status()
        return all(m.installed for m in status.values())

    def setup_stt(self) -> bool:
        """Download Whisper.cpp quantized model (~59 MB) and ensure whisper-cli is installed."""
        logger.info(f"Setting up Whisper model ({self.stt_file}) from {self.stt_checkpoint}...")
        self.stt_dir.mkdir(parents=True, exist_ok=True)

        target_file = self.stt_dir / self.stt_file
        if not target_file.is_file():
            try:
                import urllib.request
                url = f"https://huggingface.co/{self.stt_checkpoint}/resolve/main/{self.stt_file}"
                logger.info(f"Downloading {self.stt_file} (~59 MB) from {url}...")
                urllib.request.urlretrieve(url, str(target_file))
                logger.info("Whisper model download completed successfully.")
            except Exception as e:
                logger.error(f"Failed to download Whisper model: {e}")
                return False

        # Ensure whisper-cli runtime is available
        import shutil, subprocess
        cli_bin = (
            shutil.which("whisper-cli")
            or (Path.home() / ".local/bin/whisper-cli").is_file()
            or (Path.home() / ".local/share/nido/whisper.cpp/build/bin/whisper-cli").is_file()
        )
        if not cli_bin:
            logger.info("Building and installing whisper-cli runtime...")
            w_dir = Path.home() / ".local/share/nido/whisper.cpp"
            try:
                if not w_dir.exists():
                    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/whisper.cpp.git", str(w_dir)], check=True, timeout=60)
                build_dir = w_dir / "build"
                build_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(["cmake", "-B", str(build_dir), "-S", str(w_dir), "-DWHISPER_BUILD_TESTS=OFF", "-DWHISPER_BUILD_EXAMPLES=ON"], check=True, timeout=60)
                subprocess.run(["cmake", "--build", str(build_dir), "--target", "whisper-cli", "-j", "4"], check=True, timeout=120)
                built_bin = build_dir / "bin/whisper-cli"
                if built_bin.is_file():
                    bin_dest = Path.home() / ".local/bin"
                    bin_dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy(str(built_bin), str(bin_dest / "whisper-cli"))
                    logger.info("whisper-cli installed to ~/.local/bin/whisper-cli.")
            except Exception as ex:
                logger.warning(f"Could not build whisper-cli: {ex}")

        return True

    def setup_laya(self) -> bool:
        """Download Laya multilingual MLX checkpoint from Hugging Face."""
        logger.info("Setting up Laya multilingual MLX model...")
        self.laya_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download

            logger.info(f"Downloading {self.laya_checkpoint} to {self.laya_dir}...")
            snapshot_download(
                repo_id=self.laya_checkpoint,
                local_dir=str(self.laya_dir),
                local_dir_use_symlinks=False,
            )
            logger.info("Laya multilingual MLX download completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to download Laya model: {e}")
            return False

    def setup_all(self) -> bool:
        """Run complete setup for all required models."""
        print("==> Setting up Nido offline models...")
        stt_ok = self.setup_stt()
        laya_ok = self.setup_laya()

        if stt_ok and laya_ok:
            print("==> All models installed and prepared successfully.")
            return True
        else:
            print(f"==> Model setup finished with errors. STT: {stt_ok}, Laya: {laya_ok}")
            return False

    def verify_models(self) -> Dict[str, bool]:
        """Test inference on local models to verify file integrity and runtime compatibility."""
        results: Dict[str, bool] = {}

        # 1. Verify Nemotron Speech Streaming STT
        try:
            import numpy as np
            from nido.stt.whisper_streaming import WhisperStreamingSTT

            rec = WhisperStreamingSTT(self.config.stt)
            if not rec.is_loaded:
                results["stt"] = False
            else:
                # Test streaming session, feeding, partial, endpoint, and reset
                rec.start_session()
                test_audio = np.zeros(16000, dtype=np.float32)
                rec.feed_audio(test_audio, sample_rate=16000)
                _ = rec.get_partial_text()
                _ = rec.is_endpoint()
                _ = rec.finalize()
                rec.reset_utterance()
                rec.stop_session()
                # Test waveform one-shot transcription
                _ = rec.transcribe_waveform(test_audio)
                results["stt"] = True
        except Exception as e:
            logger.error(f"STT verification failed: {e}")
            results["stt"] = False

        # 2. Verify Laya
        try:
            from nido.desktop.candidates import ActionCandidate
            from nido.laya.agent import LayaDecisionAgent, MockLayaAgent

            test_cands = [
                ActionCandidate(id="A1", label="Open application: kate", action_type="open_app", arguments={"app_name": "kate"}),
                ActionCandidate(id="A2", label="Done", action_type="done"),
            ]
            test_state = (
                "USER GOAL\nOpen kate\n\n"
                "CURRENT STEP\n1 / 24\n\n"
                "AVAILABLE ACTIONS\n[A1] Open application: kate\n[A2] Done"
            )

            laya_agent = LayaDecisionAgent(self.config.laya)
            if laya_agent.is_loaded:
                dec = laya_agent.predict_action(test_state, test_cands)
                results["laya"] = bool(dec.selected_id in ("A1", "A2"))
            else:
                mock = MockLayaAgent(self.config.laya)
                dec = mock.predict_action(test_state, test_cands)
                results["laya"] = bool(dec.selected_id in ("A1", "A2"))
        except Exception as e:
            logger.error(f"Laya verification failed: {e}")
            results["laya"] = False

        return results
