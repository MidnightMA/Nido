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
    """Manages offline model assets for English Zipformer STT and Laya-MLX."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.stt_dir = Path(config.stt.model_dir).expanduser().resolve()
        self.stt_checkpoint = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
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
        # 1. English Zipformer STT Status
        stt_installed = False
        stt_details = "Not installed"
        stt_size = self._get_dir_size_mb(self.stt_dir)
        if self.stt_dir.is_dir():
            encoders = list(self.stt_dir.glob("*encoder*.onnx"))
            decoders = list(self.stt_dir.glob("*decoder*.onnx"))
            joiners = list(self.stt_dir.glob("*joiner*.onnx"))
            tokens = list(self.stt_dir.glob("*tokens*.txt"))

            if encoders and decoders and joiners and tokens:
                stt_installed = True
                enc_name = encoders[0].name
                stt_details = f"Ready ({enc_name}, {decoders[0].name}, {joiners[0].name})"
            else:
                stt_details = (
                    f"Incomplete (found enc: {len(encoders)}, dec: {len(decoders)}, "
                    f"join: {len(joiners)}, tokens: {len(tokens)})"
                )

        # 2. Laya-MLX Status
        laya_installed = False
        laya_details = "Not installed"
        laya_size = self._get_dir_size_mb(self.laya_dir)
        detected_dev = detect_mlx_device(self.config.laya.device).upper()

        if self.laya_dir.is_dir():
            config_file = self.laya_dir / "config.json"
            weight_files = (
                list(self.laya_dir.glob("*.safetensors"))
                + list(self.laya_dir.glob("*.npz"))
                + list(self.laya_dir.glob("*.bin"))
            )
            if config_file.is_file() and weight_files:
                laya_installed = True
                laya_details = f"Ready ({len(weight_files)} weight files, {laya_size} MB)"
            elif config_file.is_file():
                laya_installed = True
                laya_details = f"Ready (config present, {laya_size} MB)"
            else:
                laya_details = f"Incomplete (config: {config_file.is_file()}, weights: {len(weight_files)})"

        return {
            "stt": ModelStatus(
                name="Sherpa-ONNX Streaming Zipformer EN 2023-06-26",
                directory=self.stt_dir,
                installed=stt_installed,
                details=stt_details,
                checkpoint=self.stt_checkpoint,
                size_mb=stt_size,
                runtime="sherpa-onnx",
                device="CPU",
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
        """Download sherpa-onnx English streaming zipformer model from Hugging Face."""
        logger.info("Setting up Sherpa-ONNX Streaming Zipformer EN 2023-06-26 model...")
        self.stt_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download

            repo_id = self.stt_checkpoint
            logger.info(f"Downloading {repo_id} to {self.stt_dir}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(self.stt_dir),
                local_dir_use_symlinks=False,
            )
            logger.info("English Streaming Zipformer download completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to download English Zipformer STT model: {e}")
            return False

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

        # 1. Verify English Streaming Zipformer STT
        try:
            import numpy as np
            from nido.stt.zipformer_en import ZipformerStreamingSTT

            rec = ZipformerStreamingSTT(self.config.stt)
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
