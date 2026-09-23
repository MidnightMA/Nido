"""Laya-MLX typed decision agent and runtime management for Nido."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nido.config import LayaConfig
from nido.logging import get_logger

logger = get_logger("nido.laya")

# Attempt importing laya runtime
try:
    import laya
except ImportError:
    try:
        import laya_mlx as laya  # type: ignore[no-redef]
    except ImportError:
        laya = None  # type: ignore[assignment]


def detect_mlx_device(preference: str = "auto") -> str:
    """Determine the optimal validated MLX backend device (cpu or gpu) on Linux."""
    pref = (preference or "auto").lower()
    if pref in ("cpu", "gpu"):
        return pref

    try:
        import mlx.core as mx

        # Check if default or available device has GPU support
        dev = mx.default_device()
        dev_type = str(getattr(dev, "type", dev)).lower()
        if "gpu" in dev_type or "cuda" in dev_type:
            return "gpu"
    except Exception as e:
        logger.debug(f"MLX device probe failed: {e}")

    return "cpu"


@dataclass
class ActionDecision:
    """Result of a Laya typed choice decision."""

    selected_id: str
    confidence: float = 1.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    raw_response: Any = None


class LayaDecisionAgent:
    """Long-lived in-memory decision agent using local Laya-MLX checkpoint."""

    def __init__(self, config: LayaConfig) -> None:
        self.config = config
        self.model_dir = Path(config.model_dir).expanduser().resolve()
        self.device = detect_mlx_device(config.device)
        self._agent: Optional[Any] = None
        self._load_agent()

    @property
    def is_loaded(self) -> bool:
        return self._agent is not None

    def _load_agent(self) -> None:
        """Load local checkpoint once into memory."""
        if laya is None:
            logger.info("laya / laya-mlx package not installed. LayaDecisionAgent inactive.")
            return

        if not self.model_dir.is_dir():
            logger.warning(f"Laya model directory does not exist: {self.model_dir}")
            return

        # Ensure directory contains weight / config files
        config_file = self.model_dir / "config.json"
        if not config_file.is_file():
            logger.warning(f"Missing config.json in {self.model_dir}. Run 'nido models setup'.")
            return

        try:
            logger.info(f"Loading Laya-MLX checkpoint from {self.model_dir} (device={self.device})...")
            t0 = time.time()
            load_kwargs: Dict[str, Any] = {
                "device": self.device,
            }
            if self.config.dtype:
                load_kwargs["dtype"] = self.config.dtype

            # Reusable instance
            self._agent = laya.load(str(self.model_dir), **load_kwargs)
            elapsed = (time.time() - t0) * 1000
            logger.info(f"Laya-MLX agent successfully loaded in {elapsed:.1f}ms.")
        except Exception as e:
            logger.error(f"Failed to load Laya-MLX model: {e}")
            self._agent = None

    def predict_action(
        self,
        state_text: str,
        candidates: List[Any],  # List[ActionCandidate]
    ) -> ActionDecision:
        """Perform a typed choice decision over candidate actions."""
        if not candidates:
            return ActionDecision(selected_id="BLOCKED", confidence=1.0)

        if not self.is_loaded:
            logger.warning("Laya model not loaded; falling back to rule-based MockLayaAgent.")
            mock = MockLayaAgent(self.config)
            return mock.predict_action(state_text, candidates)

        # Build choice criteria
        criteria: Dict[str, str] = {c.id: c.label for c in candidates}

        # Shortlist if candidate space exceeds budget and shortlist function is available
        working_candidates = candidates
        if (
            len(candidates) > self.config.shortlist_size
            and hasattr(self._agent, "predict_shortlist")
        ):
            try:
                shortlist_ids = self._agent.predict_shortlist(
                    state=state_text,
                    criteria=criteria,
                    k=self.config.shortlist_size,
                )
                if shortlist_ids:
                    id_set = set(shortlist_ids)
                    # Always ensure DONE and BLOCKED remain in candidate pool
                    for c in candidates:
                        if c.action_type in ("done", "blocked", "wait"):
                            id_set.add(c.id)
                    working_candidates = [c for c in candidates if c.id in id_set]
                    criteria = {c.id: c.label for c in working_candidates}
            except Exception as e:
                logger.debug(f"Shortlist prediction error: {e}. Using direct choice.")

        questions = {
            "next_action": {
                "type": "choice",
                "instructions": (
                    "Choose exactly one action that best advances the user's goal "
                    "from the current desktop state."
                ),
                "criteria": criteria,
            }
        }

        t_start = time.time()
        try:
            result = self._agent.predict(state_text, questions)
            t_elapsed_ms = (time.time() - t_start) * 1000

            return self._parse_result(result, working_candidates, t_elapsed_ms)
        except Exception as e:
            logger.error(f"Laya predict error: {e}. Falling back to mock decision.")
            mock = MockLayaAgent(self.config)
            return mock.predict_action(state_text, candidates)

    def _parse_result(
        self,
        result: Any,
        candidates: List[Any],
        elapsed_ms: float,
    ) -> ActionDecision:
        """Parse structured probabilities and selected choice from Laya output."""
        selected_id = candidates[0].id
        confidence = 1.0
        probabilities: Dict[str, float] = {}

        if isinstance(result, dict):
            q_res = result.get("next_action", result)
            if isinstance(q_res, dict):
                selected_id = (
                    q_res.get("choice")
                    or q_res.get("answer")
                    or q_res.get("selection")
                    or selected_id
                )
                confidence = float(q_res.get("confidence", q_res.get("probability", 1.0)))
                probs = q_res.get("probabilities", q_res.get("scores", {}))
                if isinstance(probs, dict):
                    probabilities = {str(k): float(v) for k, v in probs.items()}
            elif isinstance(q_res, str):
                selected_id = q_res

        # Validate selected_id belongs to candidates
        valid_ids = {c.id for c in candidates}
        if selected_id not in valid_ids:
            logger.warning(f"Laya selected ID '{selected_id}' not in candidates. Selecting first candidate.")
            selected_id = candidates[0].id

        return ActionDecision(
            selected_id=selected_id,
            confidence=confidence,
            probabilities=probabilities,
            inference_time_ms=elapsed_ms,
            raw_response=result,
        )


class MockLayaAgent:
    """Deterministic, rule-based decision agent for testing and fallback."""

    def __init__(self, config: Optional[LayaConfig] = None) -> None:
        self.config = config or LayaConfig()
        self.device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return True

    def predict_action(
        self,
        state_text: str,
        candidates: List[Any],  # List[ActionCandidate]
    ) -> ActionDecision:
        """Select action using deterministic heuristic matching."""
        if not candidates:
            return ActionDecision(selected_id="BLOCKED", confidence=1.0)

        # Extract user goal from state_text
        goal_match = re.search(r"USER GOAL\s*\n(.*?)(?:\n\n|\nCURRENT|$)", state_text, re.DOTALL)
        goal = goal_match.group(1).strip() if goal_match else state_text
        goal_lower = goal.lower()

        # Parse action history from state_text
        history_match = re.search(r"ACTION HISTORY\s*\n(.*?)(?:\n\n|\nCURRENT|\nAVAILABLE|$)", state_text, re.DOTALL)
        history_text = history_match.group(1).lower() if history_match else ""

        has_executed_action = bool(history_text and "success" in history_text)
        has_set_text = bool(
            history_text
            and any(k in history_text for k in ("set requested text", "set text", "set_ui_text", "typed", "type text", "insert_ui_text"))
            and "success" in history_text
        )

        def is_app_already_opened(app_name: str) -> bool:
            app = app_name.lower()
            app_match = re.search(r"\nApplication:\s*([^\n]+)", state_text)
            active_app = app_match.group(1).strip().lower() if app_match else ""
            if active_app and (app in active_app or active_app in app):
                return True
            return bool(
                history_text
                and (
                    f"open application: {app}" in history_text
                    or f"open application {app}" in history_text
                    or f"opened application {app}" in history_text
                )
            )

        # Candidate helpers
        wait_candidates = [c for c in candidates if c.action_type == "wait"]
        done_candidates = [c for c in candidates if c.action_type == "done"]

        open_keywords = ["open", "launch", "run", "باز کن", "اجرا کن"]
        wants_open = any(k in goal_lower for k in open_keywords)
        text_keywords = ["write", "type", "enter", "بنویس", "تایپ"]
        wants_text = any(k in goal_lower for k in text_keywords)
        calc_keywords = ["calculate", "compute", "solve", "حساب کن", "محاسبه کن"]
        wants_calc = any(k in goal_lower for k in calc_keywords) or bool(re.search(r'\d+[\s\+\-\*\/\.xX]+\d+', goal_lower))

        # 1. Check app launching tasks (prioritized if app is not yet opened)
        if wants_open:
            app_candidates = [c for c in candidates if c.action_type == "open_app"]
            for ac in app_candidates:
                app_name = ac.arguments.get("app_name", "").lower()
                if app_name and (app_name in goal_lower or self._matches_app_synonym(goal_lower, app_name)):
                    if not is_app_already_opened(app_name):
                        return ActionDecision(
                            selected_id=ac.id,
                            confidence=0.98,
                            probabilities={c.id: (0.98 if c.id == ac.id else 0.02 / len(candidates)) for c in candidates},
                        )

        # 2. Check text entry tasks
        if wants_text:
            text_candidates = [
                c for c in candidates
                if c.action_type in ("set_ui_text", "insert_ui_text", "type_text")
            ]
            if text_candidates and not has_set_text:
                return ActionDecision(
                    selected_id=text_candidates[0].id,
                    confidence=0.95,
                    probabilities={c.id: (0.95 if c.id == text_candidates[0].id else 0.05 / len(candidates)) for c in candidates},
                )
            elif not text_candidates and not has_set_text and wait_candidates and ("open application" in history_text or "opened" in history_text):
                return ActionDecision(
                    selected_id=wait_candidates[0].id,
                    confidence=0.90,
                    probabilities={c.id: (0.90 if c.id == wait_candidates[0].id else 0.10 / len(candidates)) for c in candidates},
                )

        # 3. Check calculation tasks
        if wants_calc:
            calc_text_candidates = [
                c for c in candidates
                if c.action_type in ("set_ui_text", "insert_ui_text", "type_text")
            ]
            if calc_text_candidates and not has_set_text:
                return ActionDecision(
                    selected_id=calc_text_candidates[0].id,
                    confidence=0.95,
                    probabilities={c.id: (0.95 if c.id == calc_text_candidates[0].id else 0.05 / len(candidates)) for c in candidates},
                )
            elif not calc_text_candidates and not has_set_text and wait_candidates and ("open application" in history_text or "opened" in history_text):
                return ActionDecision(
                    selected_id=wait_candidates[0].id,
                    confidence=0.90,
                    probabilities={c.id: (0.90 if c.id == wait_candidates[0].id else 0.10 / len(candidates)) for c in candidates},
                )

        # 4. Check if a requested action already completed -> DONE
        if has_executed_action and done_candidates:
            if (wants_text or wants_calc) and has_set_text:
                return ActionDecision(
                    selected_id=done_candidates[0].id,
                    confidence=0.99,
                    probabilities={c.id: (0.99 if c.id == done_candidates[0].id else 0.01 / len(candidates)) for c in candidates},
                )
            if wants_open and not wants_text and not wants_calc and ("open application" in history_text or "opened" in history_text):
                return ActionDecision(
                    selected_id=done_candidates[0].id,
                    confidence=0.99,
                    probabilities={c.id: (0.99 if c.id == done_candidates[0].id else 0.01 / len(candidates)) for c in candidates},
                )
            # If UI action or navigation already succeeded in history, finish
            if any(k in history_text for k in ("activate [", "performed click", "clicked", "executed action")):
                return ActionDecision(
                    selected_id=done_candidates[0].id,
                    confidence=0.99,
                    probabilities={c.id: (0.99 if c.id == done_candidates[0].id else 0.01 / len(candidates)) for c in candidates},
                )

        # 5. Check UI activation / buttons / navigation (excluding actions already executed)
        best_score = -1.0
        best_cand = None
        for c in candidates:
            if c.action_type in ("done", "blocked", "wait", "open_app"):
                continue
            # Avoid repeating exact same action if already in history
            c_label_lower = c.label.lower()
            if c_label_lower in history_text:
                continue
            words = [w for w in re.split(r"[\s\"']+", c_label_lower) if len(w) > 2]
            score = sum(1.0 for w in words if w in goal_lower)
            if score > best_score:
                best_score = score
                best_cand = c

        if best_cand is not None and best_score > 0:
            return ActionDecision(
                selected_id=best_cand.id,
                confidence=0.90,
                probabilities={c.id: (0.90 if c.id == best_cand.id else 0.10 / len(candidates)) for c in candidates},
            )

        # 6. Static tool candidates (volume, media, etc.)
        vol_keywords = ["volume", "sound", "صدا"]
        if any(k in goal_lower for k in vol_keywords):
            vol_cands = [c for c in candidates if "volume" in c.action_type]
            if vol_cands:
                return ActionDecision(
                    selected_id=vol_cands[0].id,
                    confidence=0.95,
                    probabilities={c.id: (0.95 if c.id == vol_cands[0].id else 0.05 / len(candidates)) for c in candidates},
                )

        media_keywords = ["play", "pause", "track", "music", "پخش", "آهنگ"]
        if any(k in goal_lower for k in media_keywords):
            media_cands = [c for c in candidates if c.action_type in ("play_pause", "next_track", "previous_track")]
            if media_cands:
                return ActionDecision(
                    selected_id=media_cands[0].id,
                    confidence=0.95,
                    probabilities={c.id: (0.95 if c.id == media_cands[0].id else 0.05 / len(candidates)) for c in candidates},
                )

        # 7. General fallback to DONE if at least one action has succeeded
        if has_executed_action and done_candidates:
            return ActionDecision(
                selected_id=done_candidates[0].id,
                confidence=0.95,
                probabilities={c.id: (0.95 if c.id == done_candidates[0].id else 0.05 / len(candidates)) for c in candidates},
            )

        # Default: first non-control candidate, or wait, or done
        non_control = [c for c in candidates if c.action_type not in ("done", "blocked")]
        chosen = non_control[0] if non_control else candidates[0]
        return ActionDecision(
            selected_id=chosen.id,
            confidence=0.70,
            probabilities={c.id: 1.0 / len(candidates) for c in candidates},
        )

    def _matches_app_synonym(self, goal: str, app_name: str) -> bool:
        synonyms = {
            "kate": ["kate", "notes", "note", "editor", "text editor", "کیت"],
            "dolphin": ["dolphin", "files", "file manager", "downloads", "folder", "دلفین"],
            "kcalc": ["kcalc", "calculator", "calc", "ماشین حساب"],
            "firefox": ["firefox", "browser", "web browser", "فایرفاکس"],
            "google-chrome": ["chrome", "google-chrome", "browser", "کروم"],
            "konsole": ["konsole", "terminal", "console", "کنسول"],
            "systemsettings": ["settings", "systemsettings", "تنظیمات"],
        }
        for syn in synonyms.get(app_name.lower(), []):
            if syn in goal:
                return True
        return False
