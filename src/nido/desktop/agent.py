"""Unified multi-step desktop agent driven by Laya-MLX and AT-SPI perception."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nido.accessibility.base import AccessibilityBackend
from nido.accessibility.models import ActionResult, DesktopSnapshot
from nido.accessibility.snapshot import compute_snapshot_diff, format_snapshot_for_prompt
from nido.config import DesktopConfig
from nido.desktop.candidates import ActionCandidate, CandidateBuilder
from nido.desktop.input import InputBackend
from nido.events import EventBus, PipelineEvent, PipelineStage
from nido.laya.agent import ActionDecision, LayaDecisionAgent, MockLayaAgent
from nido.logging import get_logger
from nido.tools.registry import ToolRegistry

logger = get_logger("nido.desktop.agent")


@dataclass
class DesktopAgentResult:
    """Result of multi-step desktop agent execution."""

    success: bool
    message: str
    steps: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class DesktopAgent:
    """Central iterative task controller orchestrating perception, Laya decisions, and actions."""

    def __init__(
        self,
        config: DesktopConfig,
        laya_agent: Optional[Any] = None,  # LayaDecisionAgent or MockLayaAgent
        accessibility: Optional[AccessibilityBackend] = None,
        input_backend: Optional[InputBackend] = None,
        candidate_builder: Optional[CandidateBuilder] = None,
        registry: Optional[ToolRegistry] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.config = config
        self.laya_agent = laya_agent or MockLayaAgent()
        self.accessibility = accessibility
        self.input_backend = input_backend
        self.candidate_builder = candidate_builder or CandidateBuilder()
        self.registry = registry
        self.event_bus = event_bus

    def _emit(self, stage: PipelineStage, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        if self.event_bus:
            event = PipelineEvent(stage=stage, message=message, data=data or {})
            self.event_bus.emit(event)
        logger.info(f"DesktopAgent: [{stage.value.upper()}] {message}")

    def execute_goal(
        self,
        goal: str = "",
        original_persian: str = "",
        max_steps: Optional[int] = None,
        translated_command: str = "",
    ) -> DesktopAgentResult:
        """Execute a multi-step task using iterative Laya-MLX decisions."""
        # Use Persian command as authoritative goal
        user_goal = original_persian or goal or translated_command
        limit = max_steps or self.config.max_steps
        action_history: List[Dict[str, Any]] = []
        prev_snapshot: Optional[DesktopSnapshot] = None

        logger.info(f"Starting Laya desktop agent loop for goal: '{user_goal}' (max_steps={limit})")

        # Verify accessibility subsystem
        if self.accessibility and not self.accessibility.is_available():
            msg = "Accessibility unavailable on this system. Verify AT-SPI2 is enabled."
            self._emit(PipelineStage.ERROR, msg, {"error": msg})
            return DesktopAgentResult(success=False, message=msg, error="accessibility_unavailable")

        step = 0
        consecutive_stale_count = 0

        while step < limit:
            step_num = step + 1

            # 1. Perception: Observe Desktop via AT-SPI
            self._emit(
                PipelineStage.OBSERVING,
                f"Observing desktop (Step {step_num}/{limit})...",
                {"step": step_num, "max_steps": limit, "goal": user_goal},
            )

            snapshot = DesktopSnapshot()
            if self.accessibility:
                snapshot = self.accessibility.get_desktop_snapshot(
                    max_elements=self.config.max_elements,
                    max_depth=self.config.max_depth,
                    include_invisible=self.config.include_invisible,
                    include_offscreen=self.config.include_offscreen,
                )

            # Check if active window is completely inaccessible
            if snapshot.active_window and not snapshot.elements:
                logger.warning(f"No accessible elements found for window '{snapshot.active_window}'")
                if action_history and action_history[-1].get("action") == "open_app":
                    # Application still loading
                    time.sleep(0.5)
                    step += 1
                    continue
                else:
                    msg = f"Accessibility unavailable for window: '{snapshot.active_window}'."
                    self._emit(PipelineStage.ERROR, msg, {"error": msg})
                    return DesktopAgentResult(
                        success=False,
                        message=msg,
                        steps=step_num,
                        history=action_history,
                        error="unsupported_accessibility",
                    )

            diff_msg = compute_snapshot_diff(prev_snapshot, snapshot)
            logger.debug(f"Snapshot diff: {diff_msg}")
            prev_snapshot = snapshot

            # 2. Build Action Candidates
            candidates = self.candidate_builder.build_candidates(
                goal=user_goal,
                snapshot=snapshot,
                registry=self.registry,
                action_history=action_history,
            )

            if not candidates:
                msg = "No valid action candidates available."
                self._emit(PipelineStage.BLOCKED, msg, {"error": msg})
                return DesktopAgentResult(
                    success=False,
                    message=msg,
                    steps=step_num,
                    history=action_history,
                    error="no_candidates",
                )

            # 3. Construct Compact Laya State
            state_text = self._build_laya_state(
                goal=user_goal,
                step=step_num,
                limit=limit,
                snapshot=snapshot,
                history=action_history,
                candidates=candidates,
            )

            # 4. Model Decision via Laya-MLX
            self._emit(
                PipelineStage.PLANNING,
                f"Selecting next action (Step {step_num}/{limit})...",
                {
                    "step": step_num,
                    "max_steps": limit,
                    "active_app": snapshot.active_application,
                    "active_window": snapshot.active_window,
                },
            )

            decision: ActionDecision = self.laya_agent.predict_action(state_text, candidates)
            selected_id = decision.selected_id

            # Find chosen candidate
            selected_cand = next((c for c in candidates if c.id == selected_id), None)
            if not selected_cand:
                logger.warning(f"Selected candidate ID '{selected_id}' not found in candidate set. Defaulting to first candidate.")
                selected_cand = candidates[0]

            logger.info(
                f"Laya selected action: {selected_cand.id} - {selected_cand.label} "
                f"(confidence={decision.confidence:.2f}, inference_ms={decision.inference_time_ms:.1f})"
            )

            # 5. Stale Element Protection
            if selected_cand.element_id:
                target_el = snapshot.get_element(selected_cand.element_id)
                if not target_el:
                    consecutive_stale_count += 1
                    logger.warning(f"Stale UI element rejected: [{selected_cand.element_id}]")
                    if consecutive_stale_count >= 3:
                        msg = "Aborted: UI is changing too rapidly (repeated stale elements)."
                        self._emit(PipelineStage.ERROR, msg, {"error": msg})
                        return DesktopAgentResult(
                            success=False,
                            message=msg,
                            steps=step_num,
                            history=action_history,
                            error="repeated_stale_elements",
                        )
                    # Settle and re-observe
                    time.sleep(self.config.settle_delay_ms / 1000.0)
                    step += 1
                    continue

            consecutive_stale_count = 0

            # 6. Terminal Actions: DONE and BLOCKED
            if selected_cand.action_type == "done":
                # Completion guard: ensure at least one useful action occurred or task is obviously finished
                has_useful_action = any(h.get("success") for h in action_history)
                if not has_useful_action and len(candidates) > 2:
                    # Reject premature DONE unless no actions could be taken
                    logger.info("Completion guard: premature DONE rejected without useful action.")
                    # Try next best candidate that isn't done/blocked
                    alt_cand = next((c for c in candidates if c.action_type not in ("done", "blocked")), None)
                    if alt_cand:
                        selected_cand = alt_cand
                    else:
                        summary = selected_cand.arguments.get("summary", "Task complete.")
                        self._emit(PipelineStage.DONE, summary, {"history": action_history})
                        return DesktopAgentResult(success=True, message=summary, steps=step_num, history=action_history)
                else:
                    summary = selected_cand.arguments.get("summary", "Goal completed successfully.")
                    self._emit(
                        PipelineStage.DONE,
                        summary,
                        {
                            "step": step_num,
                            "max_steps": limit,
                            "summary": summary,
                            "history": action_history,
                        },
                    )
                    return DesktopAgentResult(
                        success=True,
                        message=summary,
                        steps=step_num,
                        history=action_history,
                    )

            if selected_cand.action_type == "blocked":
                reason = selected_cand.arguments.get("reason", "Action blocked.")
                self._emit(PipelineStage.BLOCKED, reason, {"error": reason})
                return DesktopAgentResult(
                    success=False,
                    message=reason,
                    steps=step_num,
                    history=action_history,
                    error="blocked",
                )

            # Guard against repeated open_app loop
            if selected_cand.action_type == "open_app":
                target_app = selected_cand.arguments.get("app_name", "").lower()
                already_opened = any(
                    h.get("action") == "open_app"
                    and h.get("arguments", {}).get("app_name", "").lower() == target_app
                    and h.get("success")
                    for h in action_history
                )
                if already_opened:
                    logger.warning(f"App '{target_app}' was already opened in this task; ignoring duplicate open_app.")
                    alt_cand = next(
                        (c for c in candidates if c.action_type != "open_app" and c.action_type not in ("done", "blocked")),
                        None,
                    )
                    if alt_cand:
                        selected_cand = alt_cand
                    else:
                        done_c = next((c for c in candidates if c.action_type == "done"), None)
                        if done_c and any(h.get("success") for h in action_history):
                            selected_cand = done_c

            # 7. Action Execution
            self._emit(
                PipelineStage.INTERACTING,
                f"Executing {selected_cand.action_type}: {selected_cand.label}",
                {
                    "step": step_num,
                    "max_steps": limit,
                    "action": selected_cand.action_type,
                    "arguments": selected_cand.arguments,
                    "label": selected_cand.label,
                    "target_name": selected_cand.label,
                },
            )

            result = self._execute_candidate(selected_cand, snapshot)
            logger.info(f"Action result: success={result.success}, msg='{result.message}'")

            action_history.append({
                "step": step_num,
                "action": selected_cand.action_type,
                "label": selected_cand.label,
                "arguments": selected_cand.arguments,
                "element_id": selected_cand.element_id,
                "success": result.success,
                "message": result.message,
                "error": result.error,
                "laya_confidence": decision.confidence,
            })

            # Bounded settle delay
            time.sleep(self.config.settle_delay_ms / 1000.0)
            step += 1

        # Max steps exceeded
        msg = f"Task reached maximum allowed steps ({limit}) without finishing."
        self._emit(PipelineStage.ERROR, msg, {"error": msg})
        return DesktopAgentResult(
            success=False,
            message=msg,
            steps=limit,
            history=action_history,
            error="max_steps_exceeded",
        )

    def _build_laya_state(
        self,
        goal: str,
        step: int,
        limit: int,
        snapshot: DesktopSnapshot,
        history: List[Dict[str, Any]],
        candidates: List[ActionCandidate],
    ) -> str:
        """Serialize compact observation state for Laya decision input."""
        lines = [
            f"USER GOAL\n{goal}\n",
            f"CURRENT STEP\n{step} / {limit}\n",
        ]

        if history:
            lines.append("ACTION HISTORY")
            for h in history[-4:]:  # Keep recent 4 steps to avoid token bloat
                status = "success" if h.get("success") else "failed"
                lines.append(f"{h['step']}. {h.get('label', h['action'])} -> {status}")
            lines.append("")

        lines.append("CURRENT DESKTOP")
        lines.append(f"Session: {snapshot.session_type}")
        lines.append(f"Desktop: {snapshot.desktop_name}")
        lines.append(f"Application: {snapshot.active_application or 'None'}")
        lines.append(f"Window: {snapshot.active_window or 'None'}")
        if snapshot.focused_element_id:
            lines.append(f"Focused element: [{snapshot.focused_element_id}]")
        lines.append("")

        lines.append("AVAILABLE UI")
        ui_lines = format_snapshot_for_prompt(snapshot)
        lines.append(ui_lines)
        lines.append("")

        lines.append("AVAILABLE ACTIONS")
        for c in candidates:
            lines.append(f"[{c.id}] {c.label}")

        return "\n".join(lines)

    def _execute_candidate(
        self,
        candidate: ActionCandidate,
        snapshot: DesktopSnapshot,
    ) -> ActionResult:
        """Execute validated candidate action with priority to semantic accessibility."""
        action_type = candidate.action_type
        args = candidate.arguments
        snap_id = snapshot.snapshot_id

        # 1. Wait Action
        if action_type == "wait":
            dur = args.get("duration_ms", 200) / 1000.0
            time.sleep(dur)
            return ActionResult(success=True, action="wait", message=f"Waited {args.get('duration_ms', 200)}ms.")

        # 2. Static Tool Execution
        if candidate.source == "static" and self.registry:
            if self.registry.get_tool(action_type):
                reg_res = self.registry.execute(action_type, args)
                if action_type == "open_app":
                    time.sleep(0.5)  # Allow app time to launch
                return ActionResult(
                    success=bool(reg_res.get("success", True)),
                    action=action_type,
                    message=reg_res.get("message", f"Executed {action_type}."),
                    error=reg_res.get("error"),
                )

        # 3. Dynamic Desktop Actions
        if not self.accessibility:
            return ActionResult(success=False, action=action_type, error="no_accessibility", message="No accessibility backend.")

        if action_type == "activate_ui_element":
            el_id = args["element_id"]
            res = self.accessibility.perform_action(el_id, "click", snapshot_id=snap_id)
            if not res.success and self.input_backend:
                # Coordinate fallback if bounds available
                el = snapshot.get_element(el_id)
                if el and el.bounds and el.bounds[2] > 0 and el.bounds[3] > 0:
                    cx = el.bounds[0] + el.bounds[2] // 2
                    cy = el.bounds[1] + el.bounds[3] // 2
                    logger.info(f"AT-SPI action failed; falling back to coordinates ({cx}, {cy})")
                    ok = self.input_backend.click_coords(cx, cy)
                    return ActionResult(
                        success=ok,
                        action="activate_ui_element",
                        element_id=el_id,
                        message=f"Clicked coordinates ({cx}, {cy}) for [{el_id}].",
                    )
            return res

        elif action_type == "focus_ui_element":
            return self.accessibility.focus_element(args["element_id"], snapshot_id=snap_id)

        elif action_type == "set_ui_text":
            el_id = args["element_id"]
            text = args["text"]
            res = self.accessibility.set_text(el_id, text, snapshot_id=snap_id)
            if not res.success and self.input_backend:
                logger.info(f"AT-SPI set_text failed for [{el_id}]; attempting synthetic typing fallback.")
                self.accessibility.focus_element(el_id, snapshot_id=snap_id)
                ok = self.input_backend.type_text(text)
                return ActionResult(
                    success=ok,
                    action="set_ui_text",
                    element_id=el_id,
                    message=f"Typed text via synthetic input into [{el_id}].",
                )
            return res

        elif action_type == "insert_ui_text":
            el_id = args["element_id"]
            text = args["text"]
            res = self.accessibility.insert_text(el_id, text, snapshot_id=snap_id)
            if not res.success and self.input_backend:
                self.accessibility.focus_element(el_id, snapshot_id=snap_id)
                ok = self.input_backend.type_text(text)
                return ActionResult(
                    success=ok,
                    action="insert_ui_text",
                    element_id=el_id,
                    message=f"Typed text via synthetic input into [{el_id}].",
                )
            return res

        elif action_type == "select_ui_element":
            return self.accessibility.select_element(args["element_id"], snapshot_id=snap_id)

        elif action_type == "scroll_ui":
            return self.accessibility.scroll(args["element_id"], args.get("direction", "down"), snapshot_id=snap_id)

        elif action_type == "type_text" and self.input_backend:
            text = args.get("text", "")
            ok = self.input_backend.type_text(text)
            return ActionResult(
                success=ok,
                action="type_text",
                message=f"Typed text '{text}'.",
            )

        elif action_type == "press_key" and self.input_backend:
            ok = self.input_backend.press_key(args["key"])
            return ActionResult(success=ok, action="press_key", message=f"Pressed key '{args['key']}'.")

        elif action_type == "press_hotkey" and self.input_backend:
            ok = self.input_backend.press_hotkey(args["keys"])
            return ActionResult(success=ok, action="press_hotkey", message=f"Pressed hotkey {args['keys']}.")

        return ActionResult(
            success=False,
            action=action_type,
            error="unsupported_action",
            message=f"Action '{action_type}' is not supported.",
        )
