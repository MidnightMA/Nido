"""Iterative accessibility-driven desktop agent loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nido.accessibility.base import AccessibilityBackend
from nido.accessibility.models import ActionResult, DesktopSnapshot
from nido.accessibility.snapshot import compute_snapshot_diff, format_snapshot_for_prompt
from nido.config import DesktopConfig
from nido.desktop.input import InputBackend
from nido.desktop.tool_builder import DynamicDesktopToolBuilder
from nido.events import EventBus, PipelineEvent, PipelineStage
from nido.logging import get_logger
from nido.needle.agent import NeedleCommandRouter
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
    """Orchestrates iterative perception, dynamic action selection, and execution."""

    def __init__(
        self,
        config: DesktopConfig,
        router: NeedleCommandRouter,
        accessibility: AccessibilityBackend,
        input_backend: InputBackend,
        tool_builder: Optional[DynamicDesktopToolBuilder] = None,
        registry: Optional[ToolRegistry] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.config = config
        self.router = router
        self.accessibility = accessibility
        self.input_backend = input_backend
        self.tool_builder = tool_builder or DynamicDesktopToolBuilder()
        self.registry = registry
        self.event_bus = event_bus

    def _emit(self, stage: PipelineStage, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        if self.event_bus:
            event = PipelineEvent(stage=stage, message=message, data=data or {})
            self.event_bus.emit(event)
        logger.info(f"DesktopAgent: [{stage.value.upper()}] {message}")

    def execute_goal(
        self,
        translated_command: str,
        original_persian: str = "",
        max_steps: Optional[int] = None,
    ) -> DesktopAgentResult:
        """Execute a multi-step desktop interaction goal."""
        limit = max_steps or self.config.max_steps
        action_history: List[Dict[str, Any]] = []
        prev_snapshot: Optional[DesktopSnapshot] = None

        logger.info(f"Starting desktop agent loop for goal: '{translated_command}' (max_steps={limit})")

        # Check accessibility availability
        if not self.accessibility.is_available():
            msg = "Accessibility unavailable on this system. Verify AT-SPI2 is enabled."
            self._emit(PipelineStage.ERROR, msg, {"error": msg})
            return DesktopAgentResult(success=False, message=msg, error="accessibility_unavailable")

        step = 0
        consecutive_stale_count = 0

        while step < limit:
            step_num = step + 1

            # 1. Observe Desktop
            self._emit(
                PipelineStage.OBSERVING,
                f"Observing desktop (Step {step_num}/{limit})...",
                {"step": step_num, "max_steps": limit, "goal": translated_command},
            )

            snapshot = self.accessibility.get_desktop_snapshot(
                max_elements=self.config.max_elements,
                max_depth=self.config.max_depth,
                include_invisible=self.config.include_invisible,
                include_offscreen=self.config.include_offscreen,
            )

            # Check if active window is completely inaccessible
            if snapshot.active_window and not snapshot.elements:
                logger.warning(f"No accessible elements found for window '{snapshot.active_window}'")
                # If we just launched an app, it might still be loading
                if action_history and action_history[-1].get("action") == "open_app":
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

            prompt_state = format_snapshot_for_prompt(snapshot)
            diff_msg = compute_snapshot_diff(prev_snapshot, snapshot)
            logger.debug(f"Snapshot diff: {diff_msg}")
            prev_snapshot = snapshot

            # 2. Build Dynamic Action Space
            tool_schemas = self.tool_builder.build_tool_schemas(snapshot)

            # 3. Model Decision
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

            action_name, action_args = self.router.decide_desktop_action(
                goal=translated_command,
                desktop_state=prompt_state,
                tool_schemas=tool_schemas,
                original_persian=original_persian,
                action_history=action_history,
            )

            # 4. Action Validation
            is_valid, validation_err = self.tool_builder.validate_call(action_name, action_args, snapshot)

            if not is_valid:
                logger.warning(f"Action validation rejected: {validation_err}")
                if validation_err and "stale_ui_element" in validation_err:
                    consecutive_stale_count += 1
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
                else:
                    msg = f"Invalid action chosen: {validation_err}"
                    self._emit(PipelineStage.ERROR, msg, {"error": msg})
                    return DesktopAgentResult(
                        success=False,
                        message=msg,
                        steps=step_num,
                        history=action_history,
                        error="invalid_action",
                    )

            consecutive_stale_count = 0

            # 5. Terminal "done" Action
            if action_name == "done":
                summary = action_args.get("summary", "Goal completed successfully.")
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

            # 6. Action Execution
            target_el_name = ""
            if "element_id" in action_args:
                target_el = snapshot.get_element(action_args["element_id"])
                if target_el:
                    target_el_name = target_el.name or target_el.role

            self._emit(
                PipelineStage.INTERACTING,
                f"Executing {action_name}: '{target_el_name or action_args}'",
                {
                    "step": step_num,
                    "max_steps": limit,
                    "action": action_name,
                    "arguments": action_args,
                    "target_name": target_el_name,
                },
            )

            result = self._execute_action(action_name, action_args, snapshot)
            logger.info(f"Action result: success={result.success}, msg='{result.message}'")

            action_history.append({
                "step": step_num,
                "action": action_name,
                "arguments": action_args,
                "element_id": action_args.get("element_id"),
                "element_name": target_el_name,
                "success": result.success,
                "message": result.message,
                "error": result.error,
            })

            # Settle delay
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

    def _execute_action(
        self,
        action_name: str,
        args: Dict[str, Any],
        snapshot: DesktopSnapshot,
    ) -> ActionResult:
        """Execute a validated action via AT-SPI or fallback input backend."""
        snap_id = snapshot.snapshot_id

        if action_name == "activate_ui_element":
            el_id = args["element_id"]
            res = self.accessibility.perform_action(el_id, "click", snapshot_id=snap_id)
            if not res.success:
                # Coordinate fallback if element has bounds
                el = snapshot.get_element(el_id)
                if el and el.bounds and el.bounds[2] > 0 and el.bounds[3] > 0:
                    cx = el.bounds[0] + el.bounds[2] // 2
                    cy = el.bounds[1] + el.bounds[3] // 2
                    logger.info(f"AT-SPI action failed; attempting coordinate fallback at ({cx}, {cy})")
                    ok = self.input_backend.click_coords(cx, cy)
                    return ActionResult(
                        success=ok,
                        action="activate_ui_element",
                        element_id=el_id,
                        message=f"Clicked coordinates ({cx}, {cy}) for [{el_id}].",
                    )
            return res

        elif action_name == "focus_ui_element":
            return self.accessibility.focus_element(args["element_id"], snapshot_id=snap_id)

        elif action_name == "set_ui_text":
            el_id = args["element_id"]
            text = args["text"]
            res = self.accessibility.set_text(el_id, text, snapshot_id=snap_id)
            if not res.success:
                # Fallback to focus + synthetic typing
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

        elif action_name == "insert_ui_text":
            el_id = args["element_id"]
            text = args["text"]
            res = self.accessibility.insert_text(el_id, text, snapshot_id=snap_id)
            if not res.success:
                self.accessibility.focus_element(el_id, snapshot_id=snap_id)
                ok = self.input_backend.type_text(text)
                return ActionResult(
                    success=ok,
                    action="insert_ui_text",
                    element_id=el_id,
                    message=f"Typed text via synthetic input into [{el_id}].",
                )
            return res

        elif action_name == "select_ui_element":
            return self.accessibility.select_element(args["element_id"], snapshot_id=snap_id)

        elif action_name == "scroll_ui":
            return self.accessibility.scroll(args["element_id"], args["direction"], snapshot_id=snap_id)

        elif action_name == "press_key":
            ok = self.input_backend.press_key(args["key"])
            return ActionResult(
                success=ok,
                action="press_key",
                message=f"Pressed key '{args['key']}'.",
            )

        elif action_name == "press_hotkey":
            ok = self.input_backend.press_hotkey(args["keys"])
            return ActionResult(
                success=ok,
                action="press_hotkey",
                message=f"Pressed hotkey {args['keys']}.",
            )

        elif action_name == "open_app":
            app_name = args["app_name"]
            if self.registry and self.registry.get_tool("open_app"):
                reg_res = self.registry.execute("open_app", {"app_name": app_name})
                # Settle longer for app launch
                time.sleep(0.6)
                return ActionResult(
                    success=bool(reg_res.get("success", True)),
                    action="open_app",
                    message=reg_res.get("message", f"Opened app '{app_name}'."),
                    error=reg_res.get("error"),
                )
            return ActionResult(
                success=False,
                action="open_app",
                error="open_app_unavailable",
                message="Tool 'open_app' is not available in registry.",
            )

        elif action_name == "wait":
            dur = args.get("duration_ms", 200) / 1000.0
            time.sleep(dur)
            return ActionResult(
                success=True,
                action="wait",
                message=f"Waited {args.get('duration_ms')}ms.",
            )

        return ActionResult(
            success=False,
            action=action_name,
            error="unknown_action",
            message=f"Action '{action_name}' is not recognized.",
        )
