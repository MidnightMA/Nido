"""KDE Plasma visual overlay widget using PySide6 with dual-mode support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QGuiApplication
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGraphicsDropShadowEffect,
        QHBoxLayout,
        QLabel,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    # Dummy fallbacks for testing or environments without PySide6
    QApplication = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[misc,assignment]
    QTimer = None  # type: ignore[assignment]

from nido.config import UIConfig
from nido.events import PipelineStage
from nido.logging import get_logger

logger = get_logger("nido.ui.overlay")

OVERLAY_STYLE = """
QFrame#MainCard {
    background-color: rgba(20, 24, 33, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
}
QLabel {
    color: #E2E8F0;
    font-family: "Inter", "Segoe UI", "Cantarell", "Noto Sans", sans-serif;
}
QLabel#AppTitle {
    font-size: 13px;
    font-weight: 700;
    color: #38BDF8;
    letter-spacing: 0.5px;
}
QLabel#ModeBadge {
    font-size: 10px;
    font-weight: 700;
    color: #94A3B8;
    padding: 2px 6px;
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    letter-spacing: 0.5px;
}
QLabel#StatusBadge {
    font-size: 11px;
    font-weight: 600;
    color: #F8FAFC;
    padding: 3px 8px;
    background-color: #0284C7;
    border-radius: 6px;
}
QLabel#SectionHeader {
    font-size: 10px;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QLabel#GoalText {
    font-size: 14px;
    font-weight: 500;
    color: #F8FAFC;
}
QLabel#LiveTranscript {
    font-size: 13px;
    font-style: italic;
    color: #38BDF8;
}
QLabel#HistoryItem {
    font-size: 11px;
    color: #94A3B8;
}
QLabel#QueueBadge {
    font-size: 10px;
    color: #F59E0B;
    font-weight: 600;
}
QLabel#AppInfoText {
    font-size: 12px;
    color: #38BDF8;
    font-weight: 500;
}
QLabel#ActionText {
    font-size: 12px;
    font-family: "JetBrains Mono", "Hack", monospace;
    color: #A7F3D0;
}
QLabel#ResultText {
    font-size: 12px;
    color: #F1F5F9;
}
QLabel#TimingBadge {
    font-size: 10px;
    color: #64748B;
}
"""


class NidoOverlay(QWidget):
    """A frameless, unobtrusive KDE-style desktop overlay for Nido."""

    def __init__(self, config: UIConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config = config
        self._recent_history: List[str] = []
        self._max_history = 5

        self._setup_window_flags()
        self._setup_ui()
        self._setup_timer()

    def _setup_window_flags(self) -> None:
        if Qt is None:
            return
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setStyleSheet(OVERLAY_STYLE)

    def _setup_ui(self) -> None:
        if QWidget is object:
            return

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 12, 12, 12)

        # Card container
        self.card = QFrame(self)
        self.card.setObjectName("MainCard")
        self.card_layout = QVBoxLayout(self.card)
        self.card_layout.setContentsMargins(16, 14, 16, 14)
        self.card_layout.setSpacing(6)

        # Top Bar: Title + Mode Badge + Status Badge
        self.top_bar = QHBoxLayout()
        self.title_label = QLabel("🎙 NIDO", self.card)
        self.title_label.setObjectName("AppTitle")

        self.mode_badge = QLabel("IDLE", self.card)
        self.mode_badge.setObjectName("ModeBadge")

        self.status_badge = QLabel("READY", self.card)
        self.status_badge.setObjectName("StatusBadge")

        self.top_bar.addWidget(self.title_label)
        self.top_bar.addWidget(self.mode_badge)
        self.top_bar.addStretch()
        self.top_bar.addWidget(self.status_badge)
        self.card_layout.addLayout(self.top_bar)

        # Live Partial Transcript
        self.live_label = QLabel("", self.card)
        self.live_label.setObjectName("LiveTranscript")
        self.live_label.setWordWrap(True)
        self.live_label.hide()
        self.card_layout.addWidget(self.live_label)

        # Current Command / Goal Section
        self.goal_header = QLabel("Command:", self.card)
        self.goal_header.setObjectName("SectionHeader")
        self.goal_label = QLabel("", self.card)
        self.goal_label.setObjectName("GoalText")
        self.goal_label.setWordWrap(True)
        self.card_layout.addWidget(self.goal_header)
        self.card_layout.addWidget(self.goal_label)

        # Application / Window Section
        self.app_header = QLabel("Application:", self.card)
        self.app_header.setObjectName("SectionHeader")
        self.app_label = QLabel("", self.card)
        self.app_label.setObjectName("AppInfoText")
        self.card_layout.addWidget(self.app_header)
        self.card_layout.addWidget(self.app_label)

        # Action / Step Section
        self.action_header = QLabel("Agent Step:", self.card)
        self.action_header.setObjectName("SectionHeader")
        self.action_label = QLabel("", self.card)
        self.action_label.setObjectName("ActionText")
        self.action_label.setWordWrap(True)
        self.result_label = QLabel("", self.card)
        self.result_label.setObjectName("ResultText")
        self.result_label.setWordWrap(True)
        self.card_layout.addWidget(self.action_header)
        self.card_layout.addWidget(self.action_label)
        self.card_layout.addWidget(self.result_label)

        # Recent Transcript History
        self.history_header = QLabel("Recent Commands:", self.card)
        self.history_header.setObjectName("SectionHeader")
        self.history_label = QLabel("", self.card)
        self.history_label.setObjectName("HistoryItem")
        self.history_label.setWordWrap(True)
        self.history_header.hide()
        self.history_label.hide()
        self.card_layout.addWidget(self.history_header)
        self.card_layout.addWidget(self.history_label)

        # Bottom info: Queue indicator + Timing
        self.bottom_bar = QHBoxLayout()
        self.queue_label = QLabel("", self.card)
        self.queue_label.setObjectName("QueueBadge")
        self.queue_label.hide()
        self.timing_label = QLabel("", self.card)
        self.timing_label.setObjectName("TimingBadge")
        self.timing_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.bottom_bar.addWidget(self.queue_label)
        self.bottom_bar.addStretch()
        self.bottom_bar.addWidget(self.timing_label)
        self.card_layout.addLayout(self.bottom_bar)

        self.main_layout.addWidget(self.card)
        self.setFixedWidth(380)

    def _setup_timer(self) -> None:
        if QTimer is None:
            return
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide)

    def reposition(self) -> None:
        """Position the overlay on the selected screen corner."""
        screen = QGuiApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        margin = 24
        w = self.width()
        h = self.height()

        pos = self.config.position.lower()
        if pos == "top-left":
            x = geo.x() + margin
            y = geo.y() + margin
        elif pos == "bottom-left":
            x = geo.x() + margin
            y = geo.y() + geo.height() - h - margin
        elif pos == "bottom-right":
            x = geo.x() + geo.width() - w - margin
            y = geo.y() + geo.height() - h - margin
        else:  # top-right (default)
            x = geo.x() + geo.width() - w - margin
            y = geo.y() + margin

        self.move(x, y)

    def set_ptt_listening(self) -> None:
        """Show overlay in Push-To-Talk listening state."""
        self.hide_timer.stop()
        self.mode_badge.setText("PUSH TO TALK")
        self.status_badge.setText("RECORDING...")
        self.status_badge.setStyleSheet("background-color: #EF4444; color: white;")
        self.goal_label.setText("Listening...")
        self.goal_header.show()
        self.goal_label.show()
        self.live_label.hide()
        self.app_header.hide()
        self.app_label.hide()
        self.action_header.hide()
        self.action_label.hide()
        self.result_label.hide()
        self.timing_label.hide()
        self.reposition()
        self.show()

    def set_realtime_mode(self, active: bool = True) -> None:
        """Activate or deactivate Realtime mode on the overlay."""
        self.hide_timer.stop()
        if active:
            self.mode_badge.setText("REALTIME")
            self.status_badge.setText("LISTENING...")
            self.status_badge.setStyleSheet("background-color: #0284C7; color: white;")
            self.goal_label.setText("Waiting for voice commands...")
            self.goal_header.show()
            self.goal_label.show()
            self.reposition()
            self.show()
        else:
            self.mode_badge.setText("IDLE")
            self.status_badge.setText("READY")
            self.status_badge.setStyleSheet("background-color: #0284C7; color: white;")
            self.hide_timer.start(self.config.timeout_ms)

    def _add_history(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        if cleaned in self._recent_history:
            self._recent_history.remove(cleaned)
        self._recent_history.append(cleaned)
        if len(self._recent_history) > self._max_history:
            self._recent_history = self._recent_history[-self._max_history:]

        lines = [f"✓ {item}" for item in self._recent_history[:-1]]
        if self._recent_history:
            lines.append(f"→ {self._recent_history[-1]}")
        self.history_label.setText("\n".join(lines))
        self.history_header.show()
        self.history_label.show()

    def update_stage(self, stage_str: str, message: str, data: Dict[str, Any]) -> None:
        """Update overlay content based on pipeline and realtime stage transitions."""
        stage = PipelineStage(stage_str) if stage_str in [s.value for s in PipelineStage] else PipelineStage.IDLE

        if stage == PipelineStage.LISTENING:
            self.set_ptt_listening()
        elif stage == PipelineStage.REALTIME_STARTED:
            self.set_realtime_mode(True)
        elif stage == PipelineStage.REALTIME_STOPPED:
            self.set_realtime_mode(False)
        elif stage == PipelineStage.STT_PARTIAL:
            partial_text = data.get("text", message)
            if partial_text:
                self.live_label.setText(f"LIVE: {partial_text}...")
                self.live_label.show()
                self.reposition()
                self.show()
        elif stage == PipelineStage.STT_FINAL:
            final_text = data.get("text") or data.get("transcript", message)
            self.live_label.hide()
            self.goal_label.setText(final_text)
            self.goal_header.show()
            self.goal_label.show()
            self._add_history(final_text)
        elif stage == PipelineStage.COMMAND_QUEUED:
            qsize = data.get("queue_size", 1)
            self.queue_label.setText(f"Tasks queued: {qsize}")
            self.queue_label.show()
        elif stage == PipelineStage.COMMAND_STARTED:
            self.queue_label.setText("Executing...")
            self.queue_label.show()
            task_goal = data.get("goal", "")
            if task_goal:
                self.goal_label.setText(task_goal)
        elif stage == PipelineStage.TRANSCRIBING:
            self.status_badge.setText("TRANSCRIBING...")
            self.status_badge.setStyleSheet("background-color: #3B82F6; color: white;")
            text = data.get("text") or data.get("transcript", "")
            if text:
                self.goal_label.setText(text)
        elif stage == PipelineStage.OBSERVING:
            self.status_badge.setText("OBSERVING...")
            self.status_badge.setStyleSheet("background-color: #0284C7; color: white;")
            if "step" in data and "max_steps" in data:
                self.action_header.show()
                self.action_label.setText(f"Step {data['step']}/{data['max_steps']}: Observing desktop...")
                self.action_label.show()
        elif stage == PipelineStage.PLANNING:
            self.status_badge.setText("PLANNING...")
            self.status_badge.setStyleSheet("background-color: #8B5CF6; color: white;")
            app_str = data.get("active_app", "")
            if app_str:
                self.app_header.show()
                self.app_label.setText(app_str)
                self.app_label.show()
        elif stage == PipelineStage.INTERACTING:
            self.status_badge.setText("INTERACTING...")
            self.status_badge.setStyleSheet("background-color: #10B981; color: white;")
            step_str = f"Step {data.get('step', 1)}/{data.get('max_steps', 24)}: " if "step" in data else ""
            action_desc = data.get("label") or data.get("target_name") or data.get("action", "")
            self.action_header.show()
            self.action_label.setText(f"{step_str}{action_desc}")
            self.action_label.show()
        elif stage == PipelineStage.WAITING:
            self.status_badge.setText("WAITING...")
            self.status_badge.setStyleSheet("background-color: #F59E0B; color: white;")
            self.action_header.show()
            self.action_label.setText("Waiting for UI to settle...")
            self.action_label.show()
        elif stage in (PipelineStage.BLOCKED, PipelineStage.COMMAND_FAILED):
            self.status_badge.setText("BLOCKED" if stage == PipelineStage.BLOCKED else "FAILED")
            self.status_badge.setStyleSheet("background-color: #DC2626; color: white;")
            err = data.get("error", message)
            self.result_label.setText(f"Result: {err}")
            self.result_label.show()
            if self.mode_badge.text() != "REALTIME":
                self.hide_timer.start(self.config.timeout_ms)
        elif stage in (PipelineStage.DONE, PipelineStage.COMMAND_COMPLETED):
            self.status_badge.setText("DONE")
            self.status_badge.setStyleSheet("background-color: #059669; color: white;")
            summary = data.get("summary", message)
            self.result_label.setText(f"Done: {summary}")
            self.result_label.show()
            if "timings" in data and "total" in data["timings"]:
                self.timing_label.setText(f"Total: {data['timings']['total']}s")
                self.timing_label.show()
            if self.mode_badge.text() != "REALTIME":
                self.hide_timer.start(self.config.timeout_ms)
        elif stage == PipelineStage.ERROR:
            self.status_badge.setText("ERROR")
            self.status_badge.setStyleSheet("background-color: #DC2626; color: white;")
            err = data.get("error", message)
            self.result_label.setText(f"Error: {err}")
            self.result_label.show()
            if self.mode_badge.text() != "REALTIME":
                self.hide_timer.start(self.config.timeout_ms)

        self.adjustSize()
        self.reposition()
