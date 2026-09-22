"""Desktop automation and interaction package."""

from __future__ import annotations

from nido.desktop.agent import DesktopAgent, DesktopAgentResult
from nido.desktop.candidates import ActionCandidate, CandidateBuilder
from nido.desktop.input import InputBackend, detect_input_backend
from nido.desktop.tool_builder import DynamicDesktopToolBuilder

__all__ = [
    "DesktopAgent",
    "DesktopAgentResult",
    "ActionCandidate",
    "CandidateBuilder",
    "InputBackend",
    "detect_input_backend",
    "DynamicDesktopToolBuilder",
]
