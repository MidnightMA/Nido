"""Unit tests for Laya-MLX integration, mock agent, and decision formatting."""

from nido.config import LayaConfig
from nido.desktop.candidates import ActionCandidate
from nido.laya.agent import (
    ActionDecision,
    LayaDecisionAgent,
    MockLayaAgent,
    detect_mlx_device,
)


def test_detect_mlx_device() -> None:
    # Test auto detection returns a valid device string
    dev = detect_mlx_device("auto")
    assert dev in ("cpu", "gpu")

    # Explicit preference
    assert detect_mlx_device("cpu") == "cpu"
    assert detect_mlx_device("gpu") == "gpu"


def test_mock_laya_agent_text_entry_selection() -> None:
    agent = MockLayaAgent()
    candidates = [
        ActionCandidate(id="A1", label='Activate [e1] button "New"', action_type="activate_ui_element"),
        ActionCandidate(
            id="A2",
            label='Set requested text in [e2] text field "Editor"',
            action_type="set_ui_text",
            arguments={"element_id": "e2", "text": "Hello world"},
        ),
        ActionCandidate(id="A3", label="Done", action_type="done"),
    ]

    state_text = (
        "USER GOAL\nOpen notes and write Hello world\n\n"
        "CURRENT STEP\n2 / 24\n\n"
        "AVAILABLE ACTIONS\n[A1] Activate [e1] button \"New\"\n"
        "[A2] Set requested text in [e2] text field \"Editor\"\n"
        "[A3] Done"
    )

    decision = agent.predict_action(state_text, candidates)
    assert decision.selected_id == "A2"
    assert decision.confidence > 0.8
    assert "A2" in decision.probabilities


def test_mock_laya_agent_app_launch_selection() -> None:
    agent = MockLayaAgent()
    candidates = [
        ActionCandidate(id="A1", label="Open application: kate", action_type="open_app", arguments={"app_name": "kate"}),
        ActionCandidate(id="A2", label="Open application: dolphin", action_type="open_app", arguments={"app_name": "dolphin"}),
        ActionCandidate(id="A3", label="Done", action_type="done"),
    ]

    state_text = (
        "USER GOAL\nOpen kate\n\n"
        "CURRENT STEP\n1 / 24\n\n"
        "AVAILABLE ACTIONS\n[A1] Open application: kate\n[A2] Open application: dolphin\n[A3] Done"
    )

    decision = agent.predict_action(state_text, candidates)
    assert decision.selected_id == "A1"
    assert decision.confidence > 0.8


def test_mock_laya_agent_calculation_selection() -> None:
    agent = MockLayaAgent()
    candidates = [
        ActionCandidate(id="A1", label='Activate [e1] button "Equal"', action_type="activate_ui_element"),
        ActionCandidate(
            id="A2",
            label='Set requested text in [e2] text field "Display"',
            action_type="set_ui_text",
            arguments={"element_id": "e2", "text": "2 * 95"},
        ),
        ActionCandidate(id="A3", label="Done", action_type="done"),
    ]

    state_text = (
        "USER GOAL\nopen calculator and calculate 2*95\n\n"
        "CURRENT STEP\n2 / 24\n\n"
        "ACTION HISTORY\n1. Open application: kcalc -> success\n\n"
        "AVAILABLE ACTIONS\n[A1] Activate [e1] button \"Equal\"\n"
        "[A2] Set requested text in [e2] text field \"Display\"\n"
        "[A3] Done"
    )

    decision = agent.predict_action(state_text, candidates)
    assert decision.selected_id == "A2"
    assert decision.confidence > 0.8


def test_mock_laya_agent_done_after_action() -> None:
    agent = MockLayaAgent()
    candidates = [
        ActionCandidate(id="A1", label='Activate [e1] button "Save"', action_type="activate_ui_element"),
        ActionCandidate(id="A2", label="Done", action_type="done"),
    ]

    # History indicates an action succeeded
    state_text = (
        "USER GOAL\nOpen kate\n\n"
        "CURRENT STEP\n2 / 24\n\n"
        "ACTION HISTORY\n1. Open application: kate -> success\n\n"
        "AVAILABLE ACTIONS\n[A1] Activate [e1] button \"Save\"\n[A2] Done"
    )

    decision = agent.predict_action(state_text, candidates)
    assert decision.selected_id == "A2"


def test_laya_decision_agent_missing_dir_safe_fallback() -> None:
    # When checkpoint directory does not exist, agent falls back gracefully to MockLayaAgent
    cfg = LayaConfig(model_dir="/non/existent/model/path")
    agent = LayaDecisionAgent(cfg)
    assert agent.is_loaded is False

    candidates = [
        ActionCandidate(id="A1", label="Open Kate", action_type="open_app", arguments={"app_name": "kate"}),
        ActionCandidate(id="A2", label="Done", action_type="done"),
    ]
    state_text = "USER GOAL\nOpen kate\n\nCURRENT STEP\n1 / 24\n\nAVAILABLE ACTIONS\n[A1] Open Kate\n[A2] Done"

    dec = agent.predict_action(state_text, candidates)
    assert dec.selected_id in ("A1", "A2")
    assert dec.confidence > 0.0


def test_laya_decision_agent_parse_result() -> None:
    cfg = LayaConfig()
    agent = LayaDecisionAgent(cfg)

    candidates = [
        ActionCandidate(id="A1", label="Action 1", action_type="test1"),
        ActionCandidate(id="A2", label="Action 2", action_type="test2"),
    ]

    # Test dictionary with choice and probabilities
    mock_resp = {
        "next_action": {
            "choice": "A2",
            "confidence": 0.88,
            "probabilities": {"A1": 0.12, "A2": 0.88},
        }
    }
    decision = agent._parse_result(mock_resp, candidates, elapsed_ms=15.0)
    assert decision.selected_id == "A2"
    assert decision.confidence == 0.88
    assert decision.probabilities["A2"] == 0.88
    assert decision.inference_time_ms == 15.0
