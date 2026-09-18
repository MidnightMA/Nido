"""Tests for Nido CLI diagnostic subcommands."""

import contextlib
import io
import sys
from nido.cli import main


def test_cli_version() -> None:
    buf = io.StringIO()
    exited = False
    with contextlib.redirect_stdout(buf):
        try:
            main(["--version"])
        except SystemExit as e:
            exited = True
            assert e.code == 0
    assert exited
    assert "nido" in buf.getvalue()


def test_cli_tools_list() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["tools", "list"])
    assert code == 0
    out = buf.getvalue()
    assert "Registered Tools" in out
    assert "open_app" in out
    assert "open_url" in out


def test_cli_models_status() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["models", "status"])
    assert code in (0, 1)
    out = buf.getvalue()
    assert "Nido Offline Models Status" in out


def test_cli_test_translate() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["test-translate", "کروم را باز کن"])
    assert code == 0
    out = buf.getvalue()
    assert "Persian:" in out
    assert "Translated:" in out


def test_cli_test_command() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["test-command", "open firefox"])
    assert code == 0
    out = buf.getvalue()
    assert "Selected Tools" in out
    assert "open_app" in out
