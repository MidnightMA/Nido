#!/usr/bin/env python3
"""Installs user-level systemd service and KDE/XDG desktop autostart entry for Nido."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_nido_executable(repo_root: Path) -> str:
    """Find the absolute path to the nido executable."""
    # 1. Check current python environment (e.g. .venv/bin/nido)
    curr_nido = Path(sys.executable).parent / "nido"
    if curr_nido.is_file() and os.access(curr_nido, os.X_OK):
        return str(curr_nido)

    # 2. Check repo .venv
    venv_nido = repo_root / ".venv" / "bin" / "nido"
    if venv_nido.is_file() and os.access(venv_nido, os.X_OK):
        return str(venv_nido)

    # 3. Check ~/.local/bin/nido
    local_nido = Path.home() / ".local" / "bin" / "nido"
    if local_nido.is_file() and os.access(local_nido, os.X_OK):
        return str(local_nido)

    # 4. Check PATH
    which_nido = shutil.which("nido")
    if which_nido:
        return which_nido

    return str(Path.home() / ".local" / "bin" / "nido")


def install_systemd(repo_root: Path) -> bool:
    systemd_user_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_user_dir.mkdir(parents=True, exist_ok=True)

    src_service = repo_root / "systemd" / "nido.service"
    dest_service = systemd_user_dir / "nido.service"

    if not src_service.is_file():
        print(f"Error: {src_service} not found.", file=sys.stderr)
        return False

    nido_exec = find_nido_executable(repo_root)
    print(f"Configuring systemd service with executable: {nido_exec}")

    content = src_service.read_text(encoding="utf-8")
    content = content.replace("%h/.local/bin/nido", nido_exec)

    dest_service.write_text(content, encoding="utf-8")
    print(f"Installed systemd user service: {dest_service}")

    # Reload systemd user daemon
    if shutil.which("systemctl"):
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        print("Run the following to restart Nido service:")
        print("  systemctl --user restart nido.service")
        print("  systemctl --user status nido.service")
    return True


def install_desktop_autostart(repo_root: Path) -> bool:
    autostart_dir = Path.home() / ".config" / "autostart"
    autostart_dir.mkdir(parents=True, exist_ok=True)

    src_desktop = repo_root / "desktop" / "nido.desktop"
    dest_desktop = autostart_dir / "nido.desktop"

    if not src_desktop.is_file():
        print(f"Error: {src_desktop} not found.", file=sys.stderr)
        return False

    nido_exec = find_nido_executable(repo_root)
    content = src_desktop.read_text(encoding="utf-8")
    content = content.replace("Exec=nido", f"Exec={nido_exec}")

    dest_desktop.write_text(content, encoding="utf-8")
    print(f"Installed XDG/KDE autostart desktop entry: {dest_desktop}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Nido autostart integrations.")
    parser.add_argument(
        "--type",
        choices=["systemd", "desktop", "all"],
        default="all",
        help="Autostart mechanism to install (default: all).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    ok = True
    if args.type in ("systemd", "all"):
        ok = install_systemd(repo_root) and ok

    if args.type in ("desktop", "all"):
        ok = install_desktop_autostart(repo_root) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
