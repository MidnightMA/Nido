#!/usr/bin/env python3
"""Installs user-level systemd service and KDE/XDG desktop autostart entry for Nido."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def install_systemd(repo_root: Path) -> bool:
    systemd_user_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_user_dir.mkdir(parents=True, exist_ok=True)

    src_service = repo_root / "systemd" / "nido.service"
    dest_service = systemd_user_dir / "nido.service"

    if not src_service.is_file():
        print(f"Error: {src_service} not found.", file=sys.stderr)
        return False

    shutil.copy(src_service, dest_service)
    print(f"Installed systemd user service: {dest_service}")

    # Reload systemd user daemon
    if shutil.which("systemctl"):
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        print("Run the following to enable and start Nido on login:")
        print("  systemctl --user enable nido.service")
        print("  systemctl --user start nido.service")
    return True


def install_desktop_autostart(repo_root: Path) -> bool:
    autostart_dir = Path.home() / ".config" / "autostart"
    autostart_dir.mkdir(parents=True, exist_ok=True)

    src_desktop = repo_root / "desktop" / "nido.desktop"
    dest_desktop = autostart_dir / "nido.desktop"

    if not src_desktop.is_file():
        print(f"Error: {src_desktop} not found.", file=sys.stderr)
        return False

    shutil.copy(src_desktop, dest_desktop)
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
