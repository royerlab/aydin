#!/usr/bin/env python3
"""Build Aydin Studio native installers using conda-constructor.

Generates a construct.yaml from configuration and invokes ``constructor``
to produce a platform-native installer:
  - Linux:  .sh   (shell script installer)
  - macOS:  .pkg  (native GUI installer)
  - Windows: .exe (NSIS-based GUI installer)

Requires the ``aydin-build-installer`` conda environment
(see packaging/environments/build_installer.yml).

Usage:
    # Build installer for current platform
    python packaging/build_installer.py

    # Dry run (print construct.yaml without building)
    python packaging/build_installer.py --dry-run

    # Custom channels and output directory
    python packaging/build_installer.py \\
        --channels conda-forge royerlab --output-dir _work
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

try:
    from ruamel.yaml import YAML
except ImportError:
    YAML = None

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGING_DIR = Path(__file__).resolve().parent

# Default Python version for the installer environment
DEFAULT_PYTHON_VERSION = "3.11"

# Default conda channels (order matters: first has highest priority)
DEFAULT_CHANNELS = ["conda-forge"]


def _get_version() -> str:
    """Read the Aydin version from src/aydin/__init__.py."""
    init_file = REPO_ROOT / "src" / "aydin" / "__init__.py"
    text = init_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        raise RuntimeError(f"Could not find __version__ in {init_file}")
    return match.group(1)


def _detect_platform() -> str:
    """Detect the current platform as a conda-style platform string."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        return "linux-64" if machine in ("x86_64", "amd64") else f"linux-{machine}"
    elif system == "darwin":
        return "osx-arm64" if machine == "arm64" else "osx-64"
    elif system == "windows":
        return "win-64"
    else:
        raise RuntimeError(f"Unsupported platform: {system}-{machine}")


def _installer_filename(version: str, platform_str: str) -> str:
    """Generate the installer output filename."""
    ext_map = {
        "linux": "sh",
        "osx": "pkg",
        "win": "exe",
    }
    os_key = platform_str.split("-")[0]
    ext = ext_map.get(os_key, "sh")
    return f"AydinStudio-{version}-{platform_str}.{ext}"


def _build_definitions(
    version: str,
    platform_str: str,
    channels: list,
    python_version: str,
) -> dict:
    """Build the construct.yaml definitions dictionary."""
    os_key = platform_str.split("-")[0]

    definitions = {
        "name": "AydinStudio",
        "version": version,
        "company": "Royer Lab",
        "reverse_domain_identifier": "org.royerlab.aydin",
        "installer_filename": _installer_filename(version, platform_str),
        "channels": channels,
        "specs": [
            f"python={python_version}.*",
            "aydin",
            "aydin-menu",
            # Explicit core deps for solver reliability:
            "numpy",
            "scipy",
            "scikit-image",
            "numba",
            "pytorch-cpu",  # CPU-only PyTorch from conda-forge
            "catboost",
            "lightgbm",
            "napari",
            "pyqt6",
            "qtpy",
            "qdarkstyle",
        ],
        "menu_packages": ["aydin-menu"],
        "initialize_by_default": False,
        "register_python_default": False,
        "register_envs": False,
        "condarc": {
            "channels": channels,
            "channel_priority": "strict",
        },
    }

    # License file (will be copied to output_dir by _write_construct_yaml)
    rtf_license = PACKAGING_DIR / "LICENSE.rtf"
    txt_license = REPO_ROOT / "LICENSE.txt"
    if os_key in ("osx", "win") and rtf_license.exists():
        definitions["_license_src"] = rtf_license
        definitions["license_file"] = "LICENSE.rtf"
    else:
        definitions["_license_src"] = txt_license
        definitions["license_file"] = "LICENSE.txt"

    # Platform-specific settings
    post_install = None
    if os_key == "linux":
        definitions["installer_type"] = "sh"
        definitions["default_prefix"] = "$HOME/.local/aydin"
        post_install = PACKAGING_DIR / "post_install" / "post_install.sh"
    elif os_key == "osx":
        definitions["installer_type"] = "pkg"
        definitions["pkg_name"] = "org.royerlab.aydin"
        definitions["default_location_pkg"] = "/Library"
        post_install = PACKAGING_DIR / "post_install" / "post_install.sh"
        # Signing (enabled via env vars when certificates are available)
        signing_id = os.environ.get("CONSTRUCTOR_SIGNING_IDENTITY")
        if signing_id:
            definitions["signing_identity_name"] = signing_id
        notarization_id = os.environ.get("CONSTRUCTOR_NOTARIZATION_IDENTITY")
        if notarization_id:
            definitions["notarization_identity_name"] = notarization_id
    elif os_key == "win":
        definitions["installer_type"] = "exe"
        definitions["default_prefix"] = "%LOCALAPPDATA%\\aydin"
        icon_path = PACKAGING_DIR / "icons" / "aydin_icon.ico"
        if icon_path.exists():
            definitions["_icon_src"] = icon_path
            definitions["icon_image"] = "aydin_icon.ico"
        post_install = PACKAGING_DIR / "post_install" / "post_install.bat"
        # Signing (enabled via env var when certificate is available)
        cert = os.environ.get("CONSTRUCTOR_SIGNING_CERTIFICATE")
        if cert:
            definitions["signing_certificate"] = cert

    if post_install and post_install.exists():
        definitions["_post_install_src"] = post_install
        definitions["post_install"] = post_install.name

    return definitions


def _write_construct_yaml(definitions: dict, output_dir: Path) -> Path:
    """Write construct.yaml to the output directory.

    Copies referenced files (license, post_install, icon) into the output
    directory so that construct.yaml can use relative paths. Internal keys
    prefixed with ``_`` are stripped before writing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy referenced files into the output directory
    for key in ("_license_src", "_post_install_src", "_icon_src"):
        src = definitions.pop(key, None)
        if src and Path(src).exists():
            shutil.copy2(str(src), str(output_dir / Path(src).name))

    construct_path = output_dir / "construct.yaml"

    if YAML is not None:
        yaml = YAML()
        yaml.default_flow_style = False
        with open(construct_path, "w") as f:
            yaml.dump(definitions, f)
    else:
        # Fallback: simple YAML writing without ruamel
        import json

        # Convert to YAML-compatible format via JSON round-trip
        with open(construct_path, "w") as f:
            for key, value in definitions.items():
                if isinstance(value, list):
                    f.write(f"{key}:\n")
                    for item in value:
                        f.write(f"  - {item}\n")
                elif isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        if isinstance(v, list):
                            f.write(f"  {k}:\n")
                            for item in v:
                                f.write(f"    - {item}\n")
                        else:
                            f.write(f"  {k}: {json.dumps(v)}\n")
                elif isinstance(value, bool):
                    f.write(f"{key}: {'true' if value else 'false'}\n")
                else:
                    f.write(f"{key}: {value}\n")

    return construct_path


def _run_constructor(output_dir: Path, platform_str: str = None) -> None:
    """Invoke constructor to build the installer."""
    if not shutil.which("constructor"):
        raise RuntimeError(
            "constructor not found. Install it with:\n"
            "  conda env create -f packaging/environments/build_installer.yml\n"
            "  conda activate aydin-build-installer"
        )

    cmd = ["constructor", str(output_dir)]
    if platform_str:
        cmd.extend(["--platform", platform_str])

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"constructor failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Aydin Studio native installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                              # Build for current platform\n"
            "  %(prog)s --dry-run                     # Print construct.yaml only\n"
            "  %(prog)s --channels conda-forge        # Use only conda-forge\n"
            "  %(prog)s --platform linux-64           # Cross-compile target\n"
        ),
    )
    parser.add_argument(
        "--python-version",
        default=DEFAULT_PYTHON_VERSION,
        help=f"Python version for the installer (default: {DEFAULT_PYTHON_VERSION})",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=DEFAULT_CHANNELS,
        help="Conda channels in priority order (default: conda-forge)",
    )
    parser.add_argument(
        "--output-dir",
        default="_work",
        help="Output directory for construct.yaml and installer (default: _work)",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Target platform (default: auto-detect). "
        "Options: linux-64, osx-arm64, osx-64, win-64",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate construct.yaml and print it, but do not run constructor",
    )
    args = parser.parse_args()

    version = _get_version()
    platform_str = args.platform or _detect_platform()
    output_dir = Path(args.output_dir)

    print(f"Aydin version:  {version}")
    print(f"Target platform: {platform_str}")
    print(f"Python version:  {args.python_version}")
    print(f"Channels:        {args.channels}")
    print(f"Output dir:      {output_dir}")

    definitions = _build_definitions(
        version=version,
        platform_str=platform_str,
        channels=args.channels,
        python_version=args.python_version,
    )

    construct_path = _write_construct_yaml(definitions, output_dir)
    print(f"\nGenerated: {construct_path}")

    # Print the construct.yaml contents
    print("\n--- construct.yaml ---")
    print(construct_path.read_text())
    print("--- end ---\n")

    if args.dry_run:
        print("Dry run complete. Skipping constructor invocation.")
        return

    _run_constructor(output_dir, platform_str)

    # List output files
    print("\nBuild artifacts:")
    for f in sorted(output_dir.iterdir()):
        if f.name != "construct.yaml":
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
