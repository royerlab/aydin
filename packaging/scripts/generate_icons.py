#!/usr/bin/env python3
"""Generate .ico and .icns icon files from the source PNG.

Requires: Pillow (pip install Pillow)
On macOS: uses iconutil (ships with Xcode Command Line Tools) for best .icns
On other platforms: uses Pillow's built-in .icns support (limited sizes)

Usage:
    python packaging/scripts/generate_icons.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PNG = REPO_ROOT / "src" / "aydin" / "gui" / "resources" / "aydin_icon.png"
OUT_DIR = REPO_ROOT / "packaging" / "icons"

# ICO sizes (Windows supports up to 256x256)
ICO_SIZES = [16, 32, 48, 64, 128, 256]

# ICNS sizes required by iconutil
ICNS_SIZES = {
    "icon_16x16.png": 16,
    "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32,
    "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256,
}


def generate_ico(src: Path, out_dir: Path) -> None:
    """Generate multi-size .ico file for Windows."""
    img = Image.open(src).convert("RGBA")
    sizes = [(s, s) for s in ICO_SIZES]
    out_path = out_dir / "aydin_icon.ico"
    img.save(str(out_path), format="ICO", sizes=sizes)
    print(f"  Generated {out_path} ({len(sizes)} sizes)")


def generate_icns_iconutil(src: Path, out_dir: Path) -> bool:
    """Generate .icns using macOS iconutil (best quality)."""
    if not shutil.which("iconutil"):
        return False

    img = Image.open(src).convert("RGBA")
    with tempfile.TemporaryDirectory() as tmpdir:
        iconset = Path(tmpdir) / "aydin.iconset"
        iconset.mkdir()

        for name, size in ICNS_SIZES.items():
            resized = img.resize((size, size), Image.LANCZOS)
            resized.save(str(iconset / name), format="PNG")

        out_path = out_dir / "aydin_icon.icns"
        result = subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(out_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  Generated {out_path} (via iconutil)")
            return True
        print(f"  Warning: iconutil failed: {result.stderr.strip()}")
    return False


def generate_icns_pillow(src: Path, out_dir: Path) -> None:
    """Generate .icns using Pillow (fallback for non-macOS).

    Pillow's ICNS writer saves the image at its current size.
    We resize to 256x256 which is the standard macOS app icon size.
    For full multi-resolution .icns, use iconutil on macOS instead.
    """
    img = Image.open(src).convert("RGBA")
    # Resize to 256x256 — the primary icon size for macOS
    img = img.resize((256, 256), Image.LANCZOS)
    out_path = out_dir / "aydin_icon.icns"
    img.save(str(out_path), format="ICNS")
    print(f"  Generated {out_path} (via Pillow, 256x256 only)")


def copy_png(src: Path, out_dir: Path) -> None:
    """Copy source PNG to icons directory."""
    out_path = out_dir / "aydin_icon.png"
    shutil.copy2(str(src), str(out_path))
    print(f"  Copied {out_path}")


def main():
    if not SRC_PNG.exists():
        print(f"Error: Source PNG not found at {SRC_PNG}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating icons from {SRC_PNG}")

    copy_png(SRC_PNG, OUT_DIR)
    generate_ico(SRC_PNG, OUT_DIR)

    # Try iconutil first (macOS), fall back to Pillow
    if not generate_icns_iconutil(SRC_PNG, OUT_DIR):
        generate_icns_pillow(SRC_PNG, OUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
