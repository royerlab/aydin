#!/usr/bin/env python
"""Generate napari plugin screenshots for documentation.

Run from any directory:
    python docs/source/_scripts/generate_napari_screenshots.py

Requires a display (macOS/Linux with X11/Wayland, or Windows).
On headless machines, this script exits gracefully and the docs build
proceeds with existing screenshots.
"""

import os
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'resources' / 'napari_tutorials'


def _has_display():
    """Check if a display is available for Qt/napari."""
    # macOS and Windows always have a display when logged in
    if os.name == 'nt' or os.uname().sysname == 'Darwin':
        return True
    # On Linux, check for DISPLAY or WAYLAND_DISPLAY
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def _normalise(image):
    import numpy

    image = image.astype(numpy.float32)
    lo, hi = image.min(), image.max()
    if hi - lo > 0:
        image = (image - lo) / (hi - lo)
    return image


def generate_widget_screenshot():
    """Generate a screenshot of the denoising widget with a sample image."""
    import napari
    from qtpy.QtCore import QTimer
    from skimage.data import camera

    from aydin.io.datasets import add_noise

    clean = _normalise(camera())
    noisy = add_noise(clean, intensity=10, variance=0.005, seed=0)

    viewer = napari.Viewer(show=True)
    viewer.add_image(noisy, name='Noisy Camera (2D)')

    from aydin.napari_plugin._widget import AydinDenoiseWidget

    widget = AydinDenoiseWidget(viewer)
    viewer.window.add_dock_widget(widget, name='Aydin Denoising', area='right')

    viewer.window._qt_window.resize(1100, 700)

    def _capture():
        time.sleep(0.5)
        pixmap = viewer.window._qt_window.grab()
        output_path = OUTPUT_DIR / 'napari_denoising_widget.png'
        pixmap.save(str(output_path))
        print(f'Saved: {output_path}')
        viewer.close()

    QTimer.singleShot(2000, _capture)
    napari.run()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _has_display():
        existing = list(OUTPUT_DIR.glob('*.png'))
        if existing:
            print(
                f'No display available — skipping screenshot generation. '
                f'Using {len(existing)} existing screenshot(s).'
            )
        else:
            print(
                'WARNING: No display available and no existing screenshots found. '
                'The napari tutorial images will be missing from the docs build. '
                'Run this script on a machine with a display to generate them.'
            )
        return

    print(f'Output directory: {OUTPUT_DIR}')

    print('\n--- Generating denoising widget screenshot ---')
    try:
        generate_widget_screenshot()
    except Exception as exc:
        print(f'WARNING: Screenshot generation failed: {exc}')
        print('Docs build will proceed with existing screenshots (if any).')
        return

    print('\nDone! Screenshots saved to:')
    for p in sorted(OUTPUT_DIR.glob('*.png')):
        print(f'  {p}')


if __name__ == '__main__':
    main()
