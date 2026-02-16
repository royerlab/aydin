"""Demo script for simulating camera images with varying exposure times."""

# flake8: noqa
import numpy

from aydin.analysis.camera_simulation import simulate_camera_image
from aydin.analysis.resolution_estimate import resolution_estimate
from aydin.analysis.snr_estimate import snr_estimate
from aydin.io.datasets import camera
from aydin.util.log.log import Log, aprint


def demo_camera_simulation():
    """Simulate a noisy video with varying exposure and estimate resolution and SNR."""
    Log.enable_output = True

    clean_image = camera()
    noisy_video = numpy.stack(
        (
            simulate_camera_image(clean_image, exposure_time_s=e / 32)
            for e in range(1, 32)
        )
    )

    fc, _ = resolution_estimate(noisy_video[16])
    snr = snr_estimate(noisy_video[16])

    aprint(f"Resolution: {fc}, snr: {snr}")

    import napari

    viewer = napari.Viewer()
    viewer.add_image(clean_image, name='clean_image')
    viewer.add_image(noisy_video, name='noisy_video')
    napari.run()


def demo_camera_simulation_video():
    """Simulate a noisy video from repeated low-exposure camera captures."""
    clean_image = camera()
    noisy_video = numpy.stack(
        (simulate_camera_image(clean_image // 32) for _ in range(64))
    )

    import napari

    viewer = napari.Viewer()
    viewer.add_image(clean_image, name='clean_image')
    viewer.add_image(noisy_video, name='noisy_video')
    napari.run()


if __name__ == "__main__":
    demo_camera_simulation()
    demo_camera_simulation_video()
