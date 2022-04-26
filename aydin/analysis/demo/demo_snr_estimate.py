# flake8: noqa
# import pytest
import pytest
from numpy.random.mtrand import normal

from aydin.analysis.snr_estimate import snr_estimate
from aydin.io.datasets import camera, normalise, newyork
from aydin.util.log.log import lprint, Log


def demo_snr_estimate(display: bool = True):
    Log.enable_output = True

    clean_image = normalise(camera())

    noise = normal(size=clean_image.size).reshape(*clean_image.shape)

    noisy_image_1 = clean_image + 100 * noise
    noisy_image_2 = clean_image + 10 * noise
    noisy_image_3 = clean_image + noise
    noisy_image_4 = clean_image + 0.1 * noise
    noisy_image_5 = clean_image + 0.01 * noise

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(clean_image, name='clean_image')
        viewer.add_image(noisy_image_1, name='noisy_image_1')
        viewer.add_image(noisy_image_2, name='noisy_image_2')
        viewer.add_image(noisy_image_3, name='noisy_image_3')
        viewer.add_image(noisy_image_4, name='noisy_image_4')
        viewer.add_image(noisy_image_5, name='noisy_image_5')
        napari.run()

    noise1_dB = snr_estimate(noisy_image_1)
    noise2_dB = snr_estimate(noisy_image_2)
    noise3_dB = snr_estimate(noisy_image_3)
    noise4_dB = snr_estimate(noisy_image_4)
    noise5_dB = snr_estimate(noisy_image_5)
    clean_dB = snr_estimate(clean_image)

    lprint(f"noise1_dB={noise1_dB}")
    lprint(f"noise2_dB={noise2_dB}")
    lprint(f"noise3_dB={noise3_dB}")
    lprint(f"noise4_dB={noise4_dB}")
    lprint(f"noise5_dB={noise5_dB}")
    lprint(f"clean_dB ={clean_dB}")

    assert noise1_dB == pytest.approx(-41.0, 1, 5)
    assert noise2_dB == pytest.approx(-31.1, 1, 1)
    assert noise3_dB == pytest.approx(-11.0, 1, 1)
    assert noise4_dB == pytest.approx(8.6, 1, 1)
    assert noise5_dB == pytest.approx(26.8, 1, 1)
    assert clean_dB == pytest.approx(31.1, 1, 1)


if __name__ == "__main__":
    demo_snr_estimate()
