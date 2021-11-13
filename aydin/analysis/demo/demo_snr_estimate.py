import pytest
from numpy.random.mtrand import normal

from aydin.analysis.snr_estimate import snr_estimate
from aydin.io.datasets import camera, normalise
from aydin.util.log.log import lprint, Log


def demo_snr_estimate(display: bool = False):
    Log.enable_output = True

    clean_image = normalise(camera())

    noise = normal(size=clean_image.size).reshape(*clean_image.shape)

    noisy_image_1 = clean_image + 100 * noise
    noisy_image_2 = clean_image + 10 * noise
    noisy_image_3 = clean_image + noise
    noisy_image_4 = clean_image + 0.1 * noise
    noisy_image_5 = clean_image + 0.01 * noise

    noise1_dB = snr_estimate(noisy_image_1)
    noise2_dB = snr_estimate(noisy_image_2)
    noise3_dB = snr_estimate(noisy_image_3)
    noise4_dB = snr_estimate(noisy_image_4)
    noise5_dB = snr_estimate(noisy_image_5)

    lprint(f"noise1_dB={noise1_dB}")
    lprint(f"noise2_dB={noise2_dB}")
    lprint(f"noise3_dB={noise3_dB}")
    lprint(f"noise4_dB={noise4_dB}")
    lprint(f"noise5_dB={noise5_dB}")

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(clean_image, name='clean_image')
            viewer.add_image(noisy_image_1, name='noisy_image_1')
            viewer.add_image(noisy_image_2, name='noisy_image_2')
            viewer.add_image(noisy_image_3, name='noisy_image_3')
            viewer.add_image(noisy_image_4, name='noisy_image_4')
            viewer.add_image(noisy_image_5, name='noisy_image_5')

    assert noise1_dB == pytest.approx(-42, 1, 3)
    assert noise2_dB == pytest.approx(-20, 1, 1)
    assert noise3_dB == pytest.approx(-0, 1, 1)
    assert noise4_dB == pytest.approx(19, 1, 1)
    assert noise5_dB == pytest.approx(33, 1, 1)


demo_snr_estimate()
