import numpy
from scipy.ndimage import gaussian_filter

from aydin.analysis.resolution_estimate import resolution_estimate
from aydin.io.datasets import normalise, add_noise, cropped_newyork
from aydin.util.log.log import Log


def demo_resolution_estimate(display: bool = True):
    Log.enable_output = True

    image = cropped_newyork()[:-3, 0:-7]
    high_res_image = normalise(image.astype(numpy.float32))
    low_res_image = gaussian_filter(high_res_image, sigma=3)
    high_res_noisy = add_noise(high_res_image)
    low_res_noisy = add_noise(low_res_image)

    fc_high_res, _ = resolution_estimate(high_res_image, display_images=display)
    fc_low_res, _ = resolution_estimate(low_res_image, display_images=display)
    fc_high_res_noisy, _ = resolution_estimate(high_res_noisy, display_images=display)
    fc_low_res_noisy, _ = resolution_estimate(low_res_noisy, display_images=display)

    print(f"fc_high_res={fc_high_res}")
    print(f"fc_low_res={fc_low_res}")
    print(f"fc_high_res_noisy={fc_high_res_noisy}")
    print(f"fc_low_res_noisy={fc_low_res_noisy}")

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='clean_image')
            viewer.add_image(high_res_image, name='high_res_image')
            viewer.add_image(low_res_image, name='low_res_image')
            viewer.add_image(high_res_noisy, name='high_res_noisy')
            viewer.add_image(low_res_noisy, name='low_res_noisy')

    assert fc_low_res < 0.6 * fc_high_res
    assert fc_low_res_noisy < 0.6 * fc_high_res_noisy


if __name__ == "__main__":
    demo_resolution_estimate()
