# flake8: noqa
from aydin.util.string.break_text import break_text

_test_line_breaks_text = """    Suppresses fixed, axis aligned, offset patterns along any combination of
    axis. Given a list of lists of axis that defines axis-aligned volumes,
    intensity fluctuations of these volumes are stabilised. You can suppress
    intensity fluctuation over time, suppress fixed offsets per pixel over
    time, suppress intensity fluctuations per row, per column, and more...

    For example, assume an image with dimensions tyx (t+2D), and you want to
    suppress fluctuations of intensity along the t axis, then you provide:
    axes=[[0]] (or simply 0 or [0]) which means that the average intensity
    for all planes along t (axis=0) will be stabilised. If instead you want
    to suppress some fixed background offset over xy planes, then you do:
    axes=[[1,2]]. If you want to do both, then you use: axes=[[0], [1,
    2]]. Please note that these corrections are applied in the order
    specified by the list of axis combinations. It is not recommended to
    reapply the pattern after denoising, unless the pattern itself is of
    value and is not considered noise.
    
        For images with little noise, applying a high-pass filter can help denoise the image by removing some of the
    image complexity. The low-frequency parts of the image do not need to be denoised because sometimes the challenge
    is disentangling the (high-frequency) noise from the high-frequencies in the image. The scale parameter must be
    chosen with care. The lesser the noise, the smaller the value. Values around 1 work well but must be tuned
    depending on the image. If the scale parameter is too low, some noise might be left untouched. The best is to
    keep the parameter as low as possible while still achieving good denoising performance. It is also possible to
    apply median filtering when computing the low-pass image which helps reducing the impact of outlier voxel values,
    for example salt&pepper noise. Note: when median filtering is on, larger values of sigma (e.g. >= 1) are
    recommended, unless when the level of noise is very low in which case a sigma of 0 (no Gaussian blur) may be
    advantageous. To recover the original denoised image the filtering is undone during post-processing. Note: this
    is ideal for treating <a href='https://en.wikipedia.org/wiki/Colors_of_noise'>'blue' noise</a> that is
    characterised by a high-frequency support."""


def test_line_breaks():

    print("")
    print(break_text(_test_line_breaks_text, 80))
