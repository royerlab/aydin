from aydin.gui.tabs.qt.base_cropping import BaseCroppingTab


class DenoisingCroppingTab(BaseCroppingTab):
    """
    Cropping Image before denoising

    You may also want to restrict denoising to a region in the image. This is independent from the cropping done for
    training and auto-tuning. Use the sliders to select a region of the image to crop before denoising.
    <moreless>
    <split>
    """
