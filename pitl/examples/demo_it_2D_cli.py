import numpy as np
from napari.util import app_context
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from pitl.features.mcfocl import MultiscaleConvolutionalFeatures
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.regression.gbm import GBMRegressor


def demo():
    pass

#TODO: do something here...