import numpy as np
from fitl.old.multiscale_convolutions import MultiscaleConvolutionalFeatures
from skimage.data import camera
from skimage.exposure import rescale_intensity

from src.pitl.features.multiscale_convolutions import MultiscaleConvolutionalFeatures


def demo_multiscale_convolutions_2d():
    image = camera().astype(np.float32)  # [0:3,0:3]
    # image = np.zeros((3,3))
    image[0, 0] = 1
    image[1, 1] = 1
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    msf = MultiscaleConvolutionalFeatures(exclude_center=True)

    features = msf.compute(image)

    print(image)
    print(features)
    print(features.shape)


def demo_multiscale_convolutions_3d():
    # image = np.random.rand(48, 773, 665)*0;
    image = np.ones((48, 773, 665), dtype=np.float32);

    msf = MultiscaleConvolutionalFeatures(exclude_center=True)

    features = msf.compute(image)

    print(image)
    print(features)
    print(features.shape)


demo_multiscale_convolutions_3d()
